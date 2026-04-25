import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import torch
from gymnasium.wrappers import TimeLimit
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from time import sleep

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fnn import FNN
from src.replay_buffer import ReplayBuffer
from src.utils import (
    device, get_bradley_terry_loss, get_multivariate_normal_size, get_tanh_multivariate_normal, polyak_update,
)

if len(sys.argv) != 5:
    print("Usage: python script.py <num_episodes> <c> <T> <folder_name>")
    sys.exit(1)

num_episodes = int(sys.argv[1])
c = float(sys.argv[2])
T = float(sys.argv[3])
folder_name = sys.argv[4]
image_dir = os.path.join("temp", "images", folder_name)
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

num_episodes = 1000000
batch_size = 128
replay_capacity = 100000
gamma = 0.99
lambd = 0.95
actor_lr = 3e-4
critic_lr = 3e-4
actor_polyak = 0.005
critic_polyak = 0.005
actor_update_interval = 1
update_every_steps = 20
max_grad_norm = 0.5
rollout_buffer_capacity = 500
epsilon = .2
num_epochs = 4
save_interval = 500

env = gym.make("HalfCheetah-v5")
env = TimeLimit(env, max_episode_steps = 500)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy_param_dim = get_multivariate_normal_size(action_dim)

actor = FNN(
    input_size = state_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = action_dim + action_dim * (action_dim + 1) // 2,
).to(device)
critic = FNN(
    input_size = state_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr = actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr = critic_lr)
rng = np.random.default_rng(0)

def get_gae_and_return(b_state, b_reward, b_done, b_next_state):
    with torch.no_grad():
        b_value = critic(b_state)
        b_next_value = critic(b_next_state)
        b_gae = []
        b_return = []
        prev_gae = 0
        prev_return = critic(b_next_state[-1])
        for i in range(b_state.shape[0] - 1, -1, -1):
            delta = b_reward[i] + gamma * (1 - b_done[i]) * b_next_value[i] - b_value[i]
            gae = delta + gamma * lambd * (1 - b_done[i]) * prev_gae
            prev_gae = gae
            retrn = b_reward[i] + gamma * (1 - b_done[i]) * prev_return
            prev_return = retrn
            b_gae.append(gae)
            b_return.append(retrn)
        b_gae.reverse()
        b_return.reverse()
        return torch.stack(b_gae), torch.stack(b_return)

rollout_buffer = []

iter_count = 0
update_count = 0
episode_rewards = []
for episode in trange(num_episodes):
    done = False
    episode_reward = 0
    state, _ = env.reset()

    while not done:
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
            policy_params = actor(state)
            if policy_params.isnan().any():
                action = torch.zeros((1, action_dim), dtype = torch.float32, device = device)
                log_prob = torch.zeros((1, 1), dtype = torch.float32, device = device)
                print("\nnans produced")
            else:
                policy = get_tanh_multivariate_normal(policy_params, action_dim)
                action = policy.sample()
                log_prob = policy.log_prob(action).unsqueeze(0).nan_to_num(nan = 0).clamp(min = -10, max = 10)
            value = critic(state)

        next_state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        episode_reward += reward
        done = terminated or truncated
        rollout_buffer.append((
            state.cpu(),
            action.cpu(),
            log_prob.cpu(),
            value.cpu(),
            torch.tensor([[reward]], dtype = torch.float32),
            torch.tensor([[done]], dtype = torch.int32),
            torch.tensor([next_state], dtype = torch.float32),
        ))

        state = next_state

        iter_count += 1

        if len(rollout_buffer) == rollout_buffer_capacity:
            b_state, b_action, b_log_prob, b_value, b_reward, b_done, b_next_state = zip(*rollout_buffer)
            b_state = torch.cat(b_state).to(device)
            b_action = torch.cat(b_action).to(device)
            b_reward = torch.cat(b_reward).to(device)
            b_done = torch.cat(b_done).to(device)
            b_next_state = torch.cat(b_next_state).to(device)
            b_log_prob = torch.cat(b_log_prob).to(device).nan_to_num(nan = 0).clamp(min = -10, max = 10)
            b_gae, b_return = get_gae_and_return(b_state, b_reward, b_done, b_next_state)

            random_idx = torch.randperm(b_state.shape[0], device = device)
            b_state = b_state[random_idx]
            b_action = b_action[random_idx]
            b_done = b_done[random_idx]
            b_next_state = b_next_state[random_idx]
            b_log_prob = b_log_prob[random_idx]
            b_gae = b_gae[random_idx]
            b_return = b_return[random_idx]

            for i in range(num_epochs):
                start_idx = b_state.shape[0] // num_epochs * i
                end_idx = b_state.shape[0] // num_epochs * (i + 1)
                mb_state = b_state[start_idx : end_idx]
                mb_action = b_action[start_idx : end_idx]
                mb_log_prob = b_log_prob[start_idx : end_idx]
                mb_gae = b_gae[start_idx : end_idx]
                mb_return = b_return[start_idx : end_idx]

                critic_optimizer.zero_grad(set_to_none = True)
                value = critic(mb_state)
                critic_loss = mse_loss(value, mb_return)
                critic_loss.backward()
                clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
                critic_optimizer.step()

                actor_optimizer.zero_grad(set_to_none = True)
                policy_params = actor(mb_state)
                if policy_params.isnan().any():
                    print("\nnans produced")
                else:
                    policy = get_tanh_multivariate_normal(policy_params, action_dim)
                    new_log_prob = policy.log_prob(mb_action).unsqueeze(1).nan_to_num(nan = 0).clamp(min = -10, max = 10)
                    with torch.no_grad():
                        prob_ratio = (new_log_prob - mb_log_prob).nan_to_num(nan = 0).clamp(min = -10, max = 10).exp()
                        prob_ratio_clipped = prob_ratio.nan_to_num(nan = 1).clamp(min = 1 - epsilon, max = 1 + epsilon)
                    gae = torch.min(mb_gae * prob_ratio, mb_gae * prob_ratio_clipped)
                    actor_loss = -(new_log_prob * (gae - gae.mean())/(gae.std()+1e-7)).mean().nan_to_num(nan = 0)
                    actor_loss.backward()
                    clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
                    actor_optimizer.step()

            rollout_buffer.clear()
            if update_count % 1 == 0:
                print(f"\n{update_count}, al: {actor_loss.item():.8f}, cl: {critic_loss.item():.8f}, er: {episode_rewards[-1] if episode_rewards else 0:.8f}")
            update_count += 1

    if (episode + 1) % save_interval == 0:
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah-v5 SAC")
        plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
        plt.close()
    
    episode_rewards.append(episode_reward)