import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import torch
from gymnasium.wrappers import RecordVideo, TimeLimit
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fnn import FNN
from src.replay_buffer import ReplayBuffer
from src.utils import (
    device, get_bradley_terry_loss, get_multivariate_normal_size, get_tanh_multivariate_normal, polyak_update,
)

num_episodes = int(input("iterations: "))
image_dir = os.path.join("temp", "images", "ranking_rl_test")
video_dir = os.path.join("temp", "videos", "ranking_rl_test")
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
os.makedirs(image_dir)
os.makedirs(video_dir)

batch_size = 128
replay_capacity = 15000
gamma = 0.99
actor_lr = 3e-4
critic_lr = 3e-4
actor_polyak = 0.001
critic_polyak = 0.005
update_every_steps = 20
max_grad_norm = 0.5
alpha = 0.5
alpha_decay = 0.9999
alpha_min = 0.01
hidden_size = 256
num_hidden_layers = 5
print_interval = 100
save_interval = 20

env = gym.make("HalfCheetah-v5")
env = TimeLimit(env, max_episode_steps = 500)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy_param_dim = get_multivariate_normal_size(action_dim)

actor = FNN(
    input_size = state_dim,
    hidden_size = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size = policy_param_dim,
).to(device)
critic = FNN(
    input_size = state_dim + action_dim,
    hidden_size = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size = 1,
).to(device)
target_actor = FNN(
    input_size = state_dim,
    hidden_size = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size = policy_param_dim,
).to(device)
target_critic = FNN(
    input_size = state_dim + action_dim,
    hidden_size = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size = 1,
).to(device)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
for param in target_actor.parameters():
    param.requires_grad_(False)
for param in target_critic.parameters():
    param.requires_grad_(False)

actor_optimizer = optim.Adam(actor.parameters(), lr = actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr = critic_lr)
rng = np.random.default_rng(0)
replay_buffer = ReplayBuffer(
    capacity = replay_capacity,
    batch_size = batch_size,
    device = device,
    rng = rng,
)

iter_count = 0
update_count = 0
episode_rewards = []
for episode in trange(num_episodes):
    done = False
    episode_reward = 0
    state, _ = env.reset(seed = episode)

    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
            policy_params = actor(state_t)
            policy = get_tanh_multivariate_normal(policy_params, action_dim)
            action = policy.sample()

        next_state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        episode_reward += reward
        done = terminated or truncated
        replay_buffer.add((
            state_t,
            action,
            torch.tensor([[reward]], dtype = torch.float32),
            torch.tensor([[float(done)]], dtype = torch.float32),
            torch.tensor([next_state], dtype = torch.float32),
        ))

        if iter_count % update_every_steps == 0 and replay_buffer.ready():
            batch_state, batch_action, batch_reward, batch_done, batch_next_state = replay_buffer.sample()

            critic_optimizer.zero_grad(set_to_none = True)
            with torch.no_grad():
                next_policy_params = actor(batch_next_state)
                next_policy = get_tanh_multivariate_normal(next_policy_params, action_dim)
                next_action = next_policy.sample()
                next_q = target_critic(torch.cat([batch_next_state, next_action], dim = 1))
                target_q = batch_reward + gamma * (1 - batch_done) * next_q
            q = critic(torch.cat([batch_state, batch_action], dim = 1))
            critic_loss = mse_loss(q, target_q)
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optimizer.step()
            polyak_update(target_critic, critic, critic_polyak)

            actor_optimizer.zero_grad(set_to_none = True)
            policy_params = actor(batch_state)
            policy = get_tanh_multivariate_normal(policy_params, action_dim)
            action1 = policy.sample()
            action2 = policy.sample()
            log_prob1 = policy.log_prob(action1)
            log_prob2 = policy.log_prob(action2)
            with torch.no_grad():
                q1 = target_critic(torch.cat([batch_state, action1], dim = 1))
                q2 = target_critic(torch.cat([batch_state, action2], dim = 1))
            actor_loss, _ = get_bradley_terry_loss(q1, q2, log_prob1, log_prob2, c = 5e-3, T = .5)
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optimizer.step()
            polyak_update(target_actor, actor, actor_polyak)

            if update_count % print_interval == 0:
                print(f"{update_count}, al: {actor_loss.item():.8f}, cl: {critic_loss.item():.8f}, er: {episode_rewards[-1] if episode_rewards else 0:.8f}")
            alpha = max(alpha * alpha_decay, alpha_min)
            update_count += 1

        state = next_state
        iter_count += 1

    episode_rewards.append(episode_reward)

    if (episode + 1) % save_interval == 0:
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah-v5 Ranking RL")
        plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
        plt.close()

        video_env = gym.make("HalfCheetah-v5", render_mode = "rgb_array")
        video_env = TimeLimit(video_env, max_episode_steps = 500)
        video_env = RecordVideo(
            video_env,
            video_folder = video_dir,
            name_prefix = f"episode_{episode + 1}",
        )
        state_np, _ = video_env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            with torch.no_grad():
                state_v = torch.tensor(state_np, dtype = torch.float32, device = device).unsqueeze(0)
                policy_params_v = actor(state_v)
                policy_v = get_tanh_multivariate_normal(policy_params_v, action_dim)
                action_v = policy_v.sample()
            state_np, _, terminated, truncated, _ = video_env.step(action_v[0].cpu().numpy())
        video_env.close()
        print(f"\nCheck {image_dir} and {video_dir} for saved images and videos.")

plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("HalfCheetah-v5 Ranking RL")
plt.savefig(os.path.join(image_dir, "rewards_final.png"))
plt.close()

print("training done", len(episode_rewards), episode_rewards[-1])
env.close()
