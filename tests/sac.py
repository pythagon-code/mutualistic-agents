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

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(__file__), ".."), ".."))
from src.fnn import FNN
from src.replay_buffer import ReplayBuffer
from src.utils import (
    device, get_bradley_terry_loss, get_multivariate_normal_size, get_tanh_multivariate_normal, polyak_update,
)

inputs = input("Enter iterations (int), c (float), T (float), folder name (str) in a single row: ")
num_episodes, c, T, folder_name = inputs.split()
num_episodes = int(num_episodes)
c = float(c)
T = float(T)
image_dir = os.path.join("temp", "images", folder_name)
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

num_episodes = 1000000
batch_size = 128
replay_capacity = 100000
gamma = 0.99
actor_lr = 3e-4
critic_lr = 3e-4
actor_polyak = 0.005
critic_polyak = 0.005
actor_update_interval = 1
update_every_steps = 20
max_grad_norm = 0.5
alpha = 0.002
alpha_decay = 0.9995
alpha_min = 0.0001
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
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)

critic1 = FNN(
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)
critic2 = FNN(
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)

target_actor = FNN(
    input_size = state_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = action_dim + action_dim * (action_dim + 1) // 2,
).to(device)

target_critic = FNN(
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)

target_critic1 = FNN(
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)
target_critic2 = FNN(
    input_size = state_dim + action_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = 1,
).to(device)

target_actor.load_state_dict(actor.state_dict())
for param in target_actor.parameters():
    param.requires_grad_(False)
target_critic.load_state_dict(critic.state_dict())
for param in target_critic.parameters():
    param.requires_grad_(False)
target_critic1.load_state_dict(critic1.state_dict())
for param in target_critic1.parameters():
    param.requires_grad_(False)
target_critic2.load_state_dict(critic2.state_dict())
for param in target_critic2.parameters():
    param.requires_grad_(False)

actor_optimizer = optim.Adam(actor.parameters(), lr = actor_lr)
critic1_optimizer = optim.Adam(critic1.parameters(), lr = critic_lr)
critic2_optimizer = optim.Adam(critic2.parameters(), lr = critic_lr)
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
    state, _ = env.reset()

    while not done:
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
            policy_params = actor(state)
            policy = get_tanh_multivariate_normal(policy_params, action_dim)
            action = policy.sample()

        next_state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        episode_reward += reward
        done = terminated or truncated
        replay_buffer.add((
            state,
            action,
            torch.tensor([[reward]], dtype = torch.float32),
            torch.tensor([[done]], dtype = torch.int32),
            torch.tensor([next_state], dtype = torch.float32),
        ))

        if iter_count % update_every_steps == 0 and replay_buffer.ready():
            batch_state, batch_action, batch_reward, batch_done, batch_next_state = replay_buffer.sample()

            critic1_optimizer.zero_grad(set_to_none = True)
            critic2_optimizer.zero_grad(set_to_none = True)
            with torch.no_grad():
                next_policy_params = actor(batch_next_state)
                next_policy = get_tanh_multivariate_normal(next_policy_params, action_dim)
                next_action = next_policy.sample()
                next_log_prob = next_policy.log_prob(next_action).unsqueeze(1)
                next_q1 = target_critic1(torch.cat([batch_next_state, next_action], dim = 1))
                next_q2 = target_critic2(torch.cat([batch_next_state, next_action], dim = 1))
                next_q = torch.min(next_q1, next_q2)
                target_q = batch_reward + gamma * (1 - batch_done) * (next_q - alpha * next_log_prob)
            q = critic1(torch.cat([batch_state, batch_action], dim = 1))
            critic1_loss = mse_loss(q, target_q)
            critic1_loss.backward()
            clip_grad_norm_(critic1.parameters(), max_grad_norm)
            critic1_optimizer.step()
            polyak_update(target_critic1, critic1, critic_polyak)
            q = critic2(torch.cat([batch_state, batch_action], dim = 1))
            critic2_loss = mse_loss(q, target_q)
            critic2_loss.backward()
            clip_grad_norm_(critic2.parameters(), max_grad_norm)
            critic2_optimizer.step()
            polyak_update(target_critic2, critic2, critic_polyak)
            
            actor_optimizer.zero_grad(set_to_none = True)
            policy_params = actor(batch_state)
            policy = get_tanh_multivariate_normal(policy_params, action_dim)
            action = policy.rsample()
            log_prob = policy.log_prob(action).unsqueeze(1)
            q1 = target_critic1(torch.cat([batch_state, action], dim = 1))
            q2 = target_critic2(torch.cat([batch_state, action], dim = 1))
            q = torch.min(q1, q2)
            actor_loss = (alpha * log_prob - q).mean()
            if actor_loss.isnan().any():
                print("\nnans produced")
            else:
                actor_loss.backward()
                clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizer.step()
                polyak_update(target_actor, actor, actor_polyak)

            if update_count % 100 == 0:
                print(f"\n{update_count}, al: {actor_loss.item():.8f}, cl: {critic1_loss.item():.8f}, er: {episode_rewards[-1] if episode_rewards else 0:.8f}, a: {alpha:.8f}")
            update_count += 1
            alpha = max(alpha * alpha_decay, alpha_min)
        state = next_state
        iter_count += 1

        if (episode + 1) % save_interval == 0:
            plt.figure()
            plt.plot(episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.title("HalfCheetah-v5 Ranking RL")
            plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
            plt.close()
    
    episode_rewards.append(episode_reward)