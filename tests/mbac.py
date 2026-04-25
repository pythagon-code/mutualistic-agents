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
video_dir = os.path.join("temp", "videos", folder_name)
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
os.makedirs(video_dir)

num_episodes = 1000000
batch_size = 128
replay_capacity = 100000
gamma = 0.99
actor_lr = 3e-4
critic_lr = 3e-4
actor_polyak = 0.005
critic_polyak = 0.005
actor_update_interval = 2
update_every_steps = 20
max_grad_norm = 0.5
noise_std = .05
noise_std_min = 1e-4
noise_std_decay = .9996
save_interval = 100

env = gym.make("HalfCheetah-v5")
env = TimeLimit(env, max_episode_steps = 500)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = FNN(
    input_size = state_dim,
    hidden_size = 256,
    num_hidden_layers = 5,
    output_size = action_dim,
    output_activation = nn.Tanh(),
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
            action = actor(state)
            action = action + torch.randn_like(action) * noise_std

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
                next_action = actor(batch_next_state)
                next_action = next_action + torch.randn_like(next_action) * noise_std
                next_q1 = target_critic1(torch.cat([batch_next_state, next_action], dim = 1))
                next_q2 = target_critic2(torch.cat([batch_next_state, next_action], dim = 1))
                next_q = torch.min(next_q1, next_q2)
                target_q = batch_reward + gamma * (1 - batch_done) * next_q
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
            
            if update_count % actor_update_interval == 0:
                actor_optimizer.zero_grad(set_to_none = True)
                action = actor(batch_state)
                q1 = target_critic1(torch.cat([batch_state, action], dim = 1))
                q2 = target_critic2(torch.cat([batch_state, action], dim = 1))
                q = torch.min(q1, q2)
                actor_loss = -q.mean()
                actor_loss.backward()
                clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizer.step()

            if update_count % 100 == 0:
                print(f"\n{update_count}, al: {actor_loss.item():.8f}, cl: {critic1_loss.item():.8f}, er: {episode_rewards[-1] if episode_rewards else 0:.8f}")
            update_count += 1
        state = next_state
        iter_count += 1

    if (episode + 1) % save_interval == 0:
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah-v5 TD3")
        plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
        plt.close()

        env_recording = RecordVideo(
            gym.make("HalfCheetah-v5", max_episode_steps = 500, render_mode = "rgb_array"),
            video_folder = video_dir,
            name_prefix = f"video_{episode + 1}",
        )
        with torch.no_grad():
            state, _ = env_recording.reset(seed = episode)
            done = False
            while not done:
                state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
                action = actor(state_t)
                state, _, truncated, terminated, _ = env_recording.step(action[0].cpu().numpy())
                done = truncated or terminated
        env_recording.close()
        print("\nDone saving.")
    
    episode_rewards.append(episode_reward)