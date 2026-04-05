import sys
import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, TimeLimit
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torchvision.io import video
from tqdm import trange
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fnn import FNN
from src.replay_buffer import ReplayBuffer
from src.utils import device, polyak_update

# --- setup ---
num_episodes = int(input("iterations: "))
image_dir    = os.path.join("temp", "images", "ddpg_test")
video_dir    = video_folder=os.path.join("temp", "videos", "ddpg_test")
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
os.makedirs(image_dir)

# --- hyperparams ---
batch_size           = 256
replay_capacity      = 100000
gamma                = 0.99
actor_lr             = 3e-4
critic_lr            = 3e-4
actor_polyak         = 0.005
critic_polyak        = 0.005
actor_update_interval = 1
update_every_steps   = 10
noise_std            = 0.05
noise_std_min        = 0.0001
noise_std_decay      = 0.9999
hidden_size          = 256
num_hidden_layers    = 3
print_interval       = 200
save_interval        = 10

# --- env ---
env = gym.make("HalfCheetah-v5")
env = TimeLimit(env, max_episode_steps=500)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low  = torch.tensor(env.action_space.low,  dtype=torch.float32, device=device)
action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

# --- networks ---
actor = FNN(
    input_size         = state_dim,
    hidden_size        = hidden_size,
    num_hidden_layers  = num_hidden_layers,
    output_size        = action_dim,
    output_activation  = nn.Tanh(),
).to(device)
critic = FNN(
    input_size        = state_dim + action_dim,
    hidden_size       = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size       = 1,
).to(device)
actor_target = FNN(
    input_size        = state_dim,
    hidden_size       = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size       = action_dim,
    output_activation = nn.Tanh(),
).to(device)
critic_target = FNN(
    input_size        = state_dim + action_dim,
    hidden_size       = hidden_size,
    num_hidden_layers = num_hidden_layers,
    output_size       = 1,
).to(device)

for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)
for param in actor_target.parameters():
    param.requires_grad_(False)
for param in critic_target.parameters():
    param.requires_grad_(False)

actor_optimizer  = optim.Adam(actor.parameters(),  lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# --- replay buffer ---
rng           = np.random.default_rng(0)
replay_buffer = ReplayBuffer(capacity=replay_capacity, batch_size=batch_size, device=device, rng=rng)

# --- training ---
episode_rewards = []
train_updates   = 0
env_steps       = 0

for episode in trange(num_episodes):
    state_np, _ = env.reset(seed=episode)
    state        = torch.tensor(state_np, dtype=torch.float32, device=device)
    episode_reward = 0.0
    terminated   = False
    truncated    = False

    while not terminated and not truncated:
        with torch.no_grad():
            action = actor(state.unsqueeze(0))[0]
        action = action + torch.randn_like(action) * noise_std
        action = action.clamp(min=action_low, max=action_high)

        next_state_np, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done       = bool(terminated or truncated)
        next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device)

        replay_buffer.add((state, action, torch.tensor([reward], dtype=torch.float32, device=device), torch.tensor([float(done)], dtype=torch.float32, device=device), next_state))

        state          = next_state
        episode_reward += float(reward)
        env_steps      += 1

        if replay_buffer.ready() and env_steps % update_every_steps == 0:
            batch_state, batch_action, batch_reward, batch_done, batch_next_state = replay_buffer.sample()

            with torch.no_grad():
                next_action       = actor_target(batch_next_state)
                next_state_action = torch.cat([batch_next_state, next_action], dim=1)
                next_q            = critic_target(next_state_action)
                target_q          = batch_reward + (1.0 - batch_done) * gamma * next_q

            state_action = torch.cat([batch_state, batch_action], dim=1)
            q            = critic(state_action)
            critic_loss  = mse_loss(q, target_q)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            polyak_update(critic_target, critic, critic_polyak)

            actor_loss = torch.tensor(float("nan"), device=device)
            if train_updates % actor_update_interval == 0:
                actor_action       = actor(batch_state)
                actor_state_action = torch.cat([batch_state, actor_action], dim=1)
                actor_q            = critic(actor_state_action)
                actor_loss         = -actor_q.mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad(set_to_none=True)
                polyak_update(actor_target, actor, actor_polyak)

            train_updates += 1
            if train_updates % print_interval == 0:
                print(f"{train_updates}, cl: {critic_loss.item():.6f}, al: {actor_loss.item():.6f}, tq: {target_q.mean().item():.6f}, ns: {noise_std:.6f}, re: {episode_rewards[-1] if episode_rewards else 0.0:.6f}")

            noise_std = max(noise_std_min, noise_std * noise_std_decay)

    episode_rewards.append(episode_reward)

    if (episode + 1) % save_interval == 0:
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah-v5")
        plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
        plt.close()

        video_env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
        video_env = TimeLimit(video_env, max_episode_steps=500)
        video_env = RecordVideo(
            video_env,
            video_folder=video_dir,
            name_prefix=f"episode_{episode + 1}"
        )
        state_np, _ = video_env.reset()
        state = torch.tensor(state_np, dtype=torch.float32, device=device)
        terminated = truncated = False
        while not terminated and not truncated:
            with torch.no_grad():
                action = actor(state.unsqueeze(0))[0].clamp(min=action_low, max=action_high)
            state_np, _, terminated, truncated, _ = video_env.step(action.cpu().numpy())
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
        video_env.close()

        print(f"\nCheck {image_dir} and {video_dir} for saved images and videos.")

# --- final plot ---
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("HalfCheetah-v5")
plt.savefig(os.path.join(image_dir, "rewards_final.png"))
plt.close()

print("training done", len(episode_rewards), episode_rewards[-1])
env.close()