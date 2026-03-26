import sys
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit, RecordVideo
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modular_network import ModularNetwork
from src.replay_buffer import ReplayBuffer
from src.utils import device

# --- input ---
num_episodes = int(input("iterations: "))

image_dir = os.path.join("temp", "images", "modular_ddpg_test")
video_dir = os.path.join("temp", "videos", "modular_ddpg_test")
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
if os.path.exists(video_dir):
    shutil.rmtree(video_dir)
os.makedirs(image_dir)
os.makedirs(video_dir)

# --- env ---
env = gym.make("HalfCheetah-v5")
env = TimeLimit(env, max_episode_steps=100)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# --- hyperparams ---
num_encoders        = 2
tree_depth          = 2
fan_in              = 2
embed_dim           = 128
num_hidden_layers   = 4
discount_rate       = 0.97
learning_rate       = 3e-4
critic_polyak       = 0.05
actor_polyak        = 0.05
batch_size          = 256
replay_buffer_size  = 50000
noise_std           = 0.1
noise_std_min       = 0.0001
noise_std_decay     = 0.9998
update_interval     = 7
print_interval      = 200
save_interval       = 50

rng = np.random.default_rng(0)

# --- networks ---
actor = ModularNetwork(
    input_dim         = state_dim,
    embed_dim         = embed_dim,
    num_hidden_layers = num_hidden_layers,
    num_encoders      = num_encoders,
    tree_depth        = tree_depth,
    fan_in            = fan_in,
    output_dim        = action_dim,
    output_activation = nn.Tanh(),
    rng               = rng,
).to(device)

critic = ModularNetwork(
    input_dim         = state_dim + action_dim,
    embed_dim         = embed_dim,
    num_hidden_layers = num_hidden_layers,
    num_encoders      = num_encoders,
    tree_depth        = tree_depth,
    fan_in            = fan_in,
    output_dim        = 1,
    rng               = rng,
).to(device)

actor_optimizer  = optim.Adam(actor.parameters(),  lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# --- replay buffer ---
replay          = ReplayBuffer(capacity=replay_buffer_size, batch_size=batch_size, rng=rng)
episode_rewards = []
train_updates   = 0
env_steps       = 0

for episode in trange(num_episodes):
    state_np, _ = env.reset(seed=episode)
    state        = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.0
    terminated   = False
    truncated    = False

    while not terminated and not truncated:
        with torch.no_grad():
            actor_module_id = int(rng.integers(0, len(actor._online_modules)))
            action          = actor(state, online_module_id=actor_module_id)
        action = action + torch.randn_like(action) * noise_std
        action = action.clamp(min=-2.0, max=2.0)

        next_state_np, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
        next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
        reward_t   = torch.tensor([[reward]],                       dtype=torch.float32, device=device)
        done_t     = torch.tensor([[float(terminated or truncated)]], dtype=torch.float32, device=device)

        replay.add((state, action, reward_t, done_t, next_state))

        state          = next_state
        episode_reward += float(reward)
        env_steps      += 1

        if replay.ready() and env_steps % update_interval == 0:
            batch_state, batch_action, batch_reward, batch_done, batch_next_state = replay.sample()

            # critic update
            critic_module_idx = int(rng.integers(0, len(critic._online_modules)))
            state_action      = torch.cat([batch_state, batch_action], dim=1)
            q                 = critic(state_action, online_module_id=critic_module_idx)
            with torch.no_grad():
                next_action       = actor(batch_next_state)
                next_state_action = torch.cat([batch_next_state, next_action], dim=1)
                next_q            = critic(next_state_action)
                target_q          = batch_reward + (1.0 - batch_done) * discount_rate * next_q

            critic_loss = mse_loss(q, target_q)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            clip_grad_norm_(critic._online_modules[critic_module_idx].parameters(), max_norm=1.0)
            critic_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            critic.polyak_update(module_id=critic_module_idx, polyak_factor=critic_polyak)

            # actor update
            for _ in range(2):
                actor_module_idx = int(rng.integers(0, len(actor._online_modules)))
                live_encoder_id  = int(rng.integers(0, num_encoders))
                actor_action     = actor(batch_state, online_module_id=actor_module_idx)
                state_action     = torch.cat([batch_state, actor_action], dim=1)
                actor_q          = critic(state_action, live_encoder_id=live_encoder_id)
                actor_loss       = -actor_q.mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                clip_grad_norm_(actor._online_modules[actor_module_idx].parameters(), max_norm=1.0)
                actor_optimizer.step()
                actor_optimizer.zero_grad(set_to_none=True)
                actor.polyak_update(module_id=actor_module_idx, polyak_factor=actor_polyak)

            train_updates += 1
            if train_updates % print_interval == 0:
                print(f"{train_updates}, cl: {critic_loss.item():.6f}, al: {actor_loss.item():.6f}, tq: {target_q.mean().item():.6f}, ns: {noise_std:.6f}, re: {episode_rewards[-1] if episode_rewards else 0:.6f}")

            noise_std = max(noise_std_min, noise_std * noise_std_decay)

    episode_rewards.append(episode_reward)

    if (episode + 1) % save_interval == 0:
        plt.figure()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("HalfCheetah-v5 Modular Critic")
        plt.savefig(os.path.join(image_dir, f"rewards_{episode + 1}.png"))
        plt.close()

        video_env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
        video_env = TimeLimit(video_env, max_episode_steps=100)
        video_env = RecordVideo(video_env, video_folder=video_dir, name_prefix=f"episode_{episode + 1}")
        state_np, _ = video_env.reset()
        state        = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        terminated   = truncated = False
        while not terminated and not truncated:
            with torch.no_grad():
                action = actor(state).clamp(min=-2.0, max=2.0)
            state_np, _, terminated, truncated, _ = video_env.step(action[0].cpu().numpy())
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        video_env.close()

        print(f"\nCheck {image_dir} and {video_dir} for saved images and videos.")

# --- final plot ---
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("HalfCheetah-v5 Modular Critic")
plt.savefig(os.path.join(image_dir, "rewards_final.png"))
plt.close()

env.close()