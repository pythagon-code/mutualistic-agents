import sys
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modular_network import ModularNetwork
from src.utils import device, polyak_update

# --- input ---
steps = int(input("iterations: "))

image_dir = os.path.join("temp", "images", "modular_norm_test")
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

# --- hyperparams ---
input_dim         = 16
batch_size        = 512
p                 = 2.5
print_interval    = 200
save_interval     = 2000
num_encoders      = 4
tree_depth        = 3
fan_in            = 2
embed_dim         = 128
num_hidden_layers = 5
output_dim        = 1
lr                = 3e-4
polyak_factor     = 0.005

rng = np.random.default_rng(0)

# --- network ---
actor = ModularNetwork(
    input_dim         = input_dim,
    embed_dim         = embed_dim,
    num_hidden_layers = num_hidden_layers,
    num_encoders      = num_encoders,
    tree_depth        = tree_depth,
    fan_in            = fan_in,
    output_dim        = output_dim,
    rng               = rng,
).to(device)

optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

# --- training ---
loss_history = []

for step in trange(steps):
    state  = torch.rand((batch_size, input_dim), dtype=torch.float32, device=device)
    target = state.norm(p=p, dim=-1, keepdim=True)

    module_idx = int(rng.integers(0, len(actor._online_modules)))
    output     = actor(state, online_module_id=module_idx)
    loss       = mse_loss(output, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(actor._online_modules[module_idx].parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    actor.polyak_update(module_id=module_idx, polyak_factor=polyak_factor)

    if step % print_interval == 0:
        loss_history.append(loss.item())
        print(f"{step}, loss: {loss.item():.6f}")

    if (step + 1) % save_interval == 0:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel(f"Step (x{print_interval})")
        plt.ylabel("Loss")
        plt.title("Modular Network Norm Regression")
        plt.savefig(os.path.join(image_dir, f"loss_{step + 1}.png"))
        plt.close()

# --- final plot ---
plt.figure()
plt.plot(loss_history)
plt.xlabel(f"Step (x{print_interval})")
plt.ylabel("Loss")
plt.title("Modular Network Norm Regression")
plt.savefig(os.path.join(image_dir, "loss_final.png"))
plt.close()