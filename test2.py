from fnn import FNN
import random
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torch.nn import CosineSimilarity
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from typing import TypeVar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar("T", float, Tensor)


def get_ema(ema: T, data: T, factor: float) -> T:
    return data * factor + ema * (1 - factor)


def get_ema_and_emv(ema: T, emv: T, data: T, factor: float) -> tuple[T, T]:
    var = (data - ema) ** 2
    return get_ema(ema, data, factor), get_ema(emv, var, factor)


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(get_ema(target_param.data, online_param.data, polyak_factor))


def test_predict_3norm() -> None:
    vec_dim = 4
    batch = 128
    model = FNN(
        input_size = vec_dim * 2,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = 2,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr = 1e-3)

    for i in trange(10000, desc = "predict 3-norm"):
        vec1 = torch.rand(batch, vec_dim, device = device)
        vec2 = torch.rand(batch, vec_dim, device = device)
        x = torch.cat([vec1, vec2], dim = -1)
        target_3 = torch.norm(vec1 - vec2, dim = -1, p = 3)
        target_2 = torch.norm(100 * vec1 - 111 * vec2, dim = -1, p = 2)
        pred = model(x).squeeze(-1)
        loss_3 = mse_loss(pred[:, 0], target_3)
        loss_2 = mse_loss(pred[:, 1], target_2)
        loss = loss_3 + loss_2
        opt.zero_grad(set_to_none = True)
        loss.backward()
        opt.step()

        if i % 1000 == 0:
            print(f"step: {i}, loss_3: {loss_3.item()}, loss_2: {loss_2.item()}")


def test_output_vector_two_conditions() -> None:
    vec_dim = 8
    batch = 128
    cos_sim = CosineSimilarity(dim = -1)
    model = FNN(
        input_size = vec_dim * 2,
        hidden_size = 32,
        num_hidden_layers = 4,
        output_size = vec_dim * 2,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr = 1e-3)

    all_params = list(model.parameters())
    for i in trange(10000, desc = "output vector two conditions"):
        vec1 = torch.rand(batch, vec_dim, device = device)
        vec2 = torch.rand(batch, vec_dim, device = device)
        x = torch.cat([vec1, vec2], dim = -1)
        out = model(x)
        out1, out2 = out[:, :vec_dim], out[:, vec_dim:]
        loss_full = (1 - cos_sim(out, x)).mean()
        loss_halves = 10 * (cos_sim(out1, vec1).mean() - cos_sim(out2, vec2).mean())

        opt.zero_grad(set_to_none = True)
        loss_full.backward(retain_graph = True)
        grad_full = [p.grad.clone() if p.grad is not None else None for p in all_params]
        opt.zero_grad(set_to_none = True)
        loss_halves.backward()
        grad_halves = [p.grad.clone() if p.grad is not None else None for p in all_params]
        opt.zero_grad(set_to_none = True)

        g_full = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_full)
        ])
        g_halves = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_halves)
        ])
        eps = 1e-8
        g1, g2 = (g_full, g_halves) if random.random() < 0.5 else (g_halves, g_full)
        dot = (g1 * g2).sum()
        if dot < 0:
            g2 = g2 - dot * g1 / (g1.pow(2).sum() + eps)
        g_total = g1 + g2
        offset = 0
        for p in all_params:
            n = p.numel()
            p.grad = g_total[offset:offset + n].view_as(p).clone()
            offset += n
        clip_grad_norm_(model.parameters(), max_norm = 1.0)
        opt.step()

        if i % 500 == 0:
            print(f"step: {i}, loss_full: {loss_full.item()}, loss_halves: {loss_halves.item()}")


def test_ddpg() -> None:
    state_dim = 4
    actor = FNN(
        input_size = state_dim,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = state_dim,
    ).to(device)
    actor_target = FNN(
        input_size = state_dim,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = state_dim,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())
    for param in actor_target.parameters():
        param.requires_grad_(False)
    critic = FNN(
        input_size = state_dim * 2,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = 1,
    ).to(device)
    critic_target = FNN(
        input_size = state_dim * 2,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = 1,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for param in critic_target.parameters():
        param.requires_grad_(False)
    actor_opt = optim.Adam(actor.parameters(), lr = 1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr = 1e-3)

    for i in trange(20000, desc = "DDPG"):
        state = torch.rand(128, state_dim, device = device)
        with torch.no_grad():
            action = actor_target(state)
            target = torch.norm(state - action, dim = -1)
        pred = critic(torch.cat([state, action], dim = -1)).squeeze(-1)
        critic_loss = mse_loss(pred, target)
        critic_opt.zero_grad(set_to_none = True)
        critic_loss.backward()
        critic_opt.step()
        polyak_update(critic_target, critic, 0.01)

        state = torch.rand(128, state_dim, device = device)
        action = actor(state)
        q_pred = critic_target(torch.cat([state, action], dim = -1)).squeeze(-1)
        actor_loss = q_pred.mean()
        actor_opt.zero_grad(set_to_none = True)
        actor_loss.backward()
        actor_opt.step()
        polyak_update(actor_target, actor, 0.01)

        if i % 500 == 0:
            with torch.no_grad():
                state = torch.rand(1024, state_dim, device = device)
                action = actor_target(state)
                target = torch.norm(state - action, dim = -1)
                pred = critic(torch.cat([state, action], dim = -1)).squeeze(-1)
                mse = mse_loss(pred, target)
                print(f"step: {i}, mse: {mse.item()}")

    assert mse < 1e-3


if __name__ == "__main__":
    test_output_vector_two_conditions()