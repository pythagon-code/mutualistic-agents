from torch.nn.parameter import Parameter
from fnn import FNN
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
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


def sample_action_and_logprob(params: Tensor) -> tuple[Tensor, Tensor]:
    mean = params[..., 0:1]
    log_std = params[..., 1:2].clamp(min = -4.0, max = 1.0)
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    sample = dist.rsample()
    log_prob = dist.log_prob(sample)
    return sample, log_prob


def test_a2c_mutualism():
    # Mid Actor: State -> Embedding (Deterministic)
    # Final Actor: Embedding -> (Mean, Log_Std) for Sampling
    embed_dim = 8
    mid_actor = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = embed_dim,
    ).to(device)
    final_actor = FNN(
        input_size = embed_dim,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 2,
    ).to(device)

    # Targets to break feedback loops
    mid_target = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = embed_dim,
    ).to(device)
    final_target = FNN(
        input_size = embed_dim,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 2,
    ).to(device)

    mid_target.load_state_dict(mid_actor.state_dict())
    final_target.load_state_dict(final_actor.state_dict())

    mid_opt = optim.Adam(mid_actor.parameters(), lr = 1e-3)
    final_opt = optim.Adam(final_actor.parameters(), lr = 1e-3)

    for i in trange(500000, desc="A2C mutualism"):
        state = torch.rand(512, 1, device = device)
        target = state / torch.pi

        # --- 1. Train Mid Actor through the Final Policy ---
        # We use the Target Final Actor to provide a stable gradient signal
        embedding = mid_actor(state)
        out_final = final_target(embedding)
        sample, log_prob = sample_action_and_logprob(out_final)

        reward = -((sample - target) ** 2).squeeze(-1)
        centered = reward - reward.mean()
        std = reward.std(unbiased = False) + 1e-8
        advantage = (centered / std).detach()

        mid_loss = -(advantage * log_prob.squeeze(-1)).mean()
        mid_opt.zero_grad(set_to_none=True)
        mid_loss.backward()
        mid_opt.step()

        # --- 2. Train Final Actor using the Mid Embedding ---
        # We use the Target Mid Actor to provide a stable input feature
        with torch.no_grad():
            frozen_embed = mid_target(state)

        out_final = final_actor(frozen_embed)
        sample, log_prob = sample_action_and_logprob(out_final)

        reward = -((sample - target) ** 2).squeeze(-1)
        centered = reward - reward.mean()
        std = reward.std(unbiased = False) + 1e-8
        advantage = (centered / std).detach()

        final_loss = -(advantage * log_prob.squeeze(-1)).mean()
        final_opt.zero_grad(set_to_none=True)
        final_loss.backward()
        final_opt.step()

        # --- 3. Synchronize ---
        polyak_update(mid_target, mid_actor, 0.05)
        polyak_update(final_target, final_actor, 0.05)

        if i % 500 == 0:
            with torch.no_grad():
                test_embed = mid_actor(state)
                test_mean = final_actor(test_embed)[..., 0:1]
                mse = mse_loss(test_mean, target)
            print(f"step: {i}, mse: {mse.item()}")

    # Final check: Deterministic evaluation
    with torch.no_grad():
        test_embed = mid_actor(state)
        test_mean = final_actor(test_embed)[..., 0:1]
        final_mse = mse_loss(test_mean, target)

    assert final_mse < 1e-3


def test_mutualism_prediction() -> None:
    embed_dim = 8
    mid_actor = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = embed_dim,
    ).to(device)
    final_actor = FNN(
        input_size = embed_dim,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 1,
    ).to(device)
    mid_target = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = embed_dim,
    ).to(device)
    final_target = FNN(
        input_size = embed_dim,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 1,
    ).to(device)

    mid_target.load_state_dict(mid_actor.state_dict())
    final_target.load_state_dict(final_actor.state_dict())

    mid_opt = optim.Adam(mid_actor.parameters(), lr = 1e-3)
    final_opt = optim.Adam(final_actor.parameters(), lr = 1e-3)

    for i in trange(20000, desc = "mutualism prediction"):
        state = torch.rand(64, 1, device = device)
        target = (state / torch.pi).detach()

        # 1) Train mid_actor using fixed final_target
        embedding = mid_actor(state)
        pred_mid = final_target(embedding)
        mid_loss = mse_loss(pred_mid, target)
        mid_opt.zero_grad(set_to_none = True)
        mid_loss.backward()
        mid_opt.step()

        # 2) Train final_actor using fixed mid_target
        with torch.no_grad():
            frozen_embed = mid_target(state)
        pred_final = final_actor(frozen_embed)
        final_loss = mse_loss(pred_final, target)
        final_opt.zero_grad(set_to_none = True)
        final_loss.backward()
        final_opt.step()

        # 3) Synchronize targets
        polyak_update(mid_target, mid_actor, 0.05)
        polyak_update(final_target, final_actor, 0.05)

        if i % 1000 == 0:
            print(f"step: {i}, mid_loss: {mid_loss.item()}, final_loss: {final_loss.item()}")
    with torch.no_grad():
        state = torch.rand(1024, 1, device = device)
        target = (state / torch.pi).detach()
        embedding = mid_actor(state)
        pred = final_actor(embedding)
        final_mse = mse_loss(pred, target)
    assert final_mse < 1e-3


def test_a2c_single_actor() -> None:
    actor = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 2,
    ).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr = 3e-4)
    for i in trange(20000, desc = "A2C single actor"):
        state = torch.rand(64, 1, device = device)
        target = state / torch.pi
        params = actor(state)
        sample, log_prob = sample_action_and_logprob(params)
        reward = -((sample - target) ** 2).squeeze(-1)
        centered = reward - reward.mean()
        std = reward.std(unbiased = False) + 1e-8
        advantage = (centered / std).detach()
        actor_loss = -(advantage * log_prob.squeeze(-1)).mean()
        actor_opt.zero_grad(set_to_none = True)
        actor_loss.backward()
        actor_opt.step()
        if i % 1000 == 0:
            with torch.no_grad():
                mean = actor(state)[..., 0:1]
                mse = mse_loss(mean, target)
            print(f"step: {i}, mse: {mse.item()}")
    with torch.no_grad():
        state = torch.rand(1024, 1, device = device)
        target = state / torch.pi
        mean = actor(state)[..., 0:1]
        final_mse = mse_loss(mean, target)
    assert final_mse < 1e-3
    

def test_ddpg_norm_prediction() -> None:
    state_dim = 4
    net = FNN(
        input_size = state_dim,
        hidden_size = 16,
        num_hidden_layers = 4,
        output_size = 1,
    ).to(device)
    opt = optim.Adam(net.parameters(), lr = 1e-3)
    for i in trange(20000, desc = "norm prediction"):
        state = torch.rand(128, state_dim, device = device)
        target = torch.norm(state, dim = -1)
        pred = net(state).squeeze(-1)
        loss = mse_loss(pred, target)
        opt.zero_grad(set_to_none = True)
        loss.backward()
        opt.step()
        if i % 250 == 0:
            with torch.no_grad():
                state = torch.rand(1024, state_dim, device = device)
                target = torch.norm(state, dim = -1)
                pred = net(state).squeeze(-1)
                mse = mse_loss(pred, target)
                print(f"step: {i}, mse: {mse.item()}")
    with torch.no_grad():
        state = torch.rand(1024, state_dim, device = device)
        target = torch.norm(state, dim = -1)
        pred = net(state).squeeze(-1)
        final_mse = mse_loss(pred, target)
    assert final_mse < 1e-3


def test_shared_representation() -> None:
    embed_dim = 16
    actor_critic = FNN(
        input_size = 1,
        hidden_size = 64,
        num_hidden_layers = 4,
        output_size = 1 + embed_dim,
    ).to(device)
    critic = FNN(
        input_size = 1 + embed_dim,
        hidden_size = 64,
        num_hidden_layers = 4,
        output_size = 1,
    ).to(device)
    all_params = list[Parameter](actor_critic.parameters()) + list[Parameter](critic.parameters())
    joint_opt = optim.Adam(all_params, lr = 1e-3)
    for i in trange(200000, desc = "shared representation"):
        joint_opt.zero_grad(set_to_none = True)
        state = torch.rand(64, 1, device = device)
        target = state * torch.pi
        params = actor_critic(state)
        action = params[..., 0:1]
        reward = -((action - target) ** 2).squeeze(-1).detach()
        action, embed = params[..., 0:1], params[..., 1:]
        params_actor = torch.cat([action, embed.detach()], dim = -1)
        q_pred_actor = critic(params_actor).squeeze(-1)
        actor_loss_raw = -q_pred_actor.mean()
        actor_loss_raw.backward()
        grad_actor = [p.grad.clone() if p.grad is not None else None for p in all_params]
        
        
        joint_opt.zero_grad(set_to_none = True)
        state = torch.rand(64, 1, device = device)
        target = state * torch.pi
        params = actor_critic(state)
        action = params[..., 0:1]
        reward = -((action - target) ** 2).squeeze(-1).detach()
        action, embed = params[..., 0:1], params[..., 1:]
        params_critic = torch.cat([action, embed], dim = -1)
        q_pred_critic = critic(params_critic).squeeze(-1)
        critic_loss_raw = mse_loss(q_pred_critic, reward)
        critic_loss_raw.backward()
        grad_critic = [p.grad.clone() if p.grad is not None else None for p in all_params]
        g_a = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_actor)
        ])
        g_c = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_critic)
        ])
        eps = 1e-8
        dot = (g_a * g_c).sum()
        if dot < 0 and False:
            norm_a_sq = g_a.pow(2).sum()
            norm_c_sq = g_c.pow(2).sum()
            g_a_proj = g_a - (dot / (norm_c_sq)) * g_c
            g_c_proj = g_c - (dot / (norm_a_sq)) * g_a
            g_total = g_a_proj + g_c_proj
        else:
            g_total = g_a + g_c
        offset = 0
        for p in all_params:
            n = p.numel()
            p.grad = g_total[offset : offset + n].view_as(p).clone()
            offset += n
        joint_opt.step()
        if i % 500 == 0:
            with torch.no_grad():
                params = actor_critic(state)
                action = params[..., 0:1]
                target = state * torch.pi
                reward = -((action - target) ** 2).squeeze(-1)
                q_pred = critic(params).squeeze(-1)
                mse = -reward.mean()
                critic_mse = mse_loss(q_pred, reward)
            print(f"step: {i}, mse: {mse.item()}, critic_mse: {critic_mse.item()}")
    with torch.no_grad():
        state = torch.rand(1024, 1, device = device)
        target = state * torch.pi
        action = actor_critic(state)[..., 0:1]
        final_mse = mse_loss(action, target)
    assert final_mse < 1e-3


def test_loss_harmonization() -> None:
    net = FNN(
        input_size = 1,
        hidden_size = 32,
        num_hidden_layers = 2,
        output_size = 2,
    ).to(device)
    opt = optim.Adam(net.parameters(), lr = 1e-3)
    all_params = list(net.parameters())
    for i in trange(200000, desc = "loss harmonization"):
        state = torch.rand(64, 1, device = device)
        target_a = state / torch.pi
        target_b = state * 1000.0
        out = net(state)
        pred_a = out[..., 0:1]
        pred_b = out[..., 1:2]
        loss_a = mse_loss(pred_a, target_a)
        loss_b = mse_loss(pred_b, target_b)
        opt.zero_grad(set_to_none = True)
        loss_a.backward(retain_graph = True)
        grad_a = [p.grad.clone() if p.grad is not None else None for p in all_params]
        opt.zero_grad(set_to_none = True)
        loss_b.backward()
        grad_b = [p.grad.clone() if p.grad is not None else None for p in all_params]
        opt.zero_grad(set_to_none = True)
        g_a = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_a)
        ])
        g_b = torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(all_params, grad_b)
        ])
        eps = 1e-8
        dot = (g_a * g_b).sum()
        if dot < 0:
            if i % 2 == 0:
                g_total = g_a - dot * g_b / (g_b.pow(2).sum() + eps)
            else:
                g_total = g_b - dot * g_a / (g_a.pow(2).sum() + eps)
        else:
            g_total = g_a + g_b
        offset = 0
        for p in all_params:
            n = p.numel()
            p.grad = g_total[offset:offset + n].view_as(p).clone()
            offset += n
        opt.step()
        if i % 1000 == 0:
            print(f"step: {i}, mse_a: {loss_a.item()}, mse_b: {loss_b.item()}")
    with torch.no_grad():
        state = torch.rand(1024, 1, device = device)
        target_a = state / torch.pi
        target_b = state * 1000.0
        out = net(state)
        final_mse_a = mse_loss(out[..., 0:1], target_a)
        final_mse_b = mse_loss(out[..., 1:2], target_b)
    assert final_mse_a < 1e-3
    assert final_mse_b < 1000


if __name__ == "__main__":
    test_ddpg_norm_prediction()