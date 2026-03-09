from fnn import FNN
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from typing import Callable, TypeVar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTOR_POLYAK_FACTOR = 0.0001
CRITIC_POLYAK_FACTOR = 0.01

T = TypeVar("T", float, Tensor)


def get_ema(ema: T, data: T, factor: float) -> T:
    return data * factor + ema * (1 - factor)


def polyak_update(target: nn.Module, online: nn.Module, polyak_factor: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(get_ema(target_param.data, online_param.data, polyak_factor))


def cos_sim_of_cube(vec_a: Tensor, vec_b: Tensor) -> Tensor:
    num = (vec_a ** 3 * vec_b ** 3).sum(dim = -1)
    denom = vec_a.pow(3).norm(dim = -1) * vec_b.pow(3).norm(dim = -1) + 1e-8
    return num / denom


class Actor(nn.Module):
    def __init__(self, vec_dim: int, embed_dim: int) -> None:
        super().__init__()
        self._vec_dim = vec_dim
        self._embed_dim = embed_dim
        self._enc_a = FNN(
            input_size = vec_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._enc_b = FNN(
            input_size = vec_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._head = FNN(
            input_size = embed_dim * 2,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = 1,
        ).to(device)
        self._enc_a_target = FNN(
            input_size = vec_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._enc_b_target = FNN(
            input_size = vec_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._head_target = FNN(
            input_size = embed_dim * 2,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = 1,
        ).to(device)
        self._enc_a_target.load_state_dict(self._enc_a.state_dict())
        self._enc_b_target.load_state_dict(self._enc_b.state_dict())
        self._head_target.load_state_dict(self._head.state_dict())
        for param in self._enc_a_target.parameters():
            param.requires_grad_(False)
        for param in self._enc_b_target.parameters():
            param.requires_grad_(False)
        for param in self._head_target.parameters():
            param.requires_grad_(False)
        self._enc_a_opt = optim.Adam(self._enc_a.parameters(), lr = 1e-3)
        self._enc_b_opt = optim.Adam(self._enc_b.parameters(), lr = 1e-3)
        self._head_opt = optim.Adam(self._head.parameters(), lr = 1e-3)


    def forward(self, vec_a: Tensor, vec_b: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            embed_a = self._enc_a_target(vec_a)
            embed_b = self._enc_b_target(vec_b)
            head_in = torch.cat([embed_a, embed_b], dim = -1)
            pred = self._head_target(head_in).squeeze(-1)
        return vec_a, vec_b, embed_a, embed_b, pred, target


    def train(self, critic: "Critic", vec_a: Tensor, vec_b: Tensor, target: Tensor) -> None:
        with torch.no_grad():
            embed_a_head = self._enc_a_target(vec_a)
            embed_b_head = self._enc_b_target(vec_b)
        head_in = torch.cat([embed_a_head, embed_b_head], dim = -1)
        pred_head = self._head(head_in).squeeze(-1)
        actor_loss_head = critic.forward(
            vec_a = vec_a,
            vec_b = vec_b,
            embed_a = embed_a_head,
            embed_b = embed_b_head,
            pred = pred_head,
        )
        head_loss = actor_loss_head.mean()
        self._head_opt.zero_grad(set_to_none = True)
        head_loss.backward()
        clip_grad_norm_(self._head.parameters(), max_norm = 1.0)
        self._head_opt.step()
        polyak_update(self._head_target, self._head, ACTOR_POLYAK_FACTOR)

        with torch.no_grad():
            embed_b_for_a = self._enc_b_target(vec_b)
        embed_a = self._enc_a(vec_a)
        head_in_a = torch.cat([embed_a, embed_b_for_a], dim = -1)
        pred_a = self._head_target(head_in_a).squeeze(-1)
        actor_loss_a = critic.forward(
            vec_a = vec_a,
            vec_b = vec_b,
            embed_a = embed_a,
            embed_b = embed_b_for_a,
            pred = pred_a,
        )
        enc_a_loss = actor_loss_a.mean()
        self._enc_a_opt.zero_grad(set_to_none = True)
        enc_a_loss.backward()
        clip_grad_norm_(self._enc_a.parameters(), max_norm = 1.0)
        self._enc_a_opt.step()
        polyak_update(self._enc_a_target, self._enc_a, ACTOR_POLYAK_FACTOR)

        with torch.no_grad():
            embed_a_for_b = self._enc_a_target(vec_a)
        embed_b = self._enc_b(vec_b)
        head_in_b = torch.cat([embed_a_for_b, embed_b], dim = -1)
        pred_b = self._head_target(head_in_b).squeeze(-1)
        actor_loss_b = critic.forward(
            vec_a = vec_a,
            vec_b = vec_b,
            embed_a = embed_a_for_b,
            embed_b = embed_b,
            pred = pred_b,
        )
        enc_b_loss = actor_loss_b.mean()
        self._enc_b_opt.zero_grad(set_to_none = True)
        enc_b_loss.backward()
        clip_grad_norm_(self._enc_b.parameters(), max_norm = 1.0)
        self._enc_b_opt.step()
        polyak_update(self._enc_b_target, self._enc_b, ACTOR_POLYAK_FACTOR)


    def eval_mse(self, critic: "Critic", target_fn: Callable, batch_size: int = 4096) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            vec_a = torch.rand(batch_size, self._vec_dim, device = device)
            vec_b = torch.rand(batch_size, self._vec_dim, device = device)
            target = target_fn(vec_a, vec_b)
            vec_a, vec_b, embed_a, embed_b, pred, target = self.forward(vec_a, vec_b, target)
            actor_mse = mse_loss(pred, target)

            critic_pred = critic.forward(
                vec_a = vec_a,
                vec_b = vec_b,
                embed_a = embed_a,
                embed_b = embed_b,
                pred = pred,
            )
            actor_loss = critic_pred.mean()
            target_loss = (pred - target) ** 2
            critic_mse = mse_loss(critic_pred, target_loss)
            return actor_mse, actor_loss, critic_mse


class Critic(nn.Module):
    def __init__(self, vec_dim: int, actor_embed_dim: int, embed_dim: int) -> None:
        super().__init__()
        self._vec_dim = vec_dim
        self._actor_embed_dim = actor_embed_dim
        self._embed_dim = embed_dim
        self._enc_a = FNN(
            input_size = vec_dim + actor_embed_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._enc_b = FNN(
            input_size = vec_dim + actor_embed_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._head = FNN(
            input_size = embed_dim * 2 + 1,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = 1,
        ).to(device)
        self._enc_a_target = FNN(
            input_size = vec_dim + actor_embed_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._enc_b_target = FNN(
            input_size = vec_dim + actor_embed_dim,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = embed_dim,
        ).to(device)
        self._head_target = FNN(
            input_size = embed_dim * 2 + 1,
            hidden_size = 16,
            num_hidden_layers = 4,
            output_size = 1,
        ).to(device)
        self._enc_a_target.load_state_dict(self._enc_a.state_dict())
        self._enc_b_target.load_state_dict(self._enc_b.state_dict())
        self._head_target.load_state_dict(self._head.state_dict())
        for param in self._enc_a_target.parameters():
            param.requires_grad_(False)
        for param in self._enc_b_target.parameters():
            param.requires_grad_(False)
        for param in self._head_target.parameters():
            param.requires_grad_(False)
        self._enc_a_opt = optim.Adam(self._enc_a.parameters(), lr = 1e-3)
        self._enc_b_opt = optim.Adam(self._enc_b.parameters(), lr = 1e-3)
        self._head_opt = optim.Adam(self._head.parameters(), lr = 1e-3)


    def forward(
        self,
        vec_a: Tensor,
        vec_b: Tensor,
        embed_a: Tensor,
        embed_b: Tensor,
        pred: Tensor,
    ) -> Tensor:
        state_action_a = torch.cat([vec_a, embed_a], dim = -1)
        state_action_b = torch.cat([vec_b, embed_b], dim = -1)
        critic_embed_a = self._enc_a_target(state_action_a)
        critic_embed_b = self._enc_b_target(state_action_b)
        head_in = torch.cat([critic_embed_a, critic_embed_b, pred.unsqueeze(-1)], dim = -1)
        return self._head_target(head_in).squeeze(-1)


    def train(
        self,
        vec_a: Tensor,
        vec_b: Tensor,
        embed_a: Tensor,
        embed_b: Tensor,
        pred: Tensor,
        target: Tensor,
    ) -> None:
        with torch.no_grad():
            target_loss = (pred - target) ** 2
            state_action_a = torch.cat([vec_a, embed_a], dim = -1)
            state_action_b = torch.cat([vec_b, embed_b], dim = -1)
            critic_embed_a = self._enc_a_target(state_action_a)
            critic_embed_b = self._enc_b_target(state_action_b)
        head_in = torch.cat([critic_embed_a, critic_embed_b, pred.unsqueeze(-1)], dim = -1)
        pred_loss = self._head(head_in).squeeze(-1)
        head_loss = mse_loss(pred_loss, target_loss)
        self._head_opt.zero_grad(set_to_none = True)
        head_loss.backward()
        clip_grad_norm_(self._head.parameters(), max_norm = 1.0)
        self._head_opt.step()
        polyak_update(self._head_target, self._head, CRITIC_POLYAK_FACTOR)

        with torch.no_grad():
            state_action_b = torch.cat([vec_b, embed_b], dim = -1)
            critic_embed_b = self._enc_b_target(state_action_b)
        state_action_a = torch.cat([vec_a, embed_a], dim = -1)
        critic_embed_a = self._enc_a(state_action_a)
        head_in_a = torch.cat([critic_embed_a, critic_embed_b, pred.unsqueeze(-1)], dim = -1)
        pred_loss_a = self._head_target(head_in_a).squeeze(-1)
        enc_a_loss = mse_loss(pred_loss_a, target_loss)
        self._enc_a_opt.zero_grad(set_to_none = True)
        enc_a_loss.backward()
        clip_grad_norm_(self._enc_a.parameters(), max_norm = 1.0)
        self._enc_a_opt.step()
        polyak_update(self._enc_a_target, self._enc_a, CRITIC_POLYAK_FACTOR)

        with torch.no_grad():
            state_action_a = torch.cat([vec_a, embed_a], dim = -1)
            critic_embed_a = self._enc_a_target(state_action_a)
        state_action_b = torch.cat([vec_b, embed_b], dim = -1)
        critic_embed_b = self._enc_b(state_action_b)
        head_in_b = torch.cat([critic_embed_a, critic_embed_b, pred.unsqueeze(-1)], dim = -1)
        pred_loss_b = self._head_target(head_in_b).squeeze(-1)
        enc_b_loss = mse_loss(pred_loss_b, target_loss)
        self._enc_b_opt.zero_grad(set_to_none = True)
        enc_b_loss.backward()
        clip_grad_norm_(self._enc_b.parameters(), max_norm = 1.0)
        self._enc_b_opt.step()
        polyak_update(self._enc_b_target, self._enc_b, CRITIC_POLYAK_FACTOR)


    def eval_mse(self, actor: Actor, batch_size: int, target_fn) -> Tensor:
        with torch.no_grad():
            vec_a = torch.rand(batch_size, self._vec_dim, device = device)
            vec_b = torch.rand(batch_size, self._vec_dim, device = device)
            target = target_fn(vec_a, vec_b)
            vec_a, vec_b, embed_a, embed_b, pred, target = actor.forward(vec_a, vec_b, target)
            target_loss = (pred - target) ** 2
            critic_pred = self.forward(
                vec_a = vec_a,
                vec_b = vec_b,
                embed_a = embed_a,
                embed_b = embed_b,
                pred = pred,
            )
            return mse_loss(critic_pred, target_loss)


def test_mutual_cosine_similarity() -> None:
    actor = Actor(
        vec_dim = 4,
        embed_dim = 8,
    )
    critic = Critic(
        vec_dim = 4,
        actor_embed_dim = 8,
        embed_dim = 8,
    )
    for i in trange(20000, desc = "mutual cosine"):
        vec_a = torch.rand(128, 4, device = device)
        vec_b = torch.rand(128, 4, device = device)
        target = cos_sim_of_cube(vec_a, vec_b)
        critic_params = actor.forward(vec_a, vec_b, target)
        actor.train(critic, vec_a, vec_b, target)
        critic.train(*critic_params)
        if i % 100 == 0:
            actor_mse, actor_loss, critic_mse = actor.eval_mse(
                critic = critic,
                target_fn = cos_sim_of_cube,
                batch_size = 1024,
            )
            print(
                f"step: {i}, actor_mse: {actor_mse.item():.10f}, "
                f"critic_loss: {critic_loss.item():.10f}, critic_mse: {critic_mse.item():.10f}"
            )

    final_actor_mse, final_critic_loss, final_critic_mse = actor.eval_mse(
        critic = critic,
        target_fn = cos_sim_of_cube,
        batch_size = 4096,
    )
    assert final_actor_mse < 5e-3


if __name__ == "__main__":
    test_mutual_cosine_similarity()

