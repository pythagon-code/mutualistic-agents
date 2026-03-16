from dataclasses import dataclass

@dataclass
class Config:
    state_dim: int
    action_dim: int
    num_encoders: int
    tree_depth: int
    num_hidden_layers: int
    embed_dim: int
    discount_rate: float
    learning_rate: float
    polyak_factor: float
    batch_size: int
    num_episodes: int
    replay_buffer_size: int