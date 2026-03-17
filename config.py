from dataclasses import dataclass

@dataclass
class Config:
    state_dim: int
    action_dim: int
    action_tanh: bool
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
    add_frequency: float
    update_interval: int
    noise_std: float
    noise_std_min: float
    noise_std_decay: float