from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP

def preprocess_pixels(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    if x.dtype != jnp.float32:
        x = x.astype(jnp.float32)
    # If values look like uint8 [0..255], normalize to [0..1]
    x = jnp.where(x > 1.5, x / 255.0, x)
    return x

class CNNEncoder(nn.Module):
    features_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (B,100,100,1) float32 in [0,1]
        x = preprocess_pixels(obs)

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)  # 100 -> 24
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)  # 24 -> 11
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)  # 11 -> 9
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.features_dim)(x)
        x = nn.relu(x)
        return x  # (B, features_dim)

class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    features_dim: int = 256

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = CNNEncoder(self.features_dim)(observations)
        critic = MLP((*self.hidden_dims, 1))(critic)
        return jnp.squeeze(critic, -1)

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    features_dim: int = 256
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = CNNEncoder(self.features_dim)(observations) 
        inputs = jnp.concatenate([inputs, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    features_dim: int = 256
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
