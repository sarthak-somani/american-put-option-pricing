"""Double DQN building blocks: state expansion, network, replay buffer, loss.

State fed to the network is 3-dimensional: [time_fraction, time_to_expiry,
moneyness]. AmericanPutEnv itself only exposes [time_fraction, moneyness]
(see ../week7-rl-formulation/environment.py); `expand_state` appends the
redundant time_to_expiry = 1 - time_fraction feature so this module's state
matches the spec it was built to. Note this is *not* an expressiveness gain:
a single linear layer already recovers 1 - time_fraction from time_fraction
alone (weight -1, bias 1) at zero capacity cost. It's kept for parity with
the assignment's stated state vector, not because it "helps the network
converge faster" -- that stronger claim doesn't survive checking the algebra
of a linear-then-ReLU first layer.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_state(state):
    """[time_fraction, moneyness] -> [time_fraction, time_to_expiry, moneyness]."""
    time_fraction, moneyness = state[0], state[1]
    return np.array([time_fraction, 1.0 - time_fraction, moneyness], dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity transition store with its own sampling RNG.

    Sampling uses a caller-supplied `np.random.Generator` rather than the
    global `random` module, so replay-buffer randomness stays a separately
    seeded, traceable stream -- the same convention the Week 7 tabular
    Q-learner uses to decouple exploration noise from env price-path noise
    (see ../week7-rl-formulation/q_learning.py).
    """

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def compute_dqn_loss(online, target, batch, discount):
    """Double DQN loss: online net selects the next action, target net evaluates it.

    Plain DQN's target = reward + gamma * max_a Q_target(next_state, a) reuses
    the same (biased) network to both pick and score the best next action,
    which produces a systematic positive overestimation bias -- exactly the
    failure mode that would show up here as a policy value suspiciously above
    the binomial benchmark (large, sparse terminal payoffs up to K make this
    environment a textbook case for it). Double DQN decouples selection from
    evaluation:

        target = reward + (1 - done) * gamma * Q_target(s', argmax_a Q_online(s', a))
    """
    states, actions, rewards, next_states, dones = batch
    device = next(online.parameters()).device

    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_selected = online(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_online_actions = online(next_states_t).argmax(dim=1, keepdim=True)
        next_q_target = target(next_states_t).gather(1, next_online_actions).squeeze(1)
        q_target_value = rewards_t + (1.0 - dones_t) * discount * next_q_target

    loss = F.smooth_l1_loss(q_selected, q_target_value)
    return loss, q_selected.detach()


def save_artifact(path, model, hyperparams, seed, extra=None):
    """Save model weights + exact hyperparameters + seed (mirrors
    ../week6-neural-pricer/model.py's artifact convention)."""
    payload = {"model_state": model.state_dict(), "hyperparams": hyperparams, "seed": seed}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_artifact(path, map_location="cpu"):
    """weights_only=False is required: the artifact bundles a plain-dict
    hyperparams payload alongside the model state dict, which torch's default
    weights_only=True unpickler (torch>=2.6) rejects. Only load artifacts you
    saved yourself."""
    return torch.load(path, map_location=map_location, weights_only=False)
