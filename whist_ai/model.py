"""Shared masked policy network for the phase-separated controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .engine import BID_ORDER, Card
from .policies import Policy

OBSERVATION_SIZE = 138
PHASE_INDEX = {"bidding": 0, "choose_trump": 1, "choose_partner": 2, "play": 3}


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.heads = nn.ModuleDict({
            "bidding": nn.Linear(hidden_size, 11),
            "choose_trump": nn.Linear(hidden_size, 4),
            "choose_partner": nn.Linear(hidden_size, 4),
            "play": nn.Linear(hidden_size, 52),
        })
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor, phase: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observation)
        return self.heads[phase](features), self.value(features).squeeze(-1)


def encode_observation(observation: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    index = 0
    for suit, rank in observation["hand"]:
        vector[index + suit * 13 + rank] = 1.0
    index += 52
    for _, (suit, rank) in observation["trick_cards"]:
        vector[index + suit * 13 + rank] = 1.0
    index += 52
    vector[index + PHASE_INDEX[observation["phase"]]] = 1.0
    index += 4
    current_bid = observation["current_bid"]
    if current_bid is None:
        bid_index = 0
    elif current_bid == "paskrig":
        bid_index = 11
    else:
        bid_index = BID_ORDER.index(current_bid) + 1
    vector[index + bid_index] = 1.0
    index += 12
    if observation["trump_suit"] is not None:
        vector[index + observation["trump_suit"]] = 1.0
    index += 5
    vector[index + observation["player"]] = 1.0
    index += 4
    if observation["declarer"] is not None:
        vector[index + observation["declarer"]] = 1.0
    index += 5
    vector[index:index + 4] = np.asarray(observation["tricks_won"], dtype=np.float32) / 13.0
    return vector


class LearnedPolicy(Policy):
    def __init__(self, network: PolicyNetwork, device: str = "cpu", deterministic: bool = False) -> None:
        self.network = network
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.transitions: list[tuple[torch.Tensor, str, tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _choose(self, observation: dict[str, Any], phase: str, legal_indices: Sequence[int]) -> int:
        state = torch.as_tensor(encode_observation(observation), dtype=torch.float32, device=self.device)
        logits, value = self.network(state, phase)
        mask = torch.full_like(logits, float("-inf"))
        mask[list(legal_indices)] = 0.0
        distribution = Categorical(logits=logits + mask)
        action = torch.argmax(distribution.logits) if self.deterministic else distribution.sample()
        log_probability = distribution.log_prob(action)
        self.transitions.append((state, phase, tuple(legal_indices), action.detach(), log_probability, value))
        return int(action.item())

    def choose_bid(self, observation, legal_bids):
        index = {"pass": 0}
        index.update({bid: position + 1 for position, bid in enumerate(BID_ORDER)})
        action = self._choose(observation, "bidding", [index[bid] for bid in legal_bids])
        return "pass" if action == 0 else BID_ORDER[action - 1]

    def choose_trump(self, observation, legal_suits):
        return self._choose(observation, "choose_trump", legal_suits)

    def choose_partner(self, observation, legal_suits):
        return self._choose(observation, "choose_partner", legal_suits)

    def choose_card(self, observation, legal_cards):
        indices = [suit * 13 + rank for suit, rank in legal_cards]
        action = self._choose(observation, "play", indices)
        return divmod(action, 13)

    def clear_transitions(self) -> None:
        self.transitions.clear()


def save_checkpoint(network: PolicyNetwork, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": network.state_dict(), "observation_size": OBSERVATION_SIZE}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> PolicyNetwork:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("observation_size") != OBSERVATION_SIZE:
        raise ValueError("checkpoint observation version is incompatible")
    network = PolicyNetwork().to(device)
    network.load_state_dict(checkpoint["model"])
    return network
