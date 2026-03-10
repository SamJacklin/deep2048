import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from deep_2048.config import get_settings
from deep_2048.game_logic.constants import Action
from deep_2048.model.models import ActionSelectionResult, OptimisationResult, ReplayBatch
from deep_2048.model.network import DQN

settings = get_settings()


class DQNAgent:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = settings.BOARD_SIZE * settings.BOARD_SIZE
        output_dim = len(Action)

        self.online_network = DQN(
            input_dim=input_dim,
            hidden_dim=settings.HIDDEN_DIM,
            output_dim=output_dim,
        ).to(self.device)

        self.target_network = DQN(
            input_dim=input_dim,
            hidden_dim=settings.HIDDEN_DIM,
            output_dim=output_dim,
        ).to(self.device)

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=settings.LEARNING_RATE,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=settings.TRAINING_STEPS,
            eta_min=1e-6,
        )

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: list[Action],
        epsilon: float,
    ) -> ActionSelectionResult:
        if not valid_actions:
            raise ValueError("Cannot select an action when no valid actions are available.")

        if random.random() < epsilon:
            action = random.choice(valid_actions)
            return ActionSelectionResult(
                action=action,
                epsilon=epsilon,
                is_random=True,
                q_values=[],
                valid_actions=valid_actions,
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values_tensor = self.online_network(state_tensor).squeeze(0)

        masked_q_values = q_values_tensor.clone()
        invalid_actions = {action for action in Action if action not in valid_actions}
        for invalid_action in invalid_actions:
            masked_q_values[int(invalid_action)] = float("-inf")

        chosen_action = Action(int(torch.argmax(masked_q_values).item()))

        return ActionSelectionResult(
            action=chosen_action,
            epsilon=epsilon,
            is_random=False,
            q_values=q_values_tensor.detach().cpu().tolist(),
            valid_actions=valid_actions,
        )

    def optimise(self, batch: ReplayBatch) -> OptimisationResult:
        current_q_values = self.online_network(batch.states)
        chosen_q_values = current_q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next_actions = self.online_network(batch.next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(batch.next_states).gather(1, best_next_actions).squeeze(1)
            target_q_values = batch.rewards + ((1.0 - batch.dones) * settings.GAMMA * next_q_values)

        loss = F.huber_loss(chosen_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), settings.GRAD_CLIP_NORM)
        self.optimizer.step()
        self.scheduler.step()

        return OptimisationResult(
            loss=float(loss.item()),
            mean_q_value=float(chosen_q_values.mean().item()),
            mean_target_q_value=float(target_q_values.mean().item()),
            lr=float(self.scheduler.get_last_lr()[0]),
        )

    def update_target_network(self) -> None:
        for target_param, online_param in zip(
            self.target_network.parameters(), self.online_network.parameters()
        ):
            target_param.data.copy_(
                settings.TAU * online_param.data + (1.0 - settings.TAU) * target_param.data
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        online_sd = self.online_network.state_dict()
        torch.save(
            {
                "online_network_state_dict": online_sd,
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "architecture": {
                    "input_dim": online_sd["feature.0.weight"].shape[1],
                    "hidden_dim": online_sd["feature.0.weight"].shape[0],
                    "output_dim": online_sd["advantage_stream.2.weight"].shape[0],
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)

        arch = checkpoint.get("architecture")
        if arch is None:
            online_sd = checkpoint["online_network_state_dict"]
            arch = {
                "input_dim": online_sd["feature.0.weight"].shape[1],
                "hidden_dim": online_sd["feature.0.weight"].shape[0],
                "output_dim": online_sd["advantage_stream.2.weight"].shape[0],
            }

        self.online_network = DQN(**arch).to(self.device)
        self.target_network = DQN(**arch).to(self.device)

        self.online_network.load_state_dict(checkpoint["online_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def set_eval_mode(self) -> None:
        self.online_network.eval()
        self.target_network.eval()

    def set_train_mode(self) -> None:
        self.online_network.train()
        self.target_network.eval()
