from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityPassthroughAction(ActionTerm):
    """Velocity passthrough action term.

    Assumes the real drone has a perfect inner velocity controller and responds instantaneously. 
    The policy outputs body-frame velocity commands [vx, vy, vz, yaw_rate], which are
    applied directly to the simulator state each step. Roll and pitch rates are
    locked to zero, consistent with a levelling inner loop.
    """

    cfg: VelocityPassthroughActionCfg

    def __init__(self, cfg: VelocityPassthroughActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg
        self._robot: Articulation = env.scene[self.cfg.asset_name]

        self._scale = torch.tensor(
            [cfg.max_vxy, cfg.max_vxy, cfg.max_vz, cfg.max_yaw_rate],
            device=self.device,
        )

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._vel_cmd = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clone()
        self._vel_cmd[:] = self._raw_actions.clamp(-1.0, 1.0) * self._scale
        self._processed_actions[:] = self._vel_cmd

    def apply_actions(self):
        root_vel = self._robot.data.root_vel_w.clone()
        
        root_vel[:, :3] = quat_apply(self._robot.data.root_quat_w, self._vel_cmd[:, :3])
        root_vel[:, 3] = 0.0  # lock roll rate
        root_vel[:, 4] = 0.0  # lock pitch rate
        root_vel[:, 5] = self._vel_cmd[:, 3]  # yaw rate (body ≈ world with locked roll/pitch)
        self._robot.write_root_velocity_to_sim(root_vel)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._vel_cmd[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._robot.reset(env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

@configclass
class VelocityPassthroughActionCfg(ActionTermCfg):
    """Configuration for VelocityPassthroughAction."""

    class_type: type = VelocityPassthroughAction

    asset_name: str = "robot"

    max_vxy: float = 15.0
    """Lateral velocity limit (m/s)."""

    max_vz: float = 5.0
    """Vertical velocity limit (m/s)."""

    max_yaw_rate: float = 3.14159
    """Yaw rate limit (rad/s)."""
