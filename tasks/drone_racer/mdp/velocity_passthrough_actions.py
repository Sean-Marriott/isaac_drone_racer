from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import VelocityPassthroughController
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
class VelocityPassthroughAction(ActionTerm):
    cfg: VelocityPassthroughActionCfg
    
    def __init__(self, cfg: VelocityPassthroughActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._max_thrust = (
            self._robot.data.default_mass.sum(dim=1, keepdim=True)
            * -self._env.sim.cfg.gravity[-1]
            * self.cfg.thrust_weight_ratio
        ).to(self.device)
        
        self._velocity_passthrough_controller = VelocityPassthroughController(
            self._robot.data.default_mass.sum(dim=1, keepdim=True),
            self.cfg.kv,
            self.cfg.kyaw,
            self._max_thrust,
            self.cfg.max_torque_Nm,
            self.num_envs,
            self.device,
        )
        
        self._scale = torch.tensor(
            [cfg.max_vxy, cfg.max_vxy, cfg.max_vz, cfg.max_yaw_rate],
            device=self.device,
        )
        
        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._vel_cmd     = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
    
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
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
        """Process raw policy actions into velocity and yaw commands"""
        self._raw_actions[:] = actions.clone()
        self._vel_cmd[:] = self._raw_actions.clamp(-1.0, 1.0) * self._scale
        self._processed_actions[:] = self._vel_cmd

    def apply_actions(self):
        # Computed here (physics step) not in process_actions (policy step) so the PD feedback
        # uses fresh state on every decimation sub-step.
        # See: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.apply_actions
        wrench = self._velocity_passthrough_controller.compute(
            self._vel_cmd,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
        )
        self._thrust[:, 0, :] = wrench[:, :3]
        self._moment[:, 0, :] = wrench[:, 3:]
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            self._thrust, self._moment, body_ids=self._body_id
        )
        # TEST: zero roll/pitch angular velocity each step
        root_vel = self._robot.data.root_vel_w.clone()
        root_vel[:, 3] = 0.0
        root_vel[:, 4] = 0.0
        self._robot.write_root_velocity_to_sim(root_vel)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        """Reset action buffers and robot joint state

        Args:
            env_ids: Environment indices to reset. If ``None`` or all envs,
                resets every environment.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
 
        self._raw_actions[env_ids]       = 0.0
        self._processed_actions[env_ids] = 0.0
        self._vel_cmd[env_ids]           = 0.0
        self._elapsed_time[env_ids]      = 0.0
  
        self._robot.reset(env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

    
@configclass
class VelocityPassthroughActionCfg(ActionTermCfg):
    """Configuration for VelocityPassthroughAction.
 
    Drop this into DroneRacerEnvCfg.actions instead of ControlActionCfg.
    """
 
    class_type: type = VelocityPassthroughAction
 
    asset_name: str = "robot"
 
    # ── Velocity limits ──────────────────────────────────────────────────────
    max_vxy: float = 3.0
    """Lateral velocity limit (m/s). Match PX4 MPC_XY_VEL_MAX or DJI max."""
 
    max_vz: float = 2.0
    """Vertical velocity limit (m/s). Match PX4 MPC_Z_VEL_MAX_UP / _DN."""
 
    max_yaw_rate: float = 3.14159
    """Yaw rate limit (rad/s). Match PX4 MC_YAWRATE_MAX (deg→rad)."""
 
    # ── Controller gains ─────────────────────────────────────────────────────
    kv: float = 5.0
    """Velocity P gain. Higher = snappier; lower = more sluggish / real.
    Good starting point: kv = 1.0 / bandwidth_tau."""
 
    kyaw: float = 2.0
    """Yaw rate P gain."""
 
    # # ── Bandwidth filter ─────────────────────────────────────────────────────
    # TODO
    # bandwidth_tau: float = 0.15
    # """First-order lag time constant (seconds) simulating autopilot bandwidth.

 
    # ── Physical parameters ──────────────────────────────────────────────────
 
    max_torque_Nm: float = 0.5
    """Yaw torque clamp (N·m)."""
    
    thrust_weight_ratio: float = 2.5
    """Thrust weight ratio of the drone."""