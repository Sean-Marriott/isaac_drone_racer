import torch


class VelocityPassthroughController:
    """
    Applies body-frame velocity commands directly as body-frame forces.
     
    The RL policy outputs body-frame velocity commands:
        [vx, vy, vz, yaw_rate]   all in [-1, 1], then scaled to physical units.
    
    Rather than simulating the drone's inner attitude/rate loops (Lee, PID, etc.),
    this controller enforces the velocity command directly on the rigid body via
    world-frame forces and a yaw torque. Isaac's physics engine integrates the
    result, so the drone moves as if its inner loop were perfect.
    """

    def __init__(
        self,
        mass: torch.Tensor,
        k_v: float,
        k_yaw: float,
        max_thrust: torch.Tensor,
        max_torque_Nm: float,
        num_envs: int,
        device="cpu",
    ):
        self._mass = mass.to(device)
        self._k_v = k_v
        self._k_yaw = k_yaw
        self._max_thrust = max_thrust.to(device)
        self._max_torque_Nm = max_torque_Nm
        self._num_envs = num_envs
        self._device = device

    def compute(
        self,
        vel_cmd_body: torch.Tensor,  # [N, 4] vx, vy, vz, yaw_rate (body frame)
        vel_body: torch.Tensor,      # [N, 3] actual body-frame velocity
        omega_body: torch.Tensor,    # [N, 3] body-frame angular velocity
    ) -> torch.Tensor:

        v_cmd_xyz    = vel_cmd_body[:, :3]  # [N, 3]
        yaw_rate_cmd = vel_cmd_body[:, 3]   # [N]

        force_body = self._mass * self._k_v * (v_cmd_xyz - vel_body)  # [N, 3]

        mag = torch.norm(force_body, dim=1, keepdim=True).clamp(min=1e-6)
        force_body = torch.where(
            mag > self._max_thrust,
            force_body * (self._max_thrust / mag),
            force_body,
        )

        tau_z = (self._k_yaw * (yaw_rate_cmd - omega_body[:, 2])).clamp(
            -self._max_torque_Nm, self._max_torque_Nm
        )

        wrench = torch.zeros(self._num_envs, 6, device=self._device)
        wrench[:, :3] = force_body
        wrench[:, 5] = tau_z

        return wrench
