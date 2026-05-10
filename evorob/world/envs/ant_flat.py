from pathlib import Path

import numpy as np

from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation as R



class AntFlatEnvironment(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self, render_mode=None, robot_path: str = "ant_flat_terrain.xml", **kwargs
    ):
        # Load MuJoCo environment in Gymnasium
        # Get path to XML file relative to this module
        xml_file_path = str(Path(__file__).parent / "assets" / robot_path)

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=None,  # needs to be defined after
            render_mode=render_mode,
            default_camera_config={
                "distance": 4.0,
            },
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._reset_noise_scale: float = 0.1
        self._main_body = "Base"

        # Define observation space.
        # Action space is automatically defined by MuJoCo.
        # Exclude x,y position (first 2 elements of qpos) to match _get_obs()
        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def reset_model(self):
        noise_low = -0.1
        noise_high = 0.1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.initial_y = self.data.qpos[1]

        return observation
    
    def get_yaw(self):

        quat = self.data.qpos[3:7]
        r = R.from_quat([
            quat[1],
            quat[2],
            quat[3],
            quat[0]
        ])
        yaw = r.as_euler('xyz')[2]

        return yaw

    def step(self, action):
        torso_body_id = 1
        xy_position_before = self.data.body(torso_body_id).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(torso_body_id).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, y_velocity, action)
        terminated = self._get_termination()
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        # Exclude x,y position to make task translation-invariant (like standard Ant-v4)
        position = self.data.qpos[2:].flatten()  # z, quaternion, joint angles
        velocity = self.data.qvel.flatten()  # all velocities

        return np.concatenate((position, velocity))

    def _get_rew(self, x_velocity: float, y_velocity: float, action):

        # Reward weights
        forward_reward_weight = 3.0
        healthy_reward_weight = 1.0
        ctrl_cost_weight = 0.3
        lateral_pos_weight = 0.4
        lateral_vel_weight = 0.1
        heading_weight = 1.0

        # World-frame position
        y_position = self.data.qpos[1]
        # Relative lateral deviation from spawn
        y_error = y_position - self.initial_y
        # Robot orientation
        yaw = self.get_yaw()

        # Positive rewards
        forward_reward = forward_reward_weight * x_velocity
        healthy_reward = healthy_reward_weight

        # Penalties
        # Energy / torque regularization
        ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
        lateral_position_penalty = lateral_pos_weight * (y_error ** 2)
        lateral_velocity_penalty = lateral_vel_weight * (y_velocity ** 2)
        heading_penalty = heading_weight * (yaw ** 2)

        # Stability penalties
        if self.torso_near_tipping():
            healthy_reward -= 1.0
        elif self.torso_upside_down():
            healthy_reward = -4.0

        # Final reward
        reward = (
            forward_reward
            + healthy_reward
            - ctrl_cost
            - lateral_position_penalty
            - lateral_velocity_penalty
            - heading_penalty
        )

        reward_info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "penalty_lateral_pos": -lateral_position_penalty,
            "penalty_lateral_vel": -lateral_velocity_penalty,
            "penalty_heading": -heading_penalty,
        }

        return reward, reward_info
    
    def torso_upside_down(self,):
        R = self.data.body(self._main_body).xmat.reshape(3, 3)
        torso_z_world = R[:, 2]
        # if dot(torso_z, world_z) < 0 → pointing downward → upside down
        return torso_z_world[2] < 0.0

    def torso_near_tipping(self, threshold: float = 0.5) -> bool:
        # torso_z_world[2] = cos(tilt angle); threshold=0.5 → tilted >60° from upright but not yet flipped
        R = self.data.body(self._main_body).xmat.reshape(3, 3)
        z = R[2, 2]
        return 0.0 <= z < threshold

    def _get_termination(self):
        state = self.state_vector()
        min_z_torso, max_z_torso = (0.26, 2.0)
        is_healthy = np.isfinite(state).all() and min_z_torso <= state[2] and not self.torso_upside_down()

        return not is_healthy

