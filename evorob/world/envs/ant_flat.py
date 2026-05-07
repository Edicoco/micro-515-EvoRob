from pathlib import Path

import numpy as np

from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv


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
        self, render_mode=None, robot_path: str = "terrain.xml", **kwargs
    ):
        xml_file_path = str(Path(__file__).parent / "assets" / robot_path)

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode,
            default_camera_config={"distance": 4.0},
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._reset_noise_scale: float = 0.1
        self._main_body = 1
        self.stuck = 0  # ← ajout
        self.ground = 0  # ← ajout

        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale, high=self._reset_noise_scale, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        self.stuck = 0  # ← reset
        return self._get_obs()

    def step(self, action):
        torso_body_id = 1
        xyz_position_before = self.data.body(torso_body_id).xpos[:3].copy()  # ← [:3]
        self.do_simulation(action, self.frame_skip)
        xyz_position_after = self.data.body(torso_body_id).xpos[:3].copy()

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = self._get_termination(xyz_velocity, observation)  # ← args passés

        if terminated:
            reward -= 3.0  # ← pénalité mort

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_velocity": z_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos[2:].flatten(), self.data.qvel.flatten()])
    
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

    def _get_rew(self, x_velocity: float, action):
        forward_reward = x_velocity * 3
        healthy_reward = 1.0
        ctrl_cost = -0.25 * np.sum(action ** 2)

        if self.torso_near_tipping():
            risk_penalty = -1.5  # pénalité pour être proche du basculement
            healthy_reward += risk_penalty

        reward = forward_reward + healthy_reward + ctrl_cost 

        return reward, {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": - ctrl_cost,
        }

    def _get_termination(self, xyz_velocity=None, observation=None):

        # 1. Torso à l'envers
        if self.torso_upside_down():
            return True
        
        if observation is not None:
            # 2. Robot trop bas (chute)
            z_position = self.data.qpos[2]
            if z_position < 0.26:  # seuil à ajuster selon la géométrie du robot
                return True

        # 5. Robot bloqué
        if xyz_velocity is not None:
            if np.linalg.norm(xyz_velocity) < 1e-3:
                self.stuck += 1
                if self.stuck > 30:
                    return True
            else:
                self.stuck = 0

        return False
