from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
#from gymnasium.envs.mujoco import MujocoEnv
from mujobot.envs.mujoco_env import MujocoEnv
import mujoco
import numpy as np
import math
import os

class UR5PaddleEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple",],
    }

    def __init__(self, **kwargs):

        MujocoEnv.__init__(
            self,
            model_path=os.path.dirname(os.path.abspath(__file__)) + "/ur5_paddle/scene.xml",
            frame_skip=5,
            observation_space=None,
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

        # Create a data structure to hold the simulation state
        #self.data = mujoco.MjData(self.model)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2]*3 + [-2*math.pi]*6, dtype=np.float32), high=np.array([+2]*3 + [+2*math.pi]*6, dtype=np.float32), shape=(9,))

        # Action space is 6 joint angles and the gripper open/close level (0 to 1)
        action_max = 2*math.pi / 360
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6, dtype=np.float32), high=np.array([+action_max]*6, dtype=np.float32), shape=(6,))
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        self.paddle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "paddle")


    def reset_model(self):
        self.data.qpos[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]
        # Starting position
        self.data.ctrl[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]
        self.data.ctrl[6] = 255  # Gripper closed

        # Set the new position of the ball 
        random_diff = np.random.uniform(low=[-0.1, -0.09, 0], high=[+0.09, +0.15, 0])
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        qpos_idx = self.model.jnt_qposadr[ball_joint_id]  # Get the index in qpos
        self.data.qpos[qpos_idx:qpos_idx+3] += random_diff

        #self.wait_until_stable()

        return self.get_observation()

    """
    def wait_until_stable(self, sim_steps=500, tolerance=1e-3):
        joint_pos = np.array(self.data.qpos[0:6], dtype=np.float32)

        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()
            new_joint_pos = np.array(self.data.qpos[0:6], dtype=np.float32)

            if np.linalg.norm(new_joint_pos - joint_pos, ord=np.inf) < tolerance:  # Faster error check, similar to abs np.all
                return True
        
            joint_pos = new_joint_pos
        print("Warning: The robot configuration did not stabilize")
        return False
        """

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        action[0:6] = self.data.ctrl[0:6] + action[0:6]

        action[0:6] = np.clip(action[0:6], -2*math.pi, +2*math.pi)
        action = np.array(action, dtype=np.float32)
        action = np.append(action, 255) # Gripper closed at all times

        self.do_simulation(action, self.frame_skip)

        #self.update_target_pos()

        ball_pos = self.get_ball_position()
        paddle_pos = self.get_paddle_position()
        reward = +1 # Keep alive reward
        reward += max(0, paddle_pos[2] - 0.8) # Paddle higher than 0.8, add reward
        terminated = False
        if ball_pos[2] < 0.2: # Ball on the ground or too low, fallen
            print("Ball fell!")
            reward = -500
            terminated= True
        truncated = False
        obs = self.get_observation()
        info = {}
        if self.render_mode == "human":
            self.render()
            #time.sleep(0.01)
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        ball_pos = self.get_ball_position()
        paddle_pos = self.get_paddle_position() # Geometrical center of the paddle
        pos_diff = paddle_pos - ball_pos

        joint_pos = self.data.qpos[0:6]
        return np.concatenate((pos_diff, joint_pos), dtype=np.float32)

    def get_ee_pose(self):
        pinch_name = "pinch"
        pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pinch_name)
        ee_pos_absolute = self.data.site_xpos[pinch_id]
        rotation_matrix = self.data.xmat[pinch_id].reshape(3, 3)  # Extract 3×3 rotation matrix
        quat = [0]*4  # Convert to quaternion (x, y, z, w)

        # site_* methods to access global coordinates, other * methods (xpos, xquat) access local coordinates
        return ee_pos_absolute, quat
    
    def get_ball_position(self):
        return self.data.geom_xpos[self.ball_id]
    
    def get_paddle_position(self):
        return self.data.site_xpos[self.paddle_id]

    def update_target_pos(self):
        self.target_pos = self.data.xpos[self.target_id]
        return self.target_pos

        
def generate_random_point_in_shell(cx, cy, cz, r1, r2):
    """Generate a random point inside a sphere of radius r1 but outside a smaller sphere of radius r2."""
    
    assert r2 < r1, "r2 must be smaller than r1"
    
    # Generate a random point in the larger sphere
    u = np.random.uniform(0, 1)  # Random scaling factor for radius
    theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    phi = np.arccos(np.random.uniform(-1, 1))  # Polar angle (uniform on sphere)
        
    # Convert to Cartesian coordinates
    r = (r2 + (r1 - r2) * u)  # Ensure radius is between r2 and r1
    x = r * np.sin(phi) * np.cos(theta) + cx
    y = r * np.sin(phi) * np.sin(theta) + cy
    z = r * np.cos(phi) + cz

    return x, y, z

import time
import mujobot
if __name__ == "__main__":
    env = gym.make("mujobot/ur5-paddle-v1", render_mode="human")
    for _ in range(100):
        print("Resetting")
        obs = env.reset()
        for _ in range(1000):
            action = [0]*6
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    env.close()