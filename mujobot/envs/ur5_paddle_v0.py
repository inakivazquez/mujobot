from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
import mujoco
import numpy as np
import math
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
import os



class UR5PaddleEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        #self.model = mujoco.MjModel.from_xml_path(ur5e_mj_description.PACKAGE_PATH+"/scene.xml")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = mujoco.MjModel.from_xml_path(current_dir + "/ur5_paddle/scene.xml")

        # Create a data structure to hold the simulation state
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        if self.render_mode == "human":
            # Create a viewer to visualize the simulation
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2]*3 + [-2*math.pi]*6, dtype=np.float32), high=np.array([+2]*3 + [+2*math.pi]*6, dtype=np.float32), shape=(9,))

        # Action space is 6 joint angles and the gripper open/close level (0 to 1)
        action_max = 2*math.pi / 10
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6, dtype=np.float32), high=np.array([+action_max]*6, dtype=np.float32), shape=(6,))
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        self.paddle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "paddle")


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)  # Fully resets simulation state

        self.data.qpos[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]
        # Starting position
        self.data.ctrl[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]

        # Set the new position and force of the ball 
        random_diff = np.random.uniform(low=[-0.1, -0.09, 0], high=[+0.09, +0.15, 0])
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        qpos_idx = self.model.jnt_qposadr[ball_joint_id]  # Get the index in qpos
        self.data.qpos[qpos_idx:qpos_idx+3] = random_diff
        self.data.xfrc_applied[self.ball_id, :3] = np.random.uniform(low=-0.5, high=0.5, size=3)

        self.wait_until_stable()

        return self.get_observation(), {}

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

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        action[0:6] = self.data.ctrl[0:6] + action[0:6]

        action[0:6] = np.clip(action[0:6], -2*math.pi, +2*math.pi)

        self.data.ctrl[0:6] = action
        self.data.ctrl[6] = 255 # Gripper closed at all times
        self.wait_until_stable()

        #self.update_target_pos()

        ball_pos = self.get_ball_position()
        paddle_pos = self.get_paddle_position()
        reward = +1 # Keep alive reward
        terminated = False
        if paddle_pos[2] < ball_pos[2]: # Ball is below the paddle, fallen
            reward = -10
            terminated = True
        truncated = False
        obs = self.get_observation()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.viewer.sync()

    def close(self):
        if self.render_mode == 'human':
            self.viewer.close()

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
        return self.data.xpos[self.ball_id]
    
    def get_paddle_position(self):
        return self.data.xpos[self.paddle_id]

    def update_target_pos(self):
        self.target_pos = self.data.xpos[self.target_id]
        return self.target_pos
    
    def draw_sphere(self, center, radius, color=(1, 0, 0, 1)):
        """Draws a debug sphere at a given 3D position in MuJoCo."""
        self.viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0, 0],
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=color
            )
        self.viewer.user_scn.ngeom = 1
        
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
if __name__ == "__main__":
    env = UR5PaddleEnv(render_mode="human")
    for _ in range(100):
        print("Resetting")
        obs = env.reset()
        for _ in range(1000):
            action = [0]*6
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    env.close()