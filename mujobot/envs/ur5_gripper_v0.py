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



class UR5GripperEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        #self.model = mujoco.MjModel.from_xml_path(ur5e_mj_description.PACKAGE_PATH+"/scene.xml")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = mujoco.MjModel.from_xml_path(current_dir + "/ur5_gripper/scene.xml")

        # Create a data structure to hold the simulation state
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        if self.render_mode == "human":
            # Create a viewer to visualize the simulation
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-1]*3 + [-2*math.pi]*6, dtype=np.float32), high=np.array([+1]*3 + [+2*math.pi]*6, dtype=np.float32), shape=(9,))

        # Action space is 6 joint angles and the gripper open/close level (0 to 1)
        action_max = 2*math.pi / 10
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6+[0]), high=np.array([+action_max]*6+[1]), shape=(7,))
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)  # Fully resets simulation state
        self.target_pos = np.random.uniform(-0.5, 0.5, size=3)
        self.target_pos[2] = 0.7
        self.model.body_pos[self.target_id] = self.target_pos  # Update position

        qpos_addr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")]

        self.data.qpos[qpos_addr:qpos_addr+3] = self.target_pos  # Teleport


        #if self.render_mode == "human":
            #self.draw_sphere(self.target_pos, 0.025)
        return self.get_observation(), {}

    def wait_until_stable(self, sim_steps=500):
        joint_pos = self.get_observation()[3:]
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()
            new_joint_pos = self.get_observation()[3:]
            if np.sum(np.abs(np.array(joint_pos)-np.array(new_joint_pos))) < 5e-3: # Threshold based on experience
                return True
            joint_pos = new_joint_pos
        print("Warning: The robot configuration did not stabilize")
        return False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        action[0:6] = self.data.ctrl[0:6] + action[0:6]
        action[6] = 0 if action[6] < 0.5 else 255

        self.data.ctrl[:] = action
        self.data.ctrl[:] = np.clip(self.data.ctrl, -2*math.pi, +2*math.pi)
        self.wait_until_stable()

        self.update_target_pos()

        obs = self.get_observation()
        distance = np.linalg.norm(obs[0:3])
        reward = -distance
        terminated = (distance < 0.05)
        if terminated:
            reward += 10
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.viewer.sync()

    def close(self):
        if self.render_mode == 'human':
            self.viewer.close()

    def get_observation(self):
        ee_pos, ee_quat = self.get_ee_pose()
        joint_pos = self.data.qpos[0:6]
        distance_vector = (self.target_pos - ee_pos)
        return np.concatenate((distance_vector, joint_pos), dtype=np.float32)

    def get_ee_pose(self):
        pinch_name = "pinch"
        pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pinch_name)
        ee_pos_absolute = self.data.site_xpos[pinch_id]
        rotation_matrix = self.data.xmat[pinch_id].reshape(3, 3)  # Extract 3×3 rotation matrix
        quat = [0]*4  # Convert to quaternion (x, y, z, w)

        # site_* methods to access global coordinates, other * methods (xpos, xquat) access local coordinates
        return ee_pos_absolute, quat
    
    def update_target_pos(self):
        self.target_pos = self.model.body_pos[self.target_id]
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
        

import time
if __name__ == "__main__":
    env = UR5GripperEnv(render_mode="human")
    obs = env.reset()
    for _ in range(1000):
        action = [0]*7
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1)
    env.close()