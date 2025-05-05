#!/usr/bin/env python
import os
import sys
import time
import torch
import dill
import numpy as np
import collections
import cv2
import pyrep
import rlbench
import tkinter as tk
from tkinter import ttk
from collections import defaultdict, deque

# -------------------------------
# Diffusion Policy Imports
# -------------------------------
sys.path.append("/home/olagh48652/policy_training/diffusion_policy")
from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply
from scipy.spatial.transform import Rotation


from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock, ReachRedBlock, ReachCoffee, ReachGrocery
from scipy.spatial.transform import Rotation as R

# -------------------------------
# Helper Functions
# -------------------------------
class FrameStackForTrans:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.obs_history = {}

    def reset(self, init_obs):
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = deque([init_obs[k][None] for _ in range(self.num_frames)], maxlen=self.num_frames,)
        obs = {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
        return obs 

    def add_new_obs(self, new_obs):
        for k in new_obs:
            if 'timesteps' in k or 'actions' in k: continue
            self.obs_history[k].append(new_obs[k][None])

        obs = {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
        return obs

class BaseRawObsDictGenerator:
    def __init__(self, *args, **kwargs):
        self.last_obs_dict = None

    def get_raw_obs_dict(self, state_info):
        obs_dict = {}
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    
class RobotStateRawObsDictGenerator(BaseRawObsDictGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, obs_dict, key, value, check_valid=True):
        if (
            (
                np.sum(np.abs(value)) == 0.0
                and key in ["ee_states", "joint_states", "gripper_states"]
            )
            and check_valid
            and self.last_obs_dict is not None
        ):
            value = self.last_obs_dict[key]
        obs_dict[key] = value

    def get_raw_obs_dict(self, current_obs):
        obs_dict = {}    
        try:
            ee_states = current_obs.gripper_matrix.T.flatten().astype(np.float32)
            gripper_states = np.array(current_obs.gripper_open, dtype=np.float32)
            joint_states = np.array(current_obs.joint_positions, dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to process observation: {str(e)}")
            return {}
        
        self.load(obs_dict, "ee_states", ee_states)
        self.load(obs_dict, "joint_states", joint_states)
        self.load(obs_dict, "gripper_states", gripper_states, check_valid=False)

        for state in ["ee_states", "joint_states", "gripper_states"]:
            if (
                np.sum(np.abs(obs_dict[state])) <= 1e-6
                and self.last_obs_dict is not None
            ):
                obs_dict[state] = self.last_obs_dict[state]
        self.last_obs_dict = obs_dict
        return obs_dict

raw_obs_dict_generator = RobotStateRawObsDictGenerator()

def get_current_obs(current_obs):
    try:
        obs_dict = raw_obs_dict_generator.get_raw_obs_dict(current_obs)
        obs_dict['agentview_rgb'] = current_obs.front_rgb.transpose(2, 0, 1).astype(np.float32)/255.0
        obs_dict['eye_in_hand_rgb'] = current_obs.wrist_rgb.transpose(2, 0, 1).astype(np.float32)/255.0
        return obs_dict
    except Exception as e:
        print(f"[ERROR] Failed in get_current_obs: {str(e)}")
        return {}

rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

def undo_transform_action(action):
    pos = action[..., :3]
    rot_6d = action[..., 3:9]
    gripper = action[..., -1:]
    rot = rotation_transformer.inverse(rot_6d)
    rot_sq = rot.squeeze()
    uaction_axis_angle = np.concatenate([pos, rot, gripper], axis=-1)
    quat = Rotation.from_rotvec(rot_sq).as_quat()
    quat = np.expand_dims(quat, axis=0)
    uaction_quat = np.concatenate([pos, quat, gripper], axis=-1)
    return uaction_quat, uaction_axis_angle

# -------------------------------
# Global Variables
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_paths = {
    "Checkpoint coffee": "/home/olagh48652/Downloads/coffee_180.ckpt",
    "600_epoch_128_coffee_cracker": "/home/olagh48652/Downloads/lang_600_epoch_128_coffee_cracker.ckpt",
    "coffee_cracker_128": "/home/olagh48652/Downloads/coffee_cracker_128_after_train_1000_epochs.ckpt"
}

# Initialize variables
framestacker = None
policy = None
task = None
env = None
obs_env = None

def checkpoint_selected(event):
    global policy, env, task, framestacker, obs
    selected = checkpoint_var.get()
    checkpoint = checkpoint_paths.get(selected, "Unknown checkpoint")
    with open(checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload['cfg']
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=True),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode=action_mode, obs_config=ObservationConfig(), headless=False)
    env.launch()
    task = env.get_task(ReachGrocery)
    descriptions, obs_env = task.reset()
    framestacker = FrameStackForTrans(getattr(cfg, "dataset_obs_steps", 2))

    print("Diffusion policy loaded and set to eval mode.")
    print(obs_env)
    if obs_env is not None:
        print("updated")
        obs = get_current_obs(obs_env)
        # # (Mapping: ensure that any additional keys required by your model are present.)
        obs = framestacker.reset(obs)

def run_demo():
    global obs, obs_env, task
    keys_select = ['agentview_rgb', 'joint_states', 'ee_states', 'eye_in_hand_rgb', 'gripper_states']

    done = 0
    max_steps = 30
    step = 0
    while (not done and max_steps > step):
        np_obs_dict = {key: obs[key][None, :] for key in keys_select if key in obs}
        obs_tensor = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))
        
        with torch.no_grad():
            action_dict = policy.predict_action(obs_tensor)
            
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
        env_action = np_action_dict['action']
        env_action, env_action_rot = undo_transform_action(env_action)
        env_action = env_action.squeeze()
        env_action_rot = env_action_rot.squeeze()

        for action in env_action[:2]:
            try:
                obs_env, reward, terminate = task.step(action)
            except (pyrep.errors.ConfigurationPathError, rlbench.backend.exceptions.InvalidActionError) as e:
                print(f"Error encountered: {e}")
                descriptions, obs_env = task.reset()
                done = 1
            obs = get_current_obs(obs_env)
            obs = framestacker.add_new_obs(obs)
            success = task._task.success()
            if any(success):
                done = 1  
                print("demo successful") 
                return      
        step += 1

def shutdown():
    global env
    print('Done')
    if env:
        env.shutdown()

def reset():
    global descriptions, obs_env
    descriptions, obs_env = task.reset()
    if obs_env is not None:
        print("updated")
        obs = get_current_obs(obs_env)
        # # (Mapping: ensure that any additional keys required by your model are present.)
        obs = framestacker.reset(obs)
# -------------------------------
# Main Tkinter GUI Setup
# -------------------------------
root = tk.Tk()
root.title("RLBench Control Panel")
root.geometry("300x200")

checkpoint_var = tk.StringVar()

# Dropdown menu for checkpoints
checkpoint_label = ttk.Label(root, text="Select Checkpoint:")
checkpoint_label.pack(pady=(10, 0))

checkpoint_dropdown = ttk.Combobox(
    root,
    textvariable=checkpoint_var,
    values=["Checkpoint coffee", "600_epoch_128_coffee_cracker", "coffee_cracker_128"]
)
checkpoint_dropdown.bind("<<ComboboxSelected>>", checkpoint_selected)
checkpoint_dropdown.pack(pady=(10, 10))

run_button = ttk.Button(root, text="Run Demo", command=run_demo)
run_button.pack(pady=(5, 10))

shutdown_button = ttk.Button(root, text="Shutdown", command=shutdown)
shutdown_button.pack(pady=(5, 10))

reset_button = ttk.Button(root, text="Reset", command=reset)
reset_button.pack()

root.mainloop()
