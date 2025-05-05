#!/usr/bin/env python
import os
import sys
import time
import torch
import dill
import numpy as np
import collections
import cv2

# -------------------------------
# Diffusion Policy Imports
# -------------------------------
sys.path.append("/home/olagh/policy_training/diffusion_policy")
from diffusion_policy.workspace.train_diffusion_transformer_lang_hybrid_workspace import TrainDiffusionTransformerLangHybridWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.pytorch_util import dict_apply
from scipy.spatial.transform import Rotation

# -------------------------------
# Helper Functions
# -------------------------------
from collections import defaultdict, deque


class FrameStackForTrans:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.obs_history = {}
        
    def reset(self, init_obs):
        """ 
        init_obs: dict of initial observation at the start of the episode
        return stacked obs by repeating the first observation num_frames times
        """
        
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = deque([init_obs[k][None] for _ in range(self.num_frames)], maxlen=self.num_frames,)
        obs = { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }
        return obs 

    def add_new_obs(self, new_obs):
        """ 
        new_obs: dict of new observation at current timestep
        return stacked obs
        """
        for k in new_obs:
            if 'timesteps' in k or 'actions' in k: continue
            self.obs_history[k].append(new_obs[k][None])

        obs= { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }
        return obs 

class BaseRawObsDictGenerator:
    def __init__(self, *args, **kwargs):
        self.last_obs_dict = None

    def get_raw_obs_dict(self, state_info):
        """
        Args:
           state_info (dict): A dictionary of robot state + images
        """
        obs_dict = {}
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
    
class RobotStateRawObsDictGenerator(BaseRawObsDictGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, obs_dict, key, value, check_valid=True):
        """
        This is to check if the data is correct or not. Sometimes the data will be all zero depending on the networking conditions.
        """
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
        print(f"\n[Debug] Entering get_raw_obs_dict, current_obs type: {type(current_obs)}")
            
        obs_dict = {}    
        try:
            # Debug print object structure
            print(f"[Debug] current_obs attributes: {dir(current_obs)}")
            
            # Check gripper_matrix existence
            if not hasattr(current_obs, 'gripper_matrix'):
                print("[ERROR] Missing gripper_matrix in current_obs!")

            ee_states = current_obs.gripper_matrix.T.flatten().astype(np.float32)
            gripper_states = np.array(current_obs.gripper_open, dtype=np.float32)
            joint_states = np.array(current_obs.joint_positions, dtype=np.float32)
            
            print(f"[Debug] ee_states shape: {ee_states.shape}")
            print(f"[Debug] gripper_states: {gripper_states}")
            print(f"[Debug] joint_states: {joint_states}")

        except Exception as e:
            print(f"[ERROR] Failed to process observation: {str(e)}")
            return {}
        
        
        self.load(obs_dict, "ee_states", ee_states)
        self.load(obs_dict, "joint_states", joint_states)
        # Gripper width will probably become zero
        self.load(obs_dict, "gripper_states", gripper_states, check_valid=False)

        
        for state in ["ee_states", "joint_states", "gripper_states"]:
            if (
                np.sum(np.abs(obs_dict[state])) <= 1e-6
                and self.last_obs_dict is not None
            ):
                print(f"{state} missing!!!!")
                obs_dict[state] = self.last_obs_dict[state]
        self.last_obs_dict = obs_dict
        return obs_dict


raw_obs_dict_generator = RobotStateRawObsDictGenerator()
def get_current_obs(current_obs):
    print(f"\n[Debug] Entering get_current_obs, current_obs type: {type(current_obs)}")
        
    try:
        obs_dict = raw_obs_dict_generator.get_raw_obs_dict(current_obs)
        print("[Debug] Raw obs_dict keys:", obs_dict.keys())
        
        # Image processing checks
        print(f"[Debug] front_rgb type: {type(current_obs.front_rgb)}, shape: {current_obs.front_rgb.shape}")
        print(f"[Debug] wrist_rgb type: {type(current_obs.wrist_rgb)}, shape: {current_obs.wrist_rgb.shape}")

        obs_dict['agentview_rgb'] = current_obs.front_rgb.transpose(2, 0, 1).astype(np.float32)/255.0
        obs_dict['eye_in_hand_rgb'] = current_obs.wrist_rgb.transpose(2, 0, 1).astype(np.float32)/255.0
        
        print("[Debug] Processed image shapes:")
        print(f"agentview_rgb: {obs_dict['agentview_rgb'].shape}")
        print(f"eye_in_hand_rgb: {obs_dict['eye_in_hand_rgb'].shape}")
        
        return obs_dict
    except Exception as e:
        print(f"[ERROR] Failed in get_current_obs: {str(e)}")
        return {}

# transforms first to second
rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
def undo_transform_action(action):
    # The action is guaranteed to have shape 10
    raw_shape = action.shape
    print(raw_shape)
    # Extract components of the action: position, rotation (axis-angle), and gripper state
    pos = action[..., :3]  # Position: first 3 elements
    rot_6d = action[..., 3:9]  # Rotation: next 3 elements (axis-angle)
    gripper = action[..., -1:]  # Gripper state: last element
    print(f"rot 6d {rot_6d}")
    # Undo the transformation for the rotation
    
    rot = rotation_transformer.inverse(rot_6d)  # Inverse transformation from 'rotation_6d' back to 'axis_angle'
    
    # print(f"rot: {rot}")
    rot = rot.squeeze() 
    # print(f"rot: {rot}")
    
    # Convert the axis-angle rotation to a quaternion
    quat = Rotation.from_rotvec(rot).as_quat()  # [x, y, z, w]
    # print(f"quat: {quat.shape}, {quat}")   
    # Concatenate position, quaternion, and gripper states
    quat = np.expand_dims(quat, axis=0) 
    # print(f"quat: {quat.shape}, {quat}") 
    uaction = np.concatenate([pos, quat, gripper], axis=-1)

    # print(f"uaction: {uaction.shape}, {uaction}")

    return uaction


# -------------------------------
# Set device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Using device:", device)


# -------------------------------
# Load Diffusion Policy Checkpoint
# -------------------------------
print("-"*40)
print(" Loading Checkpoint ")
print("-"*40)

# checkpoint = "/home/olagh/RLBench/dataset_new/epoch_100.ckpt"
# checkpoint = "/home/olagh/policy_training/diffusion_policy/diffusion_policy/data/outputs/training_with_no_lang_2layer_transformer_encoder/checkpoints/after_train_600_epochs.ckpt"
checkpoint = "/home/olagh/policy_training/diffusion_policy/diffusion_policy/data/outputs/lang_fixed_trained/checkpoints/after_train_600_epochs.ckpt"

# checkpoint = "/home/olagh/after_train_600_epochs.ckpt"
with open(checkpoint, 'rb') as f:
    payload = torch.load(f, pickle_module=dill)
cfg = payload['cfg']


workspace = TrainDiffusionTransformerLangHybridWorkspace(cfg, output_dir=None)

workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# # Select the policy model (use EMA model if enabled)
policy = workspace.model

if getattr(cfg.training, "use_ema", False):
    policy = workspace.ema_model
policy.to(device)
policy.eval()
print("Diffusion policy loaded and set to eval mode.")
    
# -------------------------------
# Main Inference Loop
# -------------------------------
n_obs_steps = getattr(cfg, "dataset_obs_steps", 2)
n_action_steps = getattr(cfg, "n_action_steps", 4)
max_steps = 100   # maximum steps per trial

print("Starting trial ")

# -------------------------------
# Setup RLBench Simulator
# -------------------------------

from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock, ReachRedBlock
from scipy.spatial.transform import Rotation as R

headless_val = False
obs_config = ObservationConfig()


# img_size = [640, 480] # 160, 120
#  obs_config.set_all(False)
# obs_config.wrist_camera.set_all(True)
# obs_config.head_camera.set_all(True)
# obs_config.front_camera.set_all(True)
# obs_config.set_all_low_dim(True)

# obs_config.wrist_camera.image_size = img_size
# obs_config.head_camera.image_size = img_size
# obs_config.wrist_camera.depth_in_meters = False
# obs_config.front_camera.image_size = img_size

def encode_lang_string(lang_str: str, max_len: int = 24) -> torch.Tensor:
    lang_str = lang_str.strip('\x00')
    encoded = [ord(c) for c in lang_str[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    return torch.tensor(padded, dtype=torch.int32).unsqueeze(0)  # shape [1, max_len]

print("-"*40)
print(" START TASK ")
print("-"*40)
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=True),
    gripper_action_mode=Discrete()
)
env = Environment(action_mode=action_mode,
                  obs_config=obs_config,
                 headless=headless_val
                    )
env.launch()

task = env.get_task(ReachBlueBlock)
descriptions, obs_env = task.reset()
0

# -------------------------------
# Inference / Rollout Function
# -------------------------------

import pyrep
import rlbench 

# Keys expected by the policy (images and robot state).
keys_select = ['agentview_rgb', 'joint_states', 'ee_states', 'eye_in_hand_rgb', 'gripper_states']
framestacker = FrameStackForTrans(n_obs_steps)

print(obs_env)


if obs_env is not None:
    obs = get_current_obs(obs_env)
    # # (Mapping: ensure that any additional keys required by your model are present.)
    obs = framestacker.reset(obs)


done = False
success = False
step = 0

new_lang = "reach the red block"
lang_tensor = encode_lang_string(new_lang, max_len=24)
print("Type:", type(lang_tensor))
print("Shape:", lang_tensor.shape)
print("Dtype:", lang_tensor.dtype)
lang_tensor = lang_tensor.unsqueeze(0)
print("Type:", type(lang_tensor))
print("Shape:", lang_tensor.shape)
print("Dtype:", lang_tensor.dtype)
success_count = 0
with open('actions.txt', 'w') as file:
    for trial in range(20):
        step = 0
        done = False
        while not done and step < max_steps:
            # Prepare the observation dictionary for the policy.
            if obs_env is None:
                obs_env, reward, terminate = task.step(action)
                # print("@"*20)
                # print("obs_env is None! Retrying...")
            else:
                np_obs_dict = {key: obs[key][None, :] for key in keys_select if key in obs}
                obs_tensor = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device))
                
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_tensor, lang_tensor)
                    
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                
                env_action = np_action_dict['action']
                env_action = undo_transform_action(env_action)
                env_action = env_action.squeeze()

                # adjust slicing as required by your controller
                for action in env_action[:4]: 
                    # Send control command to the robot.
                    try:
                        obs_env, reward, terminate = task.step(action)
                    
                    
                    except (pyrep.errors.ConfigurationPathError, rlbench.backend.exceptions.InvalidActionError) as e:
                        # Catch specific errors and handle them
                        print(f"Error encountered: {e}")
                        
                        # Reset the environment after the error
                        descriptions, obs_env = task.reset()
                        print("Environment reset due to an error.")
                        
                        # Optionally, you can handle any additional recovery or retry logic here.
                        done = True  # or any other flag that controls the loop
    
                    obs = get_current_obs(obs_env)
                    obs = framestacker.add_new_obs(obs)
                    file.write(str(action) + '\n')  # Writing each action on a new line
                    success = task._task.success()
                    print("Success:", success)
                    if any(success):
                        print("sucsess")
                        done = True
                        descriptions, obs_env = task.reset()
                        success_count +=1
                    
                step += 1
                # A simple termination condition: you can add your own task success logic.
                if step >= max_steps:
                    print("execeded_max steps")
                    done = True
                    descriptions, obs_env = task.reset()
                    print("maximum.")
        print("end_trial")

print('Done')
env.shutdown()