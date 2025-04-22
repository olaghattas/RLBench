import numpy as np
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock
from scipy.spatial.transform import Rotation as R
import pickle

import pandas as pd

file_path_ = "/home/olagh48652/RLBench/dataset_new/reach_red_block_5/variation0/episodes/episode0/low_dim_obs.pkl"
with open(file_path_, 'rb') as file:
    data = pickle.load(file)

headless_val = True
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

descriptions, obs = task.reset()
# List to collect all gripper poses
pose_data = []

# Loop over all instances (observations) in the dataset
for obs_ in data._observations:
    # Extract the gripper matrix (it is a 4x4 matrix) column major
    # print(f"Available attributes in obs: {dir(obs)}")
    print("gripper_pose", obs_.gripper_pose)
    # Append the pose as a list of values
    pose_data.append(obs_.gripper_pose)
    

    gripper_action = 1
    action = np.concatenate([obs_.gripper_pose, [gripper_action]])
    obs, reward, terminate = task.step(action)

# Create DataFrame from all poses
df = pd.DataFrame(pose_data, columns=[f'pose_{i}' for i in range(len(pose_data[0]))])


# Save to Excel
df.to_excel('gripper_pose.xlsx', index=False)
print('Done')
env.shutdown()
