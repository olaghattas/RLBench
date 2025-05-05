import numpy as np
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachGrocery
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

task = env.get_task(ReachGrocery)

# Check if the success conditions are met
# success = ReachBlueBlock.check_success()

task_ = task._task
# success = task_.success()
# Print the result
# print("Success:", success)
descriptions, obs = task.reset()
my_shapes_list = task_.groceries
print(task_.groceries)
# List to collect all gripper poses
pose_data = []

print("pose", task._task.robot.arm.get_pose())
for shape in my_shapes_list:
    print(shape)
    # print(dir(shape.__dict__))
    pose = shape.get_pose()
    print("Pose:", pose)
    
    new_pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0] 
    shape.set_pose(new_pose) 
    print("New Pose:", shape.get_pose())
    
    print("Shape Name: ", shape.get_name())
    print("Shape Name Type: ", type(shape.get_name()))


# for shape in my_shapes_list:
#     print(f"\n--- Shape: {shape.get_name()} ---")
#     for attr in dir(shape):
#         if not attr.startswith("_"):  # Skip private/internal attributes
#             try:
#                 value = getattr(shape, attr)
#                 if callable(value):
#                     # Try calling if it's a method with no required arguments
#                     try:
#                         result = value()
#                         print(f"{attr}(): {result}")
#                     except TypeError:
#                         print(f"{attr}(): <method requires arguments>")
#                 else:
#                     print(f"{attr}: {value}")
#             except Exception as e:
#                 print(f"{attr}: <error: {e}>")
                
# Loop over all instances (observations) in the dataset
# for obs_ in data._observations:
#     # Extract the gripper matrix (it is a 4x4 matrix) column major
#     # print(f"Available attributes in obs: {dir(obs)}")

#     print("gripper_pose", obs_.gripper_pose)
#     # Append the pose as a list of values
#     pose_data.append(obs_.gripper_pose)
    

#     gripper_action = 1
#     action = np.concatenate([obs_.gripper_pose, [gripper_action]])
#     obs, reward, terminate = task.step(action)
#     # success = task_.success()
#     # print("Success:", success)
#     # for condition in ReachBlueBlock.success_conditions:
#     #     print("Success condition not met")
#     #     if condition.condition_met():
#     #         print("Success condition met! Task completed.")
#     #         done = True



# Create DataFrame from all poses
# df = pd.DataFrame(pose_data, columns=[f'pose_{i}' for i in range(len(pose_data[0]))])


# Save to Excel
# df.to_excel('gripper_pose.xlsx', index=False)

print('Done')
env.shutdown()
