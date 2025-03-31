import pickle
import rlbench.backend.task as task
from rlbench import ObservationConfig

from rlbench.backend.observation import Observation



# file_path_ = "/home/olagh48652/RLBench/dataset/reach_blue_block/variation0/episodes/episode0/low_dim_obs.pkl"
file_path_ = "/dataset_new/reach_blue_block_5/variation0/episodes/episode0/low_dim_obs.pkl"

with open(file_path_, 'rb') as f:
    data_ = pickle.load(f)
    
print(type(data_))
print(len(data_._observations))

# for i, obs in enumerate(data_._observations[:5]):  # Inspect first 5 observations
#     print(f"Observation {i}: {obs.__dict__}")
    
    
obs_dict =  data_._observations[0].__dict__ 
print(type(obs_dict))    
# print(f"Observation0 : {obs.__dict__}")
# print("Keys in Observation object:")
# for key in obs_dict.keys():
#     print(f"key: {key} type: {type(key)}")
  
from pprint import pprint

print("Full Observation object:")
pprint(obs_dict)

# Extracting the required values
# joint_velocities = obs_dict['joint_velocities']
# joint_positions = obs_dict['joint_positions']
# gripper_pose = obs_dict['gripper_pose']
# gripper_open = obs_dict['gripper_open']
#
# # Printing values
# print("Joint Velocities:", joint_velocities)
# print("Joint Positions:", joint_positions)
# print("Gripper Pose:", gripper_pose)
# print("Gripper Open:", gripper_open)

print("left_shoulder_rgb :", obs_dict['left_shoulder_rgb'])