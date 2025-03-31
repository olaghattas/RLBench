import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock
from scipy.spatial.transform import Rotation as R
import pickle
import numpy as np

file_path_ = "/dataset_new/reach_blue_block_5/variation0/episodes/episode0/low_dim_obs.pkl"
with open(file_path_, 'rb') as file:
    data = pickle.load(file)


import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


gripper_pose_list = []
prev_joint_positions = None
# Loop over all instances (observations) in the dataset
for obs in data._observations:
    # Extract the gripper matrix (it is a 4x4 matrix) column major
    gripper_pose_list.append(obs.gripper_pose)

gripper_pose = np.array(gripper_pose_list)


# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot EE positions
ax.plot(gripper_pose[:, 0], gripper_pose[:, 1], gripper_pose[:, 2], label='Gripper Position', marker='o', color='b')


ax.set_title('3D Plot of End Effector Position and Actions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()


