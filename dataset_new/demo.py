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

obs = data._observations

# Initialize action mode and environment
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=True),
    gripper_action_mode=Discrete()
)

env = Environment(action_mode)
env.launch()

task = env.get_task(ReachBlueBlock)

descriptions, obs = task.reset()


for i in range(1, len(data._observations)):
    obs = data._observations[i]
    gripper_state = [obs.gripper_open]
    action = np.concatenate([obs.gripper_pose, [gripper_state]])
    # Apply action
    print(f"obs.gripper_pose {obs.gripper_pose}")
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()