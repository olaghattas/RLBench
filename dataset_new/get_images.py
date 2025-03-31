import numpy as np
import cv2  
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock
import pickle
from rlbench.observation_config import ObservationConfig


file_path_ = "/home/olagh48652/RLBench/dataset_new/reach_blue_block_5/variation0/episodes/episode0/low_dim_obs.pkl"
with open(file_path_, 'rb') as file:
    data = pickle.load(file)
obs_config = ObservationConfig()

# img_size = [640, 480] # 160, 120
# obs_config.set_all(False)
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



env = Environment(action_mode, obs_config=obs_config)
env.launch()

task = env.get_task(ReachBlueBlock)

descriptions, obs = task.reset()

# Loop over all instances (observations) in the dataset
# for obs_ in data._observations:
for i in range(2):
    # Extract the gripper matrix (it is a 4x4 matrix) column major
    obs_ = data._observations[i]
    # Get image and state here and wait for keyboard input to continue
    # You can use get_obs to get all images and states from the simulator
    current_obs = obs

    # Extract images
    # shape (128, 128, 3)
    # color RBG
    # uint8 (0-255)
    front_img = current_obs.front_rgb
    wrist_img = current_obs.wrist_rgb
    
    agentview_img = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
    eye_in_hand_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

    # Save images to files instead of showing them
    cv2.imwrite('front_camera.png', cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite('wrist_camera.png', cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB))

    # Extract robot state
    joint_positions = current_obs.joint_positions
    gripper_open = current_obs.gripper_open
    print("gripper_matrix", current_obs.gripper_matrix)
   
    
    
    print(f"Joint Positions: {joint_positions}")
    print(f"Gripper Open: {gripper_open}")
    # print(f"gripper_matrix : {ee_states}")
            

    # Take action
    gripper_action = 1
    action = np.concatenate([obs_.gripper_pose, [gripper_action]])
    obs, reward, terminate = task.step(action)
    

# Cleanup
print('Done')
env.shutdown()
