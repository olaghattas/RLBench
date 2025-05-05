import h5py
import numpy as np
import os
from PIL import Image
import pickle
from scipy.spatial.transform import Rotation as R
          
# Path to output HDF5 file
file_path = ('reach_100_fixed.hdf5')
tasks = ["reach_blue_block", "reach_red_block"]
path = '/home/olagh48652/RLBench/dataset_new'
internal_files= "variation0/episodes"


count_id = 0
with h5py.File(file_path, 'w') as f:
    # Create the 'mask' group to store task labels
    mask_group = f.create_group('mask')
    
    # Initialize lists to hold the demos for each task
    blue_block_demos = []
    red_block_demos = []
    
    for task in tasks:
        print(f"task: {task}")
        task_path  = os.path.join(path, task, internal_files)

        items = os.listdir(task_path)
        
        for demo_id in range(len(items)) :
            print(f"demo_id: {demo_id}")
            
            # Create or overwrite the group for the demo
            demo_group_name = f'data/demo_{count_id}'
            if demo_group_name in f:
                del f[demo_group_name]  # Remove the existing group
                
            count_id += 1
            demo_group = f.create_group(demo_group_name)

            #### agentview_rgb ####
             # Read the images from the 'front_rgb' folder for each demo
            front_rgb_path = os.path.join(task_path, items[demo_id], "front_rgb")
            # image_files = sorted(os.listdir(front_rgb_path))  # Sort to ensure correct order
            image_files = sorted(os.listdir(front_rgb_path), key=lambda x: int(x.split('.')[0]))
            print(image_files)
            # Initialize a list to store the images
            agentview_rgb_list = []
            
            for image_file in image_files:
                if image_file.endswith('.png'):
                    # Read the image

                    img_path = os.path.join(front_rgb_path, image_file)
                    img = Image.open(img_path)  # You can use any library here (PIL, imageio)
                    img_array = np.array(img)  # Convert to a NumPy array
                    
                    # Append the image to the list
                    agentview_rgb_list.append(img_array)

            # Convert the list of images to a NumPy array
            agentview_rgb = np.array(agentview_rgb_list)
            agentview_rgb = agentview_rgb[:-1]  # Remove last state to align with action size


            #### eye_in_hand_rgb ####
            # Read the images from the 'front_rgb' folder for each demo
            eye_in_hand_path = os.path.join(task_path, items[demo_id], "wrist_rgb")
            image_files = sorted(os.listdir(eye_in_hand_path), key=lambda x: int(x.split('.')[0]))
            # Initialize a list to store the images
            eye_in_hand_rgb_list = []
            
            for image_file in image_files:
                if image_file.endswith('.png'):
                    # Read the image

                    img_path = os.path.join(eye_in_hand_path, image_file)
                    img = Image.open(img_path)  # You can use any library here (PIL, imageio)
                    img_array = np.array(img)  # Convert to a NumPy array
                    
                    # Append the image to the list
                    eye_in_hand_rgb_list.append(img_array)

            # Convert the list of images to a NumPy array
            eye_in_hand_rgb = np.array(eye_in_hand_rgb_list)
            eye_in_hand_rgb = eye_in_hand_rgb[:-1]  # Remove last state to align with action

            #### ee_states ####
            #### joint_states ####
            #### gripper_states ####

            pkl_file = os.path.join(task_path, items[demo_id], "low_dim_obs.pkl")

            with open(pkl_file, 'rb') as file:
                data = pickle.load(file)

            # Initialize a list to store the transformation matrices
            ee_states_list = []
            joint_states_list = []
            gripper_states_list = []

            abs_actions_list = []
            prev_gripper_pose = False

            # Loop over all instances (observations) in the dataset
            for obs in data._observations:
                # Extract the gripper matrix (it is a 4x4 matrix) column major
                gripper_matrix = obs.gripper_matrix
                gripper_matrix_row_major = gripper_matrix.T.flatten()  # Convert to row-major
                # Add to the list
                ee_states_list.append(gripper_matrix_row_major)
                
                joint_states = obs.joint_positions
                joint_states_list.append(joint_states)
                
                gripper_state = obs.gripper_open
                gripper_states_list.append(gripper_state)
                
                # Calculate the delta action (difference in joint positions)
                if prev_gripper_pose:
                    # Extract position (xyz) and quaternion (qx, qy, qz, qw) separately
                    position = obs.gripper_pose[:3]  # Extract XYZ position
                    quaternion = obs.gripper_pose[3:]  # Extract quaternion [qx, qy, qz, qw]

                    # Convert quaternion to roll, pitch, yaw (RPY)
                    rotation = R.from_quat(quaternion)  # Create rotation object from quaternion
                    rot_vec = rotation.as_rotvec()

                    # Combine position, RPY, and gripper action into one action vector
                    abs_action = np.concatenate([position, rot_vec, [gripper_state]])
                    abs_actions_list.append(abs_action)
                
                # Update previous joint positions
                prev_gripper_pose = True

            # Convert the list to a numpy array (optional, for better storage efficiency)
            ee_states_np = np.array(ee_states_list)
            gripper_states_np = np.array(gripper_states_list).reshape(-1, 1)
            joint_states_np = np.array(joint_states_list)


            # Remove the last state from each array (align the action and state size)
            ee_states_np = ee_states_np[:-1]
            gripper_states_np = gripper_states_np[:-1]  # Remove last gripper state
            joint_states_np = joint_states_np[:-1]  # Remove last joint state

            abs_actions_np = np.array(abs_actions_list)
            
            # Now, store the images in the HDF5 file
            obs_group = demo_group.create_group('obs')
            # act_group = demo_group.create_group('actions')

            # Create the 'agentview_rgb' dataset and store the images
            obs_group.create_dataset('agentview_rgb', data=agentview_rgb)
            obs_group.create_dataset('eye_in_hand_rgb', data=eye_in_hand_rgb)
            obs_group.create_dataset('ee_states', data=ee_states_np)
            obs_group.create_dataset('gripper_states', data=gripper_states_np)
            obs_group.create_dataset('joint_states', data=joint_states_np)
            
            demo_group.create_dataset('actions', data=abs_actions_np)
            
            # Populate the mask based on task (if demo is related to blue or red block)
            if "blue" in task:  # Example condition for blue block task
                blue_block_demos.append(f'demo_{count_id}')
            elif "red" in task:  # Example condition for red block task
                red_block_demos.append(f'demo_{count_id}')


    # Convert the lists to numpy arrays to store them in datasets
  
    blue_block_demos = np.array(blue_block_demos, dtype='S6')
    red_block_demos = np.array(red_block_demos, dtype='S6')

    # Create the mask datasets with the populated demo lists
    mask_group.create_dataset('blue_block', data=blue_block_demos)
    mask_group.create_dataset('red_block', data=red_block_demos)

    print(f"HDF5 file '{file_path}' has been created with task labels.")