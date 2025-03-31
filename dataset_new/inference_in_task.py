import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachBlueBlock
from scipy.spatial.transform import Rotation as R

# task_name = ReachBlueBlock
# Initialize action mode and environment
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False, collision_checking=True),
    gripper_action_mode=Discrete()
)
env = Environment(action_mode)
env.launch()

task = env.get_task(ReachBlueBlock)

training_steps = 120
episode_length = 40
descriptions, obs = task.reset()
for i in range(training_steps):

    if i % episode_length == 0:
        print('Reset Episode')
        obs = task.reset()

    # Generate random XYZ deltas
    xyz = np.random.uniform(low=-0.01, high=0.01, size=3)

    # Generate random unit quaternion
    r = R.random()
    quat = r.as_quat()  # Format: [qx, qy, qz, qw]

    # Gripper action: open (1) or close (-1)
    gripper_action = np.random.choice([-1, 1])

    # Combine into action vector
    action = np.concatenate([xyz, quat, [gripper_action]])

    # Apply action
    obs, reward, terminate = task.step(action)

#
print('Done')
env.shutdown()


