from typing import List, Tuple
from rlbench.backend.task import Task

from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class ReachBlueRed(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.target_blue = Shape('blue_block')
        self.register_graspable_objects([self.target_blue])
        success_detector_blue = ProximitySensor('Proximity_sensor_blue')
        success_condition_blue = DetectedCondition(self.robot.arm.get_tip(), success_detector_blue)

        self.target_red = Shape('red_block')
        self.register_graspable_objects([self.target_red])
        success_detector_red = ProximitySensor('Proximity_sensor_red')
        success_condition_red = DetectedCondition(self.robot.arm.get_tip(), success_detector_red)
        self.register_success_conditions([success_condition_blue,success_condition_red])

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        
        return ['reach blue block']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return [0.0,0.0,0.0],[0.0,0.0,0.0]