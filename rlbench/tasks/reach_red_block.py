from typing import List
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

class ReachRedBlock(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.target = Shape('red_block')
        self.register_graspable_objects([self.target])
        success_detector = ProximitySensor('Proximity_sensor')
        success_condition = DetectedCondition(self.robot.arm.get_tip(), success_detector)
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return ['']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass
