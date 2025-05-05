from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary

GROCERY_NAMES = [
    'crackers',
    'coffee'
]


class ReachGrocery(Task):

    def init_task(self) -> None:  
        self.boundary = SpawnBoundary([Shape('workspace')])
        
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        # Create proximity sensors
        # self.success_detector_crackers = ProximitySensor('crackers_proximity_sensor')
        # self.success_detector_coffee = ProximitySensor('coffee_proximity_sensor')

        # # Create success conditions
        # self.condition_1 = DetectedCondition(self.robot.arm.get_tip(), self.success_detector_crackers)
        # self.condition_2 = DetectedCondition(self.robot.arm.get_tip(), self.success_detector_coffee)
        
        # Create proximity sensors
        self.success_detector_gripper = ProximitySensor('Panda_gripper_attachProxSensor')

        # Create success conditions
        # crackers
        self.condition_1 = DetectedCondition(self.groceries[0], self.success_detector_gripper)
        # coffee
        self.condition_2 = DetectedCondition(self.groceries[1], self.success_detector_gripper)

        # Register BOTH conditions
        self.register_success_conditions([self.condition_1, self.condition_2])

        # Initialize success_cause
        self.success_cause = None
        

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        print("*"*40)
        print("INIT")
        print("*"*40)

        [self.boundary.sample(g, min_distance=0.1) for g in self.groceries]

        return []

    def variation_count(self) -> int:
        return 1

    
    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return (0.0, 0.0, -1.), (0.0, 0.0, 1.)

    def success(self) -> Tuple[bool, bool]:
        cond1 = self.condition_1.condition_met()
        cond2 = self.condition_2.condition_met()
        if cond1[0]:
            self.success_cause = "cracker reached"
            print("[SUCCESS] Cracker proximity sensor triggered!")
            return True, False
        elif cond2[0]:
            self.success_cause = "coffee reached"
            print("[SUCCESS] Coffee proximity sensor triggered!")
            return True, False
        else:
            # if self.success_cause is not None:
            #     print("[DEBUG] Lost success condition. Resetting success_cause.")
            # self.success_cause = None
            return False, False

    def get_success_cause(self) -> str:
        return self.success_cause