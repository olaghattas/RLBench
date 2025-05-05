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

class ReachCoffee(Task):

    def init_task(self) -> None:
        
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        self.grasp_points = Dummy('coffee_grasp_point')
        
        self.waypoint0 = Dummy('waypoint0')
        self.boundary = SpawnBoundary([Shape('workspace')])
        
        success_detector = ProximitySensor('coffee_proximity_sensor')
        self.success_condition = DetectedCondition(self.robot.arm.get_tip(), success_detector)
        self.register_success_conditions([self.success_condition])

        
    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        print("*"*40)
        print("INIT")
        print("*"*40)
        
        [self.boundary.sample(g, min_distance=0.1) for g in self.groceries]
        
    
        self.waypoint0.set_pose(self.grasp_points.get_pose())  
        
        self.register_success_conditions([self.success_condition])
        
        return ['reach the coffee on the table']

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self) -> bool:
        return True
    
    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return (0.0, 0.0, -1.), (0.0, 0.0, 1.)

