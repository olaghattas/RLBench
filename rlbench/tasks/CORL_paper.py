from typing import List
from rlbench.backend.task import Task


class CORLPaper(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        pass

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
