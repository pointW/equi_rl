from storage.buffer import QLearningBuffer
from utils.torch_utils import ExpertTransition, augmentTransition
from utils.parameters import buffer_aug_type

class QLearningBufferAug(QLearningBuffer):
    def __init__(self, size, aug_n=9):
        super().__init__(size)
        self.aug_n = aug_n

    def add(self, transition: ExpertTransition):
        super().add(transition)
        for _ in range(self.aug_n):
            super().add(augmentTransition(transition, buffer_aug_type))





