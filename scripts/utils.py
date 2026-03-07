from collections import deque
import numpy as np

class MovingAverageFilter:
    def __init__(self, max_len=5):
        self.value_history = deque(maxlen=max_len)
        
    def compute(self, value):
        self.value_history.append(value)
        return np.mean(self.value_history)