import random
from collections import defaultdict

class BufferManager:
    def __init__(self, max_per_class=20):
        self.max_per_class = max_per_class
        self.buffer = defaultdict(list)  # key: task_id, value: list of (sample, label)

    def add_batch(self, data, labels, task_id):
        for sample, label in zip(data, labels):
            if len(self.buffer[task_id]) < self.max_per_class:
                self.buffer[task_id].append((sample.cpu(), int(label)))
            else:
                idx = random.randint(0, self.max_per_class - 1)
                self.buffer[task_id][idx] = (sample.cpu(), int(label))

    def get_buffer(self):
        return dict(self.buffer)

    def get_task_samples(self, task_id):
        return self.buffer[task_id]