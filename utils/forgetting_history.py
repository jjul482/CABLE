from collections import deque

class ForgettingHistory:
    def __init__(self, num_adapters, history_size, alpha=0.01):
        self.history_size = history_size
        self.histories = [deque(maxlen=history_size) for _ in range(num_adapters)]

    def append(self, forgetting_scores):
        num_adapters = len(forgetting_scores)
        if num_adapters > len(self.histories):
            for _ in range(num_adapters - len(self.histories)):
                self.histories.append(deque(maxlen=self.history_size))
        elif num_adapters < len(self.histories):
            self.histories = self.histories[:num_adapters]
        for i, score in enumerate(forgetting_scores):
            self.histories[i].append(score)

    def get_history(self, adapter_idx):
        return list(self.histories[adapter_idx])

    def get_all_histories(self):
        return [list(h) for h in self.histories]

    def get_mean_scores(self):
        return [sum(h)/len(h) if len(h) > 0 else 0.0 for h in self.histories]
    
    def p_value(self, new_scores):
        pvals = []
        for idx, new_score in enumerate(new_scores):
            history = self.get_history(idx)
            if not history:
                pvals.append(1.0)
            else:
                count = sum(1 for h in history if h >= new_score)
                pvals.append(count / len(history))
        return pvals