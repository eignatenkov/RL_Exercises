import numpy as np

PHEAD = 0.4


class GamblersProblem:
    def __init__(self, phead = 0.4):
        self.values = np.zeros(101)
        self.values[100] = 1
        self.phead = phead
        self.policy = np.zeros(101)

    def iterate_values(self):
        delta = 0
        for i in range(1, 100):
            max_bid = min(i, 100 - i)
            old_v = self.values[i]
            max_value = 0
            for bid in range(1, max_bid + 1):
                bid_value = (1 - self.phead) * self.values[i - bid] + self.phead * self.values[i + bid]
                max_value = max(max_value, bid_value)
            self.values[i] = max_value
            delta = max(delta, abs(old_v - max_value))
        return delta

    def solve(self, theta=1e-06):
        delta = 1
        while delta > theta:
            delta = self.iterate_values()

    def output_policy(self):
        for i in range(1, 100):
            max_bid = min(i, 100 - i)
            max_value = 0
            best_bid = 0
            for bid in range(1, max_bid + 1):
                bid_value = (1 - self.phead) * self.values[i - bid] + self.phead * self.values[i + bid]
                if bid_value > max_value:
                    max_value = bid_value
                    best_bid = bid
            self.policy[i] = best_bid
