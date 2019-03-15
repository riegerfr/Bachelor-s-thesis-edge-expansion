import numpy as np


class Poisson_Process:
    def __init__(self, min_time, max_time, lambda_poisson):
        self.event_times_positive = []
        self.event_times_negative = [0]
        current_time = 0

        # sample positive times
        while current_time < max_time:
            time = np.random.exponential(1 / lambda_poisson)
            current_time += time
            self.event_times_positive.append(current_time)

        current_time = 0
        # sample negative times
        while current_time > min_time:
            time = np.random.exponential(1 / lambda_poisson)
            current_time -= time
            self.event_times_negative.append(current_time)

    def get_number_events_happened_until_t(self, t):
        assert t != 0
        if t > 0:
            indices = [index for index, time in enumerate(self.event_times_positive) if time < t]
            return len(indices)
        if t < 0:
            indices = [index for index, time in enumerate(self.event_times_positive) if time > t]
            return len(indices)
