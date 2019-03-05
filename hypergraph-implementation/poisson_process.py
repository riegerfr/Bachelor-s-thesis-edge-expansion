import numpy as np


class Poisson_Process:
    def __init__(self, min_time, max_time, lambda_poisson):
        # create poisson process todo: refactor into seperate class
        self.event_times_positive = []  # a event happens at time= 0 per definition todo: correct?
        self.event_times_negative = [0]
        current_time = 0

        # sample positive times
        while current_time < max_time:
            time = np.random.exponential(1 / lambda_poisson)  # todo: maybe 1/lambda?
            current_time += time
            self.event_times_positive.append(current_time)

        current_time = 0
        # sample negative times
        while current_time > min_time:
            time = np.random.exponential(1 / lambda_poisson)  # todo: maybe 1/lambda?
            current_time -= time
            self.event_times_negative.append(current_time)

    def get_number_events_happened_until_t(self, t):
        assert t != 0  # todo: neccessary?
        if t > 0:
            indices = [index for index, time in enumerate(self.event_times_positive) if time < t]
            return len(indices)
        if t < 0:
            indices = [index for index, time in enumerate(self.event_times_positive) if time > t]
            return len(indices)