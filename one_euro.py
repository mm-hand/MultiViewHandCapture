import numpy as np


class OneEuro:
    def __init__(self, min_cutoff, beta, derivative_cutoff):
        self.min_cutoff, self.beta, self.derivative_cutoff = (
            min_cutoff,
            beta,
            derivative_cutoff,
        )
        self.reset()

    @staticmethod
    def _alpha(dt, cutoff):
        value = 2 * np.pi * cutoff * dt
        return value / (value + 1)

    def __call__(self, value, timestamp):
        value = np.asarray(value, float)
        if self.value is None:
            self.value, self.derivative, self.timestamp = value.copy(), np.zeros_like(value), timestamp
            return value.copy()
        dt = max(timestamp - self.timestamp, 1e-6)
        derivative = (value - self.value) / dt
        alpha = self._alpha(dt, self.derivative_cutoff)
        derivative = alpha * derivative + (1 - alpha) * self.derivative
        alpha = self._alpha(dt, self.min_cutoff + self.beta * np.abs(derivative))
        self.value = alpha * value + (1 - alpha) * self.value
        self.derivative, self.timestamp = derivative, timestamp
        return self.value.copy()

    def reset(self):
        self.value = self.derivative = self.timestamp = None
