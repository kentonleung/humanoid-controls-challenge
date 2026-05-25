class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None

    def update(self, x):
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state.copy()
