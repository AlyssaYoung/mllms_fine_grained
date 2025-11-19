class DefaultConfig:
    """
    Default hyperparameters for visual prompt optimization.
    """
    def __init__(self):
        self.lr = 0.02
        self.T = 5           # total rounds (T-1 update + 1 evaluate)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-3

        # You can include additional fields later:
        # self.alpha = 400
