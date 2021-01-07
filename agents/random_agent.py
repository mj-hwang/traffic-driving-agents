class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def get_action(self, state=None):
        return self.action_space.sample()