import gym
import highway_env
from agents.highway_dqn_agent import DQNAgent

env = gym.make("highway-v0")


# env = gym.make("BreakoutNoFrameskip-v4")
# env = NoopResetEnv(env, noop_max=30)
# env = MaxAndSkipEnv(env, skip=4)
# env = EpisodicLifeEnv(env)
# env = FireResetEnv(env)
# env = WarpFrame(env)
# env = ScaledFloatFrame(env)
# env = ClipRewardEnv(env)
# env = FrameStack(env, 4)


agent = DQNAgent(env)
agent.train()
agent.save_model(filename="my_model.h5")