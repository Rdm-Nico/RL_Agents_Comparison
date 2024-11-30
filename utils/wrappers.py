import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper
from sklearn.preprocessing import MinMaxScaler


class MinMaxObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        scaler = MinMaxScaler()
        model = scaler.fit(obs)
        return model.transform(obs).flatten()



class MinMaxReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, r):
        scaler = MinMaxScaler()
        model = scaler.fit(r)
        return model.transform(r)
