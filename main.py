from __future__ import annotations
from utils.seeds import Seeds
from project.env_system import ManufacturingSystem
import random
from scipy.stats import erlang
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from gymnasium.wrappers import NormalizeReward
import numpy as np
from utils.Analyze import compare_with_std

from pyinstrument import Profiler
import simpy


if __name__ == '__main__':

    sd = Seeds()
    random.seed(42)

    dict_z = dict()
    dict_z['DDQN_4_step new'] = [{'wip': np.random.randn(1,10)}]
    dict_z['DDQN_4_step old'] = [{'wip': np.random.randn(1,10)}]
    dict_z['TD-based A2C'] = [{'wip': np.random.randn(1,10)}]
    dict_z['MC-based A2C'] = [{'wip': np.random.randn(1,10)}]
    push_model = [{'wip': np.random.randn(1,10)}]

    compare_with_std(dict_z, push_model, param="wip")








