from __future__ import annotations


import statistics
from collections.abc import Callable, Iterable
from collections import namedtuple
from typing import Any, Tuple, Dict, List

import simpy
from gymnasium.core import ObsType, ActType
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from simpy.events import ProcessGenerator, Event
import matplotlib.pyplot as plt
from project.machines import ManufacturingServer, HeatTreatmentServer, GrindingInspectionServer
from lib.server import Server
from project.product import Product
from project.product_list import ProductList
import random
from operator import attrgetter
from bisect import insort

from gymnasium.spaces import Discrete, MultiDiscrete
import gymnasium as gym
import numpy as np



class ManufacturingSystem(gym.Env):
    """
        Gym Environment of a Manufacturing System simulate with SimPy.


    Attributes
    ----------




    Internal methods
    ----------------

    __init__:
        Constructor method
    _calculate_reward:
        Calculate reward
    _get_observations:
        Get current state observations
    _is_truncated:
        Evaluate if an episode need to be truncated
    _reset_machine_list:
        Reset the trackers of each machine



    External facing methods
    -----------------------

    close

    reset

    step

    """

    def __init__(
            self,
            inter_arrival_time_distribution: Callable[[], float],
            service_time_distribution: Callable[[int, float], float],
            rework_distribution: Callable[[], bool],
            due_dates_distribution: Callable[[], float],
            warmup_period: int,
            max_wip: float,
            sim_duration: int = 5000,
            random_seed: int = 42,
            eval_days: int = 180,
            wip_tol: float = 0.2,
            days_lookback: int = 10,
            n_trackers: int = 33,
            d_d_bound: tuple[int] = (-2, 2),
            verbose: bool = False,
            sampling: bool = False,
            sim_time_step: float = 1.,
            push_model: bool = False,
    ) -> None:

        # init the of the env
        super(ManufacturingSystem, self).__init__()

        # set attributes

        self.sim_time_step = sim_time_step
        self.n_trackers = n_trackers
        self.max_wip = max_wip
        self.eval_days = eval_days
        self.wip_tol = wip_tol
        self.d_d_bound = d_d_bound
        self.warmup_period = warmup_period
        self.random_seed = random_seed
        self.sim_duration = sim_duration
        self.days_lookback = days_lookback
        self.observation_size = days_lookback * n_trackers
        self.inter_arrival_time_distribution = inter_arrival_time_distribution
        self.service_time_distribution = service_time_distribution
        self.rework_distribution = rework_distribution
        self.due_dates_distribution = due_dates_distribution
        self.prev_act = True
        self.jobs_inter_arrival_times: list[float] = []
        self.verbose = verbose
        self.PSP = namedtuple('PSP', ('DD', 'Product'))
        self.by_dd = attrgetter('DD')
        self.psp_list: list[tuple[float, Product]] = []
        self.first_obs = False
        self.push_model = push_model
        self.interrupt_SLACK = False
        self.agent_interrogation = 0
        self.agent_release = 0
        self.agent_not_decide = 0

        self.jobs = []
        self.sampling = sampling
        self.mts_stats: list[list[float]] = [[0,0,0]]
        self.wip_tot_stats: list[float] = []
        self.wip_stats: list[list[float]] = []
        self.ea_ta_ti_stats: list[list[int]] = [[0,0,0]]

        # actions space: we have 2 actions, corresponding to "release" or "not release"
        self.action_space = Discrete(2)
        self.action_size = 2
        """
        observation space: observations is a sequence of vector corresponding to the state of the system from the current day to 10 days ago
        every vector as:
                  - n_j(int) = number of job  to do
                  - WIP_i(float) = WIP of the i-th machine
                  - PT_i(float) = Processing time of the i-th machine
                  - delta(float) = Delta time of the current urgent job in the PSP
                  - a_t-1(bool) = the last action choose in the previous step
                  - EA(int) = Earliness of the previous state
                  - TA(int) = Tiredness of the previous state
                  - MT_i (float) = Monthly mean time in system for that particular type of product 
        """

        self.observation_space = MultiDiscrete(
            np.ones((self.days_lookback, n_trackers)))  # matrix with shape (10,33)

        # create a mat of the last self.days_lookback of the system
        self.obs_mat = np.zeros(((self.days_lookback, n_trackers)), dtype=np.float64)

        # control if we are in PUSH model:
        if self.push_model:
            self.push_info = self.push_env()


    def push_env(self):
        """
        Run the PUSH Env
        """
        self.simpy_env = simpy.Environment()

        # reset list and trackers

        self.psp_list = []
        self.jobs = []
        self.wip_stats = []
        self.wip_tot_stats = []
        self.mts_stats = [[0,0,0]]
        self.jobs_inter_arrival_times = []
        self.ea_ta_ti_stats: list[list[int]] = [[0,0,0]]

        # set machine list and products

        self.machines_list: list[Server] = self.m_2_list()
        self.products = ProductList(self.service_time_distribution, self.due_dates_distribution)

        # start continues processes

        self.simpy_env.process(self.wip_in_sys())
        self.simpy_env.process(self.EA_TA_TI_counter())
        self.simpy_env.process(self.mean_time_in_sys())

        self.simpy_env.process(self.run_PUSH())

        # start the simpy Env
        self.simpy_env.run(until=self.sim_duration)

        # return info
        info = {
            'wip in system': self.wip_tot_stats,
            'number of job dones': self.finished_jobs,
            'EA_TA_TI': self.ea_ta_ti_stats,
            'number of job create': len(self.jobs),
            'Average Product Inter-Arrival time': statistics.mean(self.jobs_inter_arrival_times),
            'Average time in System': statistics.mean(self.product_time_in_system)
        }
        return info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> ndarray[Any, dtype[floating[_64Bit] | float_]] | ndarray[Any, dtype[Any]]:
        """
               Resets the environment to an initial state:
               1) Create new SimPy Env
               2) Start SimPy processes(generate product)
               3) pick a random action to the first `warm up` days -> ( Better PUSH than random action)
               4) pass back the first observations
               :param seed:
               :param options:
               :return: first agent observation and information
        """
        # set the seed
        super().reset(seed=seed)

        self.simpy_env = simpy.Environment()

        # reset list and trackers
        self.psp_list = []
        self.agent_interrogation = 0
        self.agent_release = 0
        self.agent_not_decide = 0
        self.jobs = []
        self.wip_stats = []
        self.wip_tot_stats = []
        self.mts_stats = [[0,0,0]]
        self.jobs_inter_arrival_times = []
        self.ea_ta_ti_stats: list[list[int]] = [[0,0,0]]
        self.interrupt_SLACK = False

        # set machine list and products
        self.machines_list: list[Server] = self.m_2_list()
        self.products = ProductList(self.service_time_distribution, self.due_dates_distribution)

        # start continues processes
        self.simpy_env.process(self._generate_order())
        self.simpy_env.process(self.wip_in_sys())
        self.simpy_env.process(self.EA_TA_TI_counter())
        self.simpy_env.process(self.mean_time_in_sys())

        # make the first days in PUSH approach
        slack = self.simpy_env.process(self.run_SLACK())

        # start the simpy Env
        self.simpy_env.run(until=self.warmup_period)

        # interrupt the process
        slack.interrupt()

        # return the first observations
        self._get_observations()


        return self.obs_mat.flatten()

    def step(
            self, action: ActType
    ) -> tuple[ndarray[Any, dtype[floating[_64Bit] | float_]], int, bool, bool, dict[
        str, list[list[int]] | int | list[float]]]:
        """
                Action step.
                The RL Agent has chosen what action to perform ( work or not ). This  should happen 1 every day after see the Observations of that day.
                1) First `warmup_period` days we use the PUSH method for decide the product
                2) After the RL agent choose what to do

                :param action: the action to do
                :return: next agent observation,  reward, env terminated or truncated , info
        """

        # track of agent interrogation
        self.agent_interrogation += 1



        # the RL decide what to do, if there isn't product in the psp the agent doesnt do anything
        if action and len(self.psp_list) != 0:

            # update the agent releases
            self.agent_release += 1
            # take the most urgent job
            d_d, urgent_job = self.psp_list[0]

            if self.verbose:
                print(
                    f'[{self.simpy_env.now}] the job with id:{urgent_job.product_id} with product_type: {urgent_job.product_type} and D.D.:{urgent_job.D_D} is going to start')

            # pop out the job from the PSP
            self.psp_list.pop(0)

            # start the production pipeline
            self.simpy_env.process(urgent_job.work())

        elif  len(self.psp_list) == 0:
            # for secure we correct the action if we don't do anything
            self.agent_not_decide += 1
            action = False

        # save the current action
        self.prev_act = action

        # run the simulation
        self.simpy_env.run(until=self.simpy_env.now + self.sim_time_step)

        # get the observations
        self._get_observations()

        # get the return values
        reward = self._calculate_reward()

        terminal = True if self.simpy_env.now >= self.sim_duration else False

        # find if this episode needs to be truncated
        truncated = self._is_truncated()

        if truncated:
            # very negative reward to wish to never come back here
            reward = self.simpy_env.now - self.sim_duration


        # pass the info
        info = {
            'wip in system': self.wip_tot_stats,
            'number of job dones': self.finished_jobs,
            'EA_TA_TI': self.ea_ta_ti_stats,
            'number of job create': len(self.jobs),
            'Average Product Inter-Arrival time': statistics.mean(self.jobs_inter_arrival_times),
            'Agent releases': (self.agent_release / self.agent_interrogation) * 100,
            'Agent not decide': (self.agent_not_decide / self.agent_interrogation) * 100,
            'psp_list': len(self.psp_list)
        }

        return self.obs_mat.flatten(), reward, terminal, truncated, info

    def close(self):
        """
                Clean up any necessary simulation objects
        """
        del self.simpy_env
        return 0

    def _is_truncated(self) -> bool:
        """
        Investigate if the current episode needs to be truncated. This happens if the WIP of the  entire System is higher for a long time ( 1000 time step for example).
        If this happens it's necessary to give  to the agent a very negative reward
        :return:
        """

        if self.eval_days > len(self.wip_tot_stats):
            return False

        # get last days WIP
        WIP_sys = self.wip_tot_stats[-self.eval_days:]

        # compute the RMSE
        var = np.sqrt((np.square(np.mean(WIP_sys) - self.max_wip)).mean())


        if var <= self.wip_tol:
            return True

        return False

    def run_PUSH(self):
        """RUN the Env with PUSH approach """
        while True:
            # job arrive
            inter_arrival_time = self.inter_arrival_time_distribution()
            self.jobs_inter_arrival_times.append(inter_arrival_time)

            yield self.simpy_env.timeout(inter_arrival_time)
            # we random sample the type of  product we've to manufacture:
            product_type, num_of_pieces, machines, D_D = self.products.getProduct()

            job = Product(
                env=self.simpy_env,
                arrival_time=self.simpy_env.now,
                servers=self.machines_list,
                rework=self.rework_distribution(),
                num_of_pieces=num_of_pieces,
                product_type=product_type,
                verbose=self.verbose,
                D_D=D_D + self.simpy_env.now,
                machines=machines
            )

            self.jobs.append(job)
            job.product_id = self.jobs.index(job)

            if self.verbose:
                print(
                    f'[{self.simpy_env.now}] the job with id:{job.product_id} with product_type: {job.product_type} is going to start')

            self.simpy_env.process(job.work())

    def _calculate_reward(self) -> int:
        """
        Calculate the reward of the current time step in the System.
        1) Should be the sum of the delta of EA, TA and TI of this day respect to the previous and the length of the PSP 
        :return: the reward score
        """




        today = self.ea_ta_ti_stats[-1]
        prev = self.ea_ta_ti_stats[-2]

        delta_ea = prev[0] - today[0]
        delta_ta = prev[1] - today[1]
        delta_ti = prev[2] - today[2]
        reward = delta_ea + delta_ta - delta_ti - len(self.psp_list)

        return reward

    def _get_observations(self):
        """
        Get the current observations of the system, we take:
        - Normalized WIP of every machine
        - Normalized processing times of the product
        - the delta of the current most urgent product in the PSP
        - last choice of the state
        - Compute EA and TA
        - number of products in the PSP
        - Monthly mean time in System grouped for type of product
        - Type of product
        """

        # delete the last row of the mat
        self.obs_mat = np.delete(self.obs_mat, -1, axis=0)



        # check if the psp is empty
        if len(self.psp_list) == 0:
            # create a vector of zeros
            self.obs_mat = np.vstack((np.zeros((self.n_trackers,)), self.obs_mat))
            return

        # get the last day of WIP
        obs = self.wip_stats[-1]

        # convert to np
        obs = np.array(obs)

        # normalize the vector
        obs = (obs - np.mean(obs)) / (np.std(obs) + 1e-6)

        # extract the urges product in the PSP
        d_d, urgent_job = self.psp_list[0]

        # compute the lead time for the specific product
        processing_times = self.compute_processing_times(urgent_job)

        obs = np.append(obs, processing_times)

        # delta of the product PSP
        delta = d_d - self.simpy_env.now


        obs = np.append(obs, delta)


        # last action time step
        obs = np.append(obs, self.prev_act)


        # number of job in the PSP
        obs = np.append(obs, len(self.psp_list))

        # EA and TA
        ea_ta = self.ea_ta_ti_stats[-1]

        obs = np.append(obs, [ea_ta[0], ea_ta[1]])

        # get the monthly mean time
        last_mts = self.mts_stats[-1]

        obs = np.append(obs, last_mts)


        # add the product type
        obs = np.append(obs, urgent_job.product_type)


        # add the new row in front of the mat
        self.obs_mat = np.vstack((obs, self.obs_mat))

    def compute_processing_times(self, urgent_job) -> list[float]:
        """
        Compute the processing time of every machine , this depends on the product type.

        Is a list of length eq at the max routing product
        :param urgent_job: urgent job in the PSP
        :return: list of processing times
        """

        processing_times = np.zeros((self.products.getMax_routing,), dtype=np.float16)

        match urgent_job.product_type:
            case 0:
                # product of type 0


                for i, machine  in enumerate(urgent_job.machines):

                    #unpack
                    idx,processing_time = machine
                    # get the server
                    server = urgent_job.servers[idx]

                    match server.name:
                        case "Grinding Inspection Machine":
                            """
                            here we take only the starting time of the machine
                            """
                            processing_time = server.p_t
                            processing_times[i] += processing_time

                        case "Heat Treatment Machine":
                            processing_times[i] = processing_time

                        case _:
                            # all others
                            processing_time *= urgent_job.num_of_pieces
                            processing_times[i] = processing_time

            case 1:
                # product of type 1
                for i, machine in enumerate(urgent_job.machines):

                    # unpack
                    idx, processing_time = machine

                    server = urgent_job.servers[idx]
                    # check if this machine is a HeatTreatmentServer, if is jump the calculus of processing time
                    if server.name != "Heat Treatment Machine":
                        # compute the processing time
                        processing_time *= urgent_job.num_of_pieces

                    processing_times[i] = processing_time

            case 2:
                # product of type 2
                for i, machine in enumerate(urgent_job.machines):

                    # unpack
                    idx, processing_time = machine
                    # get the server
                    server = urgent_job.servers[idx]

                    match server.name:
                        case "Grinding Inspection Machine":
                            """
                            here we take only the starting time of the machine
                            """
                            processing_time = server.p_t
                            processing_times[i] += processing_time

                        case "Heat Treatment Machine":
                            processing_times[i] = processing_time

                        case _:
                            # all others
                            processing_time *= urgent_job.num_of_pieces
                            processing_times[i] = processing_time

            case _:
                print("shouldn't be possible this case!!")
                raise SystemExit(1)

        # normalize the vector
        norma_processing_times = (processing_times - np.mean(processing_times)) / np.std(processing_times)

        return norma_processing_times.tolist()


    def EA_TA_TI_counter(self):
        """
        Measures the Earliness, Tiredness and Timing  of the product done
        """
        while True:
            yield self.simpy_env.timeout(self.sim_time_step)
            # we return the works that have been finished in the last days
            jobs_done = (job for job in self.jobs if
                         job.done[0] and job.done[1] >= self.simpy_env.now - self.sim_time_step)


            # calculate EA, TA and TI
            EA = self.ea_ta_ti_stats[-1][0]
            TA = self.ea_ta_ti_stats[-1][1]
            TI = self.ea_ta_ti_stats[-1][2]


            for job in jobs_done:
                delta = job.D_D - job.done[1]
                # check if it's inside the bound
                if delta > self.d_d_bound[1]:
                    EA += 1
                elif delta < self.d_d_bound[0]:
                    TA += 1
                else:
                    # product in time
                    TI += 1

            self.ea_ta_ti_stats.append([EA, TA, TI])

    def mean_time_in_sys(self):
        """
        Measures the monthly mean time in system
        """
        while True:
            yield self.simpy_env.timeout(30)

            means_type: list[float] = []

            for type in range(self.products.getNumOf_Type):
                job_dones = [job.time_in_system for job in self.jobs if job.done[0] and job.product_type == type]

                # if we are in the beginning and there isn't finished jobs of one type, we return 0
                if len(job_dones) == 0:
                    mean_mts = 0
                else:
                    mean_mts = np.mean(job_dones)

                # save the current mean
                means_type.append(mean_mts)

            self.mts_stats.append(means_type)

    def wip_in_sys(self):
        """
             Measures the daily wip per machine in system
             The WIP is measure from the sum of processing time of every product that's in the queue of the machine
        """
        while True:
            yield self.simpy_env.timeout(1)
            wip = [sum(machine.current_jobs_p_t) for machine in self.machines_list]
            self.wip_stats.append(wip)

            # update also the wip of the entire sys
            self.wip_tot_stats.append(sum(wip))

    def _generate_order(self):
        """
        Generate order and save it in the PSP ordered list
        """
        while True:
            # job arrive
            inter_arrival_time = self.inter_arrival_time_distribution()
            self.jobs_inter_arrival_times.append(inter_arrival_time)

            yield self.simpy_env.timeout(inter_arrival_time)
            # we random sample the type of  product we've to manufacture:
            product_type, num_of_pieces, machines, D_D = self.products.getProduct()

            job = Product(
                env=self.simpy_env,
                arrival_time=self.simpy_env.now,
                servers=self.machines_list,
                rework=self.rework_distribution(),
                num_of_pieces=num_of_pieces,
                product_type=product_type,
                verbose=self.verbose,
                D_D=D_D + self.simpy_env.now,
                machines=machines
            )

            self.jobs.append(job)
            job.product_id = self.jobs.index(job)

            # insert into the PSP list order by urges job
            product = self.PSP(job.D_D, job)
            insort(self.psp_list, product, key=self.by_dd)

    @property
    def finished_jobs(self) -> int:
        return sum(job.done[0] for job in self.jobs)

    @property
    def ut_rate(self) -> list[float]:
        """
        :return: list of utilization rate for each machine.
        """
        return [machine.utilization_rate for machine in self.machines_list]

    def run_SLACK(self) -> ProcessGenerator:
        """
        Run the system with SLACK method ( take the most urgent job and release it)
        """
        try:
            while True:
                # take the most urgent
                if len(self.psp_list) == 0:
                    # wait the product
                    yield self.simpy_env.timeout(self.sim_time_step)
                    continue

                d_d, urgent_job = self.psp_list[0]

                delta = d_d - self.simpy_env.now
                if self.verbose:
                    print(
                        f'[{self.simpy_env.now}] the job with id:{urgent_job.product_id} with product_type: {urgent_job.product_type} and D.D.:{urgent_job.D_D} is going to start')

                # pop out the job from the PSP
                self.psp_list.pop(0)

                # start the production pipeline
                self.simpy_env.process(urgent_job.work())
        except simpy.Interrupt:
            # we interrupt the process
            self.interrupt_SLACK = True

    @property
    def product_avg_delay_times(self) -> Iterable[float]:
        """
        Calculate the average delay time for each product in the system
        """
        for job in self.jobs:
            if job.done[0]:
                yield statistics.mean(job.delays)

    @property
    def product_time_in_system(self) -> Iterable[float]:
        for job in self.jobs:
            if job.done[0]:
                yield job.time_in_system


    def plot_average_inter_arrival_time(self):
        """
        plot the average inter arrival time distribution
        """
        average_product_inter_arrival_time = statistics.mean(self.jobs_inter_arrival_times)
        print(f"Average Product Inter-arrival time = {average_product_inter_arrival_time:.2f} days")
        plt.hist(self.jobs_inter_arrival_times, bins=50)
        plt.title('Inter-arrival time distribution')
        plt.show()

    def plot_product_average_delay_time(self):
        """
        plot the average delay time for each product in the system
        """
        average_product_delay_time = statistics.mean(self.product_avg_delay_times)
        print(f"Average Product delay time = {average_product_delay_time:.2f} days")
        plt.hist(list(self.product_avg_delay_times), bins=10)
        plt.title('Product delay distribution')
        plt.show()

    def plot_products_time_in_system(self):
        average_product_in_system = statistics.mean(self.product_time_in_system)
        print(f"Average Product Time in System  = {average_product_in_system:.2f} days")
        plt.hist(list(self.product_time_in_system), bins=10)
        plt.title('Product Time in System distribution')
        plt.show()


    def plot_last_day_eval(self):
        """
        Plot the last day pie charts of EA,TA, TI
        """


        sizes = np.array(self.ea_ta_ti_stats[-1]) / self.finished_jobs
        labels = ['EA','TA','TI']
        plt.pie(sizes,labels=labels, autopct='%1.1f%%')
        plt.title('Partition of jobs done in the system')
        plt.show()


    def plot_ut_rate_in_system(self):

        # Generate 10 random colors
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(10)]

        ids = [i for i, _ in enumerate(self.machines_list)]
        # Ensure the number of colors matches the number of bars
        bar_colors = colors[:len(ids)] if len(ids) <= 10 else colors * (len(ids) // 10 + 1)
        bar_colors = bar_colors[:len(ids)]
        names = [f'{i}:{machine.name}' for i, machine in enumerate(self.machines_list)]
        plt.bar(ids, self.ut_rate, label=names, color=bar_colors)
        plt.title('Utilization Rate Distribution')
        plt.xlabel('Machine ID')
        plt.ylabel('Utilization Rate')

        # Add the ID numbers on top of each bar
        for i, v in enumerate(self.ut_rate):
            plt.text(i, v + 0.01, str(ids[i]), ha='center', va='bottom')

        # hide x values
        plt.xticks([])

        plt.legend(title='List of machines', fontsize="xx-small")
        plt.show()

    def plot_wip_in_system(self):
        """
        Plot the WIP for every machine over time and the WIP of the entire system
        """
        dict_machines: dict[str, list[float]] = {}
        names = [f'{machine.name}' for i, machine in enumerate(self.machines_list)]
        for name in names:
            dict_machines[name] = []

        plt.figure(figsize=(12, 6))

        # store in a dict the stats
        for wip_day in self.wip_stats:
            for name, wip_machine in zip(names, wip_day):
                dict_machines[name].insert(len(dict_machines[name]), wip_machine)

        # add the WIP of all system 
        name = "Total System WIP"
        dict_machines[name] = []

        for wip_day in self.wip_tot_stats:
            dict_machines[name].insert(len(dict_machines[name]), wip_day)

        # create the plot
        for d in dict_machines:
            plt.stairs(dict_machines[d], label=d)

        plt.title("WIP distribution:")
        plt.xlabel("Time")
        plt.ylabel("WIP")
        plt.legend()

        plt.show()

    def m_2_list(self) -> list[Server]:
        """
        Function to store the machines in the list
        :return list of machines
        """
        _machines_list = []
        # 0
        turning = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=30,
            name="Turning Machine",
            verbose=self.verbose
        )
        _machines_list.append(turning)

        # 1
        dentition = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=25,
            name="Dentition Machine",
            verbose=self.verbose
        )
        _machines_list.append(dentition)

        # 2
        shaving = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=30,
            name="Shaving Machine",
            verbose=self.verbose
        )
        _machines_list.append(shaving)

        # 3
        composition = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=5,
            name="Composition Machine",
            verbose=self.verbose
        )
        _machines_list.append(composition)
        # 4
        h_treatment = HeatTreatmentServer(
            env=self.simpy_env,
            capacity=1,
            name="Heat Treatment Machine",
            verbose=self.verbose
        )
        _machines_list.append(h_treatment)

        # 5
        grinding = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=22,
            name="Grinding Machine",
            verbose=self.verbose
        )
        _machines_list.append(grinding)

        # 6
        grinding_inspection = GrindingInspectionServer(
            env=self.simpy_env,
            capacity=1,
            p_t=5,
            name="Grinding Inspection Machine",
            verbose=self.verbose
        )
        _machines_list.append(grinding_inspection)

        # 7
        polishing = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=20,
            name="Polishing Machine",
            verbose=self.verbose
        )
        _machines_list.append(polishing)

        # 8
        washing = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=7,
            name="Washing Machine",
            verbose=self.verbose
        )
        _machines_list.append(washing)
        # 9
        packing = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=10,
            name="Packing Machine",
            verbose=self.verbose
        )
        _machines_list.append(packing)
        # 10
        grove = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=75,
            name="Grove Machine",
            verbose=self.verbose
        )
        _machines_list.append(grove)
        # 11
        straightening = ManufacturingServer(
            env=self.simpy_env,
            capacity=1,
            p_t=12,
            name="Straightening Machine",
            verbose=self.verbose
        )
        _machines_list.append(straightening)

        return _machines_list
