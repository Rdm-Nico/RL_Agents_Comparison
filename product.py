from __future__ import annotations
from collections.abc import Sequence, MutableSequence
import numpy as np

import simpy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lib.server import Server


class Product:
    """
    Class representing a batch of pieces  with:
    - done(bool): use for setting the done flag
    - start_time (float): start time of job
    - env(simpy.Environment): environment of job
    - servers(Sequence[Server]): list of servers/machines to use
    - delays(Sequence[float]): list of delays when the job is in queue
    - routing(MutableSequence[tuple[str,float]]): list of routing tuples Server.name and service_time
    - TimeInSystem (MutableSequence[tuple[Server, float]]): save the time information of job
    - rework(bool): rework probability of job
    - product_type(int): use for label the type of product being produced
    - product_id(int): product id
    - num_of_pieces(int): number of pieces to produce
    - pieces_to_repair(int): number of pieces to repair if do_rework is True
    - status(bool): states the status of the product process
    - pos(int): the machine of where to start the work phase
    - verbose(bool): for understand the process of the product more easily
    - D_D(float): Due Dates of the product
    - machines(Sequence[tuple[int:float]]): list of tuple ( idx of the machine to use; processing time)
    """



    def __init__(self, env: simpy.Environment, arrival_time: float, servers: Sequence[Server],
                 rework: bool, num_of_pieces: int, product_type: int, D_D:float, machines:Sequence[tuple[int:float]], verbose: bool = False) -> None:

        self.done: tuple[bool,float] = (False, 0)
        self.start_time: float = arrival_time
        self.env = env
        self.arrival_time = arrival_time
        self.servers = servers
        self.delays: MutableSequence[float] = []
        self.routing: MutableSequence[tuple[int, float]] = []
        self.timeInSystem: MutableSequence[tuple[Server, float]] = []
        self.rework = rework
        self.product_type = product_type
        self.num_of_pieces = num_of_pieces
        self.pieces_to_repair = 0
        self.product_id = 0
        self.verbose = verbose
        self.status = True
        self.pos = 0
        self.D_D = D_D
        self.machines = machines

    @property
    def time_in_system(self) -> float:
        return np.sum(self.delays + self.routing[:,1])

    def work(self, pos:int=0):
        """
        Main function of product timeline
        """
        self.done = (False, 0)
        for idx, processing_time in self.machines[pos:]:

            # identify the server
            server = self.servers[idx]

            # check if this machine is a HeatTreatmentServer, if is jump the calculus of processing time
            if server.name != "Heat Treatment Machine":

                # check if the next phase is a rework or not
                if self.rework and self.pieces_to_repair != 0:
                    processing_time *= self.pieces_to_repair
                else:
                    processing_time *= self.num_of_pieces

            # add the product processing time  in the list of current product in the system
            server.current_jobs_p_t.append(processing_time)

            # store for routing
            self.routing.append((idx, processing_time))

            with server.request() as request:
                # record the time the job joint the queue
                queue_entry_time = self.env.now

                yield request

                queue_exit_time = self.env.now

                self.delays.append(queue_exit_time - queue_entry_time)

                if self.verbose:
                    print(f'[{self.env.now}] job:{self.product_id}  delays:{queue_exit_time - queue_entry_time} next_server: {server.name} process_time: {processing_time} num_of_pieces:{self.num_of_pieces} repair_pieces:{self.pieces_to_repair} rework: {self.rework} job_machine_list: {server.current_jobs_p_t}')


                # Wait for the machine to process the product
                res = yield self.env.process(server.process_customer(self, processing_time))

                # check if return an error
                if res == False:
                    # wait the selection rework
                    yield self.env.process(server.selection_rework(self, processing_time))
                    # store again for this new phase and no waiting queue:
                    self.routing.append((idx, processing_time))
                    self.delays.append(0)

                    # select the point to restart:
                    server.rework(self)


                    self.status = False

                # exiting from a stopped process
                if not self.status:
                    break

            # store the time in system for the product
            self.timeInSystem.append((server, self.env.now - self.start_time))
            # restart the start time
            self.start_time = self.env.now


        # check the status of the product process
        if  self.status:
            # convert to np array for optimization
            self.delays = np.asarray(self.delays)
            self.routing = np.asarray(self.routing)

            if self.verbose:
                print(f'-------------------[{self.env.now}] job:{self.product_id}[product_type:{self.product_type}] finished with time in System:{self.time_in_system}----------------------------')
            self.done = (True, self.env.now)
        else:
            # reset the status
            self.status = True
            # we have to finish the work of the product in the specific:
            self.env.process(self.work(pos=self.pos))
