from __future__ import annotations
from typing import TYPE_CHECKING

from simpy.events import ProcessGenerator
import simpy
import random

from lib.server import Server, InspectionServer

if TYPE_CHECKING:
    from product import Product


class ManufacturingServer(Server):
    """
    A server that simulate a manufacturing machine.
    """

    def __init__(self, env: simpy.Environment, capacity: int, name: str, p_t: float, verbose: bool = False) -> None:
        """

        :param env:
        :param capacity:
        :param p_t: pre-processing time ( in minute) before start the job
        """
        super().__init__(env, capacity, name)
        # convert pre-processing time from minute to day, where 1440 = 60*24
        self.p_t = p_t / 1440

        self.verbose = verbose

    def process_customer(self, job: Product, service_time: float) -> ProcessGenerator:
        # pre-processing time
        yield self.env.timeout(self.p_t)


        # service time
        yield self.env.timeout(service_time)


        if self.verbose:
            print(
                f'[{self.env.now}] job:{job.product_id} finish the work of machine:{self.name} job_machine_list: {self.current_jobs_p_t}')

        # update the work time
        self.worked_time += (service_time + self.p_t)


class HeatTreatmentServer(Server):
    """
    A server that simulate a heat treatment machine. The work is make outside the Company, and it's done follow a gaussian distribution of mean service_time (days) and standard deviation 0.5
    """

    def process_customer(self, job: Product, service_time: float) -> ProcessGenerator:
        process_time = service_time
        yield self.env.timeout(process_time)

        if self.verbose:
            print(f'[{self.env.now}] job:{job.product_id} finish the work of machine:{self.name}')
        self.worked_time += process_time


class GrindingInspectionServer(InspectionServer):
    """
    A server that simulate an inspection machine after a grinding process.
    In this machine we simulate also the selection phase: namely the phase of process only a cert_amount of pieces
    """

    def __init__(self, env: simpy.Environment, capacity: int, name: str, p_t: float, verbose: bool = False) -> None:
        super().__init__(env, capacity, name)
        # convert pre-processing time from minute to day, where 1440 = 60*24
        self.p_t = p_t / 1440
        self.n_rework = 0
        self.verbose = verbose

    def process_customer(self, job: Product, service_time: float) -> ProcessGenerator:

        # wait for inspection time
        yield self.env.timeout(self.p_t)
        self.worked_time += self.p_t

        # in the reality there isn't a specific Inspection phase, the Inspection is made during the process
        if job.rework and job.pieces_to_repair == 0:
            if self.verbose:
                print(f'[{self.env.now}]  job:{job.product_id} going to rework with service_time:{service_time}')

            self.n_rework += 1

            # yield the error process

            return False
        else:
            # re-init the job.rework
            job.rework = False

        if self.verbose:
            print(f'[{self.env.now}] job:{job.product_id} finish the work of machine:{self.name}')

    def selection_rework(self, job: Product, service_time: float) -> ProcessGenerator:
        """
        We simulate the phase of selection of the wrong pieces
        :param service_time: the service time for the inspection phase
        :param job: the job to review
        :return: a process generator that yield a certain service time
        """
        inspection_time = service_time
        yield self.env.timeout(inspection_time)


        job.pieces_to_repair = int(job.num_of_pieces * random.uniform(0.2, 1))

        if self.verbose:
            print(
                f'[{self.env.now}] job:{job.product_id} going to rework {job.pieces_to_repair} pieces <------------------------------------------')

        self.worked_time += service_time

    def rework(self, job: Product):
        """"
        This method decide where the job should rework the pieces
        """
        match job.product_type:
            case 0:
                # product of type 0
                job.pos = 6
            case 1:
                # product of type 1
                pass
            case 2:
                # product of type 2
                job.pos = 7
            case _:
                print("shouldn't be possible this case!!")
                raise SystemExit(1)
