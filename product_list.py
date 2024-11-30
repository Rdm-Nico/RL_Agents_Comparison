from collections.abc import Callable
import random
from functools import wraps

class ProductList:
    """
    Product list for choose a product
    """
    def decor(func):
        """
        Decorator function for refresh the dict every time the Method getProduct is call
        :return:
        """
        @wraps(func)
        def wrapper(self):
            # call the refresh of the dict here
            self._refreshDict()
            return func(self)
        return wrapper

    def convertToSimpy_sim(self,service_time:float):
        """ convert the service time of IRL in the time of Simpy simulation"""
        return service_time / 86400
    def __init__(self,service_time_distribution: Callable[[int,float], float], due_dates_distribution: Callable[[], float], var:float=0.25):
        """

        :param service_time_distribution: distribution of the processing time
        :param due_dates_distribution: distribution of the Due dates
        :param var: the variance of the service time distribution
        """
        self.var = var
        self._func = service_time_distribution
        self.due_dates_distribution = due_dates_distribution
        self._product_dict: dict[int, tuple[int, list[tuple[int, float]]]] = {}
        self._weights:list[float]=[0.22, 0.42, 0.36]

        # create the dict
        self._refreshDict()

        # calculate the max route
        self.max_route =self._compute_max_route()


    def _compute_max_route(self)-> int:
        """Method to compute the max route of the product list"""
        max_route = 0
        for machine in self._product_dict.values():
            _, list_p = machine

            if len(list_p) > max_route:
                max_route = len(list_p)
        return max_route


    def _refreshDict(self):
        self._product_dict = {0: (1000, [
            (0, self.convertToSimpy_sim(self._func(90,self.var))),
            (1, self.convertToSimpy_sim(self._func(29,self.var))),
            (2, self.convertToSimpy_sim(self._func(52,self.var))),
            (3, self.convertToSimpy_sim(self._func(5,self.var))),
            (4, 2),  # because is the heat treatment is measure in days
            (3, self.convertToSimpy_sim(self._func(5,self.var))),
            (5, self.convertToSimpy_sim(self._func(56,self.var))),
            (6, self.convertToSimpy_sim(self._func(30,self.var))),
            (5, self.convertToSimpy_sim(self._func(50,self.var))),
            (7, self.convertToSimpy_sim(self._func(47,self.var))),
            (8, self.convertToSimpy_sim(self._func(6,self.var))),
            (9, self.convertToSimpy_sim(self._func(10,self.var))),
        ]), 1: (1700, [
            (0, self.convertToSimpy_sim(self._func(45,self.var))),
            (4, 5),
            (3, self.convertToSimpy_sim(self._func(8,self.var))),
            (8, self.convertToSimpy_sim(self._func(2,self.var))),
            (9, self.convertToSimpy_sim(self._func(10,self.var))),
        ]), 2: (386,[
            (0, self.convertToSimpy_sim(self._func(75, self.var))),
            (1, self.convertToSimpy_sim(self._func(109, self.var))),
            (10, self.convertToSimpy_sim(self._func(116, self.var))),
            (3, self.convertToSimpy_sim(self._func(10, self.var))),
            (4, 7),  # because is the heat treatment is measure in days
            (3, self.convertToSimpy_sim(self._func(10, self.var))),
            (11, self.convertToSimpy_sim(self._func(35, self.var))),
            (5, self.convertToSimpy_sim(self._func(63, self.var))),
            (6, self.convertToSimpy_sim(self._func(30, self.var))),
            (8, self.convertToSimpy_sim(self._func(18, self.var))),
            (9, self.convertToSimpy_sim(self._func(10, self.var))),
        ])}

    @decor
    def getProduct(self):
        """
        Random pic a product
        :return: num of pieces to make and the machine schedule
        """
        product_id =random.choices(list(self._product_dict),weights=self._weights)[0]
        num_of_pieces = self._product_dict[product_id][0]
        D_D =  self.due_dates_distribution()

        return product_id, num_of_pieces, self._product_dict[product_id][1],D_D

    @property
    def getNumOf_Type(self) -> int:
        return len(self._product_dict)

    @property
    def getMax_routing(self) -> int:
        return self.max_route