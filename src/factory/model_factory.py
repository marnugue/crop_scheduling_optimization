from abc import ABC, abstractmethod


class SolverEngine(ABC):
    @abstractmethod
    def __build_parameters(self):
        pass

    @abstractmethod
    def __set_solver_parameters(self):
        pass

    @abstractmethod
    def execute(self, time_limit: int):
        pass

    @abstractmethod
    def __build_model(self):
        pass

    @abstractmethod
    def __build_objectives(self):
        pass

    @abstractmethod
    def _build_solution(self):
        pass
