import localsolver
from localsolver import LSSolutionStatus
import pandas as pd

from src.factory.model_factory import SolverEngine

import logging

logger = logging.getLogger(__name__)


class LSCropEngine(SolverEngine):
    def __build_parameters(self):
        # TODO retrieve this paramaters
        # with adapter pattern
        self.X = list(range(0, 5))
        self.VEGETABLES = ["H1", "H2", "H3"]
        self.TIME = list(range(0, 10))

        self.growth = {"H1": 5, "H2": 3, "H3": 2}
        self.cost_weights = [0.6, 0.4]

    def __set_solver_parameters(self, ls, time_limit, nb_threads=0, seed=42):
        ls.param.time_limit = time_limit
        ls.param.nb_threads = nb_threads
        ls.param.seed = seed

    def execute(self, time_limit=30):
        try:
            with localsolver.LocalSolver() as ls:
                self.model = ls.model
                self.__set_solver_parameters(ls, time_limit=time_limit)
                self.__build_parameters()
                self.__build_model()
                self.__build_objectives()
                self.model.close()
                # self._init_values()
                logger.info("Solving ...")
                ls.solve()
                self.status = ls.solution.status
                if ls.solution.status == LSSolutionStatus.INCONSISTENT:
                    print(ls.compute_inconsistency())
                self._build_solution()
                logger.info("\nPlanning Scheduling Finished!")
        except Exception as err:
            logger.exception(
                f"There was an unexpected error running solver {err}"
            )

    def _build_solution(self):
        df = pd.DataFrame(columns=["x", "h", "t", "value"])
        idx = 0

        for (x, h, t), value in self.var_crop_scheduling.items():
            value = value.value
            idx += 1
            df.loc[idx, :] = [x, h, t, value]
        df.to_csv("solution.csv", index=False)

        df = pd.DataFrame(columns=["x", "h", "t", "value"])
        idx = 0

        for (x, h, t), value in self.exp_crop_bin.items():
            value = value.value
            idx += 1
            df.loc[idx, :] = [x, h, t, value]
        df.to_csv("exp_crop_bin.csv", index=False)

    def __build_model(self):
        self.var_crop_scheduling = {
            (x, h, t): self._binary_variable(name=f"crop_scheduling[{x}, {h}, {t}]")
            if t > 0
            else 0
            for x in self.X
            for h in self.VEGETABLES
            for t in self.TIME
        }

        self.exp_crop_bin = {
            (x, h, t): self._exp_crop_bin(x, h, t)
            for x in self.X
            for h in self.VEGETABLES
            for t in self.TIME
        }

        self.exp_vegetables_production = {
            h: self._exp_vegetables_production(h) for h in self.VEGETABLES
        }

        self.exp_diff_vegetables_production = [
            self._exp_diff_vegetables_production(h1, h2) if h1 != h2 else 0
            for h1 in self.VEGETABLES
            for h2 in self.VEGETABLES
        ]

        for x in self.X:
            for t in self.TIME:
                self._constraint_max_vegetables(x, t)

        for x in self.X:
            for h in self.VEGETABLES:
                for t in self.TIME:
                    self._constraint_min_growth(x, h, t)

    def _init_values(self):
        # Note init_values are values that solver
        # can modify during the optimization 
        
        # At time 0 everything is set to zero
        for x in self.X:
            for h in self.VEGETABLES:
                self.var_crop_scheduling[x, h, 0].value = 0

    def _constraint_max_vegetables(self, x, t):
        return self.model.constraint(
            sum(self.var_crop_scheduling[x, h, t] for h in self.VEGETABLES) <= 1
        )

    def _constraint_min_growth(self, x, h, t):
        post_hours = min(t + self.growth[h] - 1, max(self.TIME))

        post_sum = sum(
            self.var_crop_scheduling[x, h, t_i] for t_i in range(t, post_hours + 1)
        )

        if t + self.growth[h] - 1 > post_hours:
            max_value = (max(self.TIME) - t) + 2
        else:
            max_value = self.growth[h] + 1

        return self.model.constraint(
            self.exp_crop_bin[x, h, t] * (max_value - post_sum) <= 1
        )

    def _exp_crop_diff_bin(self, x, h, t):
        if t == 0:
            return 0

        s1 = self.var_crop_scheduling[x, h, t - 1]
        s2 = self.var_crop_scheduling[x, h, t]

        return (1 - s1 * s2) * self.model.max(s1, s2) * s2

    def _exp_crop_diff_sum_bin(self, x, h, t):
        if t < self.growth[h]:
            return 0

        prev_hours = max(0, t - self.growth[h])

        prev_sum = sum(
            self.var_crop_scheduling[x, h, t_i] for t_i in range(prev_hours, t)
        )

        return self.model.iif(
            prev_sum * self.var_crop_scheduling[x, h, t] == self.growth[h],
            1,
            0,
        )

    def _exp_crop_bin(self, x, h, t):
        return self._exp_crop_diff_bin(x, h, t) + self._exp_crop_diff_sum_bin(x, h, t)

    def _exp_vegetables_production(self, h):
        return (
            sum(self.var_crop_scheduling[x, h, t] for x in self.X for t in self.TIME)
            / self.growth[h]
        )

    def _exp_diff_vegetables_production(self, h1, h2):
        return abs(
            self.exp_vegetables_production[h1] - self.exp_vegetables_production[h2]
        )

    def __build_objectives(self):
        production = sum(self.exp_vegetables_production[h] for h in self.VEGETABLES)

        diff_num_products = -sum(self.exp_diff_vegetables_production)

        cost_total = sum(
            map(lambda a, b: a * b, [production, diff_num_products], self.cost_weights)
        )
        # If you want lexicographic optimization uncomment 
        # lines below
        # self.model.minimize(diff_num_products)
        # self.model.maximize(production)
        self.model.maximize(cost_total)

    def _binary_variable(self, name):
        var = self.model.bool()
        var.name = name
        return var

    def _int_variable(self, name, low_bound=0, up_bound=5000):
        var = self.model.int(low_bound, up_bound)
        var.name = name
        return var
