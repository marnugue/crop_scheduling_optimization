import pyomo.environ as pyo
import logging
import pandas as pd
import os

logging.basicConfig(format="%(message)s", level=logging.INFO)
from pyomo.util.infeasible import log_infeasible_constraints
# from src.factory.model_factory import SolverEngine

# Warning! not done yet dont look this code until is merge
# on master

logger = logging.getLogger(__name__)


class PYCropEngine:
    def _build_parameters(self):
        self.model.VEGETABLES = pyo.Set(initialize=["H1", "H2", "H3"])
        self.model.X = pyo.RangeSet(0, 3)
        self.model.TIME = pyo.RangeSet(0, 30)
        self.model.growth = pyo.Param(
            self.model.VEGETABLES, initialize={"H1": 5, "H2": 3, "H3": 2}
        )

        self.cost_weights = [0.6, 0.4]

        self.BIG_M = 10e6
        self.EPSILON = 10e-6

    def _build_decison_variables(self):
        self.model.var_crop_scheduling = pyo.Var(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            within=pyo.Binary,
            initialize=0,
        )
        # Related to expressions
        self.model.crop_diff_sum_bin = pyo.Var(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            within=pyo.Binary,
            initialize=0,
        )
        self.model.max_s1_s2 = pyo.Var(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            within=pyo.Binary,
            initialize=0,
        )

    def _build_constraints(self):
        self.model.constraint_max_vegetables = pyo.Constraint(
            self.model.X, self.model.TIME, rule=self._constraint_max_vegetables
        )

        self.model.constraint_crop_diff_sum_bin = pyo.ConstraintList(
            rule=self._constraint_crop_diff_sum_bin
        )

        self.model.constraint_max_s1_s2 = pyo.ConstraintList(
            rule=self._constraint_max_s1_s2
        )

        self.model.constraint_min_growth = pyo.Constraint(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            rule=self._constraint_min_growth,
        )

    def _build_expressions(self):
        self.model.exp_vegetable_production = pyo.Expression(
            self.model.VEGETABLES, rule=self._exp_vegetable_production
        )

        self.model.exp_crop_diff_bin = pyo.Expression(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            rule=self._exp_crop_diff_bin,
        )

        self.model.exp_crop_bin = pyo.Expression(
            self.model.X,
            self.model.VEGETABLES,
            self.model.TIME,
            rule=self._exp_crop_bin,
        )

        self.model.exp_diff_vegetables_production = pyo.Expression(
            self.model.VEGETABLES,
            self.model.VEGETABLES,
            rule=self._exp_diff_vegetables_production,
        )

    def execute(self, time_limit: int = 30,
                solution_path: str = "./",
                solver: str = "scip", **sl_parameters):
        self.model = pyo.ConcreteModel()

        self._build_parameters()
        self._build_decison_variables()
        self._build_expressions()
        self._build_constraints()
        self._build_objectives()

        with pyo.SolverFactory(solver) as sl:
            logger.info("Scheduling ...")
            status = sl.solve(self.model, timelimit=time_limit, **sl_parameters)

            if status.solver.status == pyo.SolverStatus.ok:
                logger.info("Scheduler find feasible solution!")
                if (
                    status.solver.termination_condition
                    == pyo.TerminationCondition.optimal
                ):
                    logger.info("Solution is optimal!")
                self._build_solution(solution_path)
            else:
                log_infeasible_constraints(
                    self.model, log_expression=True, log_variables=True
                )

    # Expressions

    def _exp_crop_diff_bin(self, model, x, h, t):
        if t == 0:
            return 0

        s1 = model.var_crop_scheduling[x, h, t - 1]
        s2 = model.var_crop_scheduling[x, h, t]

        max_s1_s2 = self.model.max_s1_s2[x, h, t]

        return (1 - s1 * s2) * max_s1_s2 * s2

    def _exp_crop_bin(self, model, x, h, t):
        return model.exp_crop_diff_bin[x, h, t] + model.crop_diff_sum_bin[x, h, t]

    def _exp_vegetable_production(self, model, h):
        return (
            sum(model.var_crop_scheduling[x, h, t] for x in model.X for t in model.TIME)
            / model.growth[h]
        )

    def _exp_diff_vegetables_production(self, model, h1, h2):
        return abs(model.exp_vegetable_production[h1] - model.exp_vegetable_production[h2])

    # Constraints

    def _constraint_max_vegetables(self, model, x, t):
        return sum(model.var_crop_scheduling[x, h, t] for h in model.VEGETABLES) <= 1

    def _constraint_min_growth(self, model, x, h, t):
        if t == 0:
            return model.var_crop_scheduling[x, h, t] == 0

        post_hours = min(t + model.growth[h] - 1, max(model.TIME))

        post_sum = sum(
            model.var_crop_scheduling[x, h, t_i] for t_i in range(t, post_hours + 1)
        )

        if t + model.growth[h] - 1 > post_hours:
            max_value = (max(model.TIME) - t) + 2
        else:
            max_value = model.growth[h] + 1

        return model.exp_crop_bin[x, h, t] * (max_value - post_sum) <= 1

    def _constraint_crop_diff_sum_bin(self, model):
        for x in model.X:
            for h in model.VEGETABLES:
                for t in model.TIME:
                    if t < model.growth[h]:
                        yield model.crop_diff_sum_bin[x, h, t] == 0

                    else:
                        prev_hours = max(0, t - model.growth[h])

                        prev_sum = sum(
                            model.var_crop_scheduling[x, h, t_i]
                            for t_i in range(prev_hours, t)
                        )

                        left_clause = prev_sum * model.var_crop_scheduling[x, h, t]

                        b = model.crop_diff_sum_bin[x, h, t]

                        yield left_clause >= model.growth[
                            h
                        ] + self.EPSILON - self.BIG_M * (1 - b)

                        yield left_clause <= model.growth[h] + self.BIG_M * b

        # return c1, c2

    def _constraint_max_s1_s2(self, model):
        for x in model.X:
            for h in model.VEGETABLES:
                for t in model.TIME:
                    if t == 0:
                        yield model.max_s1_s2[x, h, t] == 0
                    else:
                        s1 = model.var_crop_scheduling[x, h, t - 1]
                        s2 = model.var_crop_scheduling[x, h, t]

                        max_value = model.max_s1_s2[x, h, t]

                        yield s1 <= max_value
                        yield s2 <= max_value
                        yield max_value <= s1 + s2

                        # return c1, c2, c3

    def _build_objectives(self):
        production = sum(
            self.model.exp_vegetable_production[h] for h in self.model.VEGETABLES
        )

        diff_num_products = -sum(
            self.model.exp_diff_vegetables_production[h1, h2]
            for h1 in self.model.VEGETABLES
            for h2 in self.model.VEGETABLES
        )

        cost_total = sum(
            map(lambda a, b: a * b, [production, diff_num_products], self.cost_weights)
        )

        self.model.production = pyo.Objective(expr=cost_total, sense=pyo.maximize)

    def _build_solution(self, solution_path: str):
        scheduling_solution = []
        
        for x in self.model.X:
            for h in self.model.VEGETABLES:
                for t in self.model.TIME:
                    value = self.model.var_crop_scheduling[x, h, t].value
                    scheduling_solution.append(tuple((f'P{x}', h, t, int(value))))
        
        sol_df = pd.DataFrame(scheduling_solution,
                              columns=['Position', 'Vegetable', 'Time', 'Value']).pivot(
            index='Time', columns=['Position', 'Vegetable'], values='Value'
        )
        
        sol_df.to_csv(os.path.join(solution_path, "solution.csv"))
