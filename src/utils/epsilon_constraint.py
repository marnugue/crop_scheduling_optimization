from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
from pyomo.environ import (
    Model,
    Objective,
    SolverFactory,
    value,
    Constraint,
    Param,
    Var,
    Set,
    NonNegativeReals,
    maximize,
    SolverStatus,
    ConcreteModel,
    Expression,
)


@dataclass
class Solution:
    _decision_vars: dict = field(default_factory=dict)
    _obj_values: dict = field(default_factory=dict)

    @property
    def decision_vars(self):
        return self._decision_vars

    @property
    def obj_values(self):
        return self._obj_values

    def add_decision_var_value(self, decision_name: str, value):
        self._decision_vars[decision_name] = value

    def add_objective_value(self, objective_name: str, value):
        self._obj_values[objective_name] = value


@dataclass
class EpsilonConstraintAugment:
    model: Model = field(default_factory=ConcreteModel)
    num_solutions: int = 5
    solver: str = "cbc"

    @staticmethod
    def __get_interval_points(
        lowerbound: float, upperbound: float, num_points: int = 4
    ) -> List[float]:
        """Calculate `num_points + 1` equidistand points inside an interval.

        By default return 5 points.

        Args:
            lowerbound (float): _description_
            upperbound (float): _description_
            num_points (int, optional):  num of equidistant points - 1.
            Defaults to 4.

        Raises:
            Exception: _description_

        Returns:
            List[float]: Equidistant points
        """

        if lowerbound > upperbound:
            raise Exception("lowerbound is higher than upperbound parameter")

        res = []

        delta = abs(upperbound - lowerbound) / num_points

        for i in range(num_points + 1):
            res.append(lowerbound + delta * i)
        res.append(upperbound)

        return res

    @staticmethod
    def __solve_instance(model: Model, timelimit=10, solver=solver):
        solver = SolverFactory('mindtpy')
        # solver.options['seconds'] = timelimit
        status = solver.solve(model, mip_solver='cbc',
                              nlp_solver='ipopt', tee=True,
                              strategy='ECP',
                              time_limit=timelimit,
                              iteration_limit=10)
        return status

    @staticmethod
    def __epsilon_objective(model: Model, objective_name: str):
        F = getattr(model, objective_name)

        model.epsilon = Param(initialize=10e-6, mutable=False)

        for s in model.S:
            F += model.epsilon * model.s[s]

        return F

    @staticmethod
    def __epsilon_constraints(model: Model, payoff_table: dict):
        for s in model.S:
            # s have s_obj_name
            f_name = s[2:]
            f = getattr(model, f_name)
            c = Constraint(expr=f - model.s[s] == payoff_table[f_name])
            setattr(model, f"c_{s}", c)

    @staticmethod
    def __delete_epsilon_constraints(model: Model):
        for s in model.S:
            model.del_component(getattr(model, f"c_{s}"))

    def __iterate_payoff_table(self, payoff_table: dict, **kwargs):
        if not payoff_table:
            raise Exception("function called with payoff table empty")

        efficient_solution_list = []
        obj_name = list(payoff_table.keys())[0]
        grid_point_list = payoff_table.pop(obj_name)

        for grid_point in grid_point_list:
            if not payoff_table:
                efficient_solution = self.__solve_epsilon_augmenton(
                    {**{obj_name: grid_point}, **kwargs}
                )
                if efficient_solution:
                    efficient_solution_list.append(efficient_solution)
            else:
                efficient_solution = self.__iterate_payoff_table(
                    self.model, payoff_table, obj_name=grid_point
                )
                efficient_solution_list.extend(efficient_solution)

        return efficient_solution_list

    @staticmethod
    def __get_solutions(model: Model):
        solution = Solution()
        s_var_list = [model.s[ss].name for ss in model.S]
        for v in model.component_data_objects(Var, active=True):
            if v.name not in s_var_list:
                solution.add_decision_var_value(v.name, value(v))
        for v in model.component_data_objects(Expression, active=True):
            if v.name not in s_var_list:
                solution.add_decision_var_value(v.name, value(v))

        for v in model.component_data_objects(Objective):
            if v.name != "F":
                solution.add_objective_value(v.name, value(v))
        return solution

    @staticmethod
    def __get_objective_list(model: Model):
        return list(model.component_data_objects(Objective, active=True))

    def __solve_epsilon_augmenton(self, grid_points: dict):
        self.__epsilon_constraints(self.model, grid_points)
        result = self.__solve_instance(self.model, self.time_limit, self.solver)

        self.__delete_epsilon_constraints(self.model)

        if result.solver.status == SolverStatus.ok:
            return self.__get_solutions(self.model)
        else:
            return None

    def execute(self, time_limit=10) -> list:
        objectives = self.__get_objective_list(self.model)
        objectives_name = []
        self.time_limit = time_limit

        # calculate payoff table
        payoff_table = defaultdict(list)

        # For each objective solve the problem in a lexicograph way
        # maximize the target objectives and the rest try to maximize
        # without loosing the objective target value found
        # With those values in each iteration we are filling the
        # table

        # Deactivate all objectives
        for obj in objectives:
            obj.deactivate()
            objectives_name.append(obj.name)

        for obj in objectives:
            obj.activate()
            self.__solve_instance(self.model, self.time_limit, self.solver)
            obj.deactivate()
            payoff_table[obj.name].append(value(obj))

            setattr(
                self.model,
                f"c_aux_{obj.name}",
                Param(initialize=payoff_table[obj.name][-1], mutable=False),
            )
            setattr(
                self.model,
                f"c_{obj.name}",
                Constraint(expr=obj >= getattr(self.model,
                                               f"c_aux_{obj.name}")),
            )

            # Solving rest objectives with lexicographic
            for obj1 in objectives:
                if obj.name != obj1.name:
                    obj1.activate()
                    self.__solve_instance(
                        self.model, timelimit=time_limit, solver=self.solver
                    )
                    payoff_table[obj1.name].append(value(obj1))
                    obj1.deactivate()
                    setattr(
                        self.model,
                        f"c_aux_{obj1.name}",
                        Param(initialize=payoff_table[obj1.name][-1],
                              mutable=False),
                    )
                    setattr(
                        self.model,
                        f"c_{obj1.name}",
                        Constraint(
                            expr=obj >= getattr(self.model, f"c_aux_{obj1.name}")
                        ),
                    )

            # delete all constraints components and
            # get min and max of each payoff table value
            for obj in objectives:
                self.model.del_component(getattr(self.model, f"c_{obj.name}"))
                self.model.del_component(getattr(self.model, 
                                                 f"c_aux_{obj.name}"))

        for obj in objectives:
            payoff_table[obj.name] = [
                min(payoff_table[obj.name]),
                max(payoff_table[obj.name]),
            ]

        payoff_table.pop(objectives[0].name)

        objective_points = {
            k: self.__get_interval_points(v[0], v[1]) for k, v in payoff_table.items()
        }

        # solve augmenton problem at once

        # S is from objective 2
        self.model.S = Set(initialize=[f"s_{obj}" for obj in objectives_name[1:]])
        self.model.s = Var(self.model.S, domain=NonNegativeReals)

        self.model.F = Objective(
            rule=self.__epsilon_objective(self.model, objectives_name[0]),
            sense=maximize,
        )

        return self.__iterate_payoff_table(objective_points)

    @staticmethod
    def __preprocess_solutions(costs: List[Solution]):
        return [list(v for _, v in sol.obj_values.items()) for sol in costs]

    def get_efficient_solutions(self, solution_list: List[Solution]):
        costs = self.__preprocess_solutions(solution_list)
        is_efficient = list(range(len(costs)))
        sol_len = len(costs[0])
        next_point_index = 0

        while next_point_index < len(costs):
            nondominated_point_mask = [
                any(costs[j][i] > costs[next_point_index][i] for i in range(sol_len))
                for j in range(len(costs))
            ]
            nondominated_point_mask[next_point_index] = True
            is_efficient = [
                is_efficient[j] for j in range(len(costs)) if nondominated_point_mask[j]
            ]
            costs = [costs[j] for j in range(len(costs)) if nondominated_point_mask[j]]
            next_point_index = sum(nondominated_point_mask[:next_point_index]) + 1

        return [solution_list[idx] for idx in is_efficient]
