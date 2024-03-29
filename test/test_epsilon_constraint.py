import unittest
from pyomo.environ import *
from src.utils.epsilon_constraint import EpsilonConstraintAugment, Solution


class TestEpsilonConstraint(unittest.TestCase):
    
    def test_that_epsilon_works(self):
        model = ConcreteModel()

        model.X1 = Var(within=NonNegativeReals)
        model.X2 = Var(within=NonNegativeReals)

        model.C1 = Constraint(expr = model.X1 <= 20)
        model.C2 = Constraint(expr = model.X2 <= 40)
        model.C3 = Constraint(expr = 5 * model.X1 + 4 * model.X2 <= 200)

        model.O_f1 = Objective(expr= model.X1 , sense=maximize)
        model.O_f2 = Objective(expr= 3 * model.X1 + 4 * model.X2 , sense=maximize)
        eps = EpsilonConstraintAugment(model)
        solutions = eps.execute()
        efficient_solutions = eps.get_efficient_solutions(solutions)
        
        print('')
    
class TestEfficientSolutionFilter(unittest.TestCase):
    def test_that_if_there_are_3_solution_and_one_of_them_is_nondominant_then_return_2_remaining_solutions(self):
        
        solutions = [Solution(_obj_values={'a':2, 'b':3}),
                            Solution(_obj_values={'a':3, 'b':2}),
                            Solution(_obj_values={'a':2, 'b':2})]
        expected_solutions = solutions[:2]
        
        eps = EpsilonConstraintAugment()
        returned_solutions = eps.get_efficient_solutions(solutions)
        
        self.assertListEqual(expected_solutions, returned_solutions)
