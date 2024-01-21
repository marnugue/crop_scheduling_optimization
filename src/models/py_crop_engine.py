import pyomo.environ as pyo
from pyomo.core.util import prod
from src.factory.model_factory import SolverEngine
from src.utils.max_constraint import max_constraints
from src.utils.if_constraint import if_constraint

# Warning! not done yet dont look this code until is merge 
# on master
class PYCropEngine(SolverEngine):
    def _build_parameters(self):
        pass

    def __set_solver_parameters(self):
        pass

    def execute(self, time_limit: int):
        self.model = pyo.AbstractModel()
        
        with pyo.SolverFactory('scip') as sl:
            status = sl.solve(self.model)
            
            if status.solver.status == pyo.SolverStatus.ok:
                # Process solution
                pass
        
    def _build_model(self):
         self.__build_constraints()
         
    def __build_constraints(self):
        
        pass
    
    @staticmethod
    def _exp_crop_diff_bin(model, x, h, t):
        if t == 0:
            return 0
        
        s1 = model.scheduling[x, h, t - 1]
        s2 = model.scheduling[x, h, t]
        s1_name = f'scheduling_{x}_{h}_{t}'
        s2_name = f'scheduling_{x}_{h}_{t-1}'
        
        max_s1_s2 = getattr(model, f'max_value_{s1_name}_{s2_name}')
        
        return (1 - s1 * s2) * max_s1_s2 * s2
    
    @staticmethod
    def _constraint_crop_diff_sum_bin(model, x, h, t):
        if t < model.growth[h]:
            return 0

        prev_hours = max(0, t - model.growth[h])

        prev_sum = sum(
            model.scheduling[x, h, t_i] for t_i in range(prev_hours, t)
        )
        
        b_name = if_constraint()
        
        

        return None
    
    @staticmethod
    def _constraint_max_vegetables(model, x, t):
        return sum(model.scheduling[x, h, t] for h in model.VEGETABLES) <= 1
        
    @staticmethod
    def _constraint_min_growth(model, x, h, t):
            
        if t == 0:
            return model.scheduling[x, h,  t] == 0
        
        post_hours = min(t + model.growth[h] - 1, max(model.TIME))
        
        post_sum = sum(model.scheduling[x, h, t_i] for t_i in range(t, post_hours + 1))
        
        if t + model.growth[h] - 1 > post_hours:
            max_value = (max(model.TIME) - t)  + 2
        else:
            max_value = model.growth[h] + 1
        
        return model.plant_vegetable_bin[x,h,t] * (max_value - post_sum)  <= 1
        
    def _build_objectives(self):
        pass

    def _build_solution(self):
        pass
    
def init_solution():
    init_dict = {(x, h, t): 0 for x in range(4) for h in ['H1', 'H2', 'H3'] for t in range(7)}
    for t in range(1,4):
        init_dict[0, 'H1', t] = 1
        init_dict[1, 'H2', t] = 1
        init_dict[2, 'H3', t] = 1
        init_dict[3, 'H2', t] = 1
        init_dict[0, 'H1', 4] = 1
        init_dict[0, 'H1', 5] = 1
    return init_dict
    

model = ConcreteModel()

# Sets
model.VEGETABLES = Set(initialize=['H1', 'H2', 'H3'])
model.X = RangeSet(0,3)
model.TIME = RangeSet(0,6)
# Parameters
model.growth = Param(model.VEGETABLES, initialize={
    'H1': 5,
    'H2': 3,
    'H3': 3
})

# Variables
model.scheduling = Var(model.X, model.VEGETABLES, model.TIME,
                       initialize=init_solution(), within=NonNegativeReals,
                       bounds=(0, 1))

# Expressions
# max_values
# max_var_names = []
for x in model.X:
    for h in model.VEGETABLES:
        for t in model.TIME:
            if t > 0:
                var1_name = f'scheduling_{x}_{h}_{t}'
                var2_name = f'scheduling_{x}_{h}_{t-1}'
                max_constraints(model,
                                model.scheduling[x, h, t],
                                model.scheduling[x, h, t - 1],
                                var1_str_name=var1_name,
                                var2_str_name=var2_name)


def diff_expr(model, x, h, t):
    if t > 0:
        var_name_1 = f'scheduling_{x}_{h}_{t}'
        var_name_2 = f'scheduling_{x}_{h}_{t-1}'
        return (1 - (model.scheduling[x, h, t - 1]) * (model.scheduling[x, h, t])) *  \
            getattr(model, f'max_value_{var_name_1}_{var_name_2}') * (model.scheduling[x, h, t])
    else:
        return 0


model.exp1 = Expression(model.X, model.VEGETABLES, model.TIME, rule=diff_expr)


def concat_expr(model, x, h, t):
    
    if t > model.growth[h] + 1:
        return prod(model.scheduling[x, h, t - t_i] for t_i in range(model.growth[h] + 1))
    else:
        return 0
    
model.exp2 = Expression(model.X, model.VEGETABLES, model.TIME, rule=concat_expr)
    
def _exp_plant_vegetable_bin(model, x, h, t):
    return  model.exp1[x, h, t]

model.plant_vegetable_bin = Expression(model.X, model.VEGETABLES, model.TIME, rule=_exp_plant_vegetable_bin )

# def _exp_plant_vegetable_bin(model, 


def vegetable_production(model, h):
    return sum(model.scheduling[x, h, t] for x in model.X for t in model.TIME)/model.growth[h]


# interchange vegetables

# def interchange_vegetables(model, x, h, t):
#     post_hours = min(t + model.growth[h], max(model.TIME))
    
#     return model.plant_vegetable_bin[x, h, t] * model.scheduling[x, h, post_hours] == 0

# model.interchange_constraint = Constraint(model.X, model.VEGETABLES, model.TIME, rule=interchange_vegetables)

model.supply_constraint = Constraint(model.X, model.VEGETABLES, model.TIME, rule=supply_time_growth)
model.demand_constraint = Constraint(model.X, model.TIME, rule=supply_unique_vegetables)

# model.abs_diff = Var(range(len(model.VEGETABLES)**2), initialize=0, domain=NonNegativeReals)

model.expr_vegetables = Expression(model.VEGETABLES, initialize=vegetable_production)

# Define constraints to linearize absolute differences
# def abs_diff_constraint_rule(model, i, h1, h2):
#     return model.abs_diff[i] >= model.expr_vegetables[h1] - model.expr_vegetables[h2]

# model.abs_diff_constraint_pos = Constraint(range(len(model.VEGETABLES)**2),
#                                            model.VEGETABLES,
#                                            model.VEGETABLES,
#                                            rule=abs_diff_constraint_rule)

# def abs_diff_constraint_rule_neg(m, i, h1, h2):
#     return m.abs_diff[i] >= - model.expr_vegetables[h1] +  model.expr_vegetables[h2]

# model.abs_diff_constraint_neg = Constraint(range(len(model.VEGETABLES)**2),
#                                            model.VEGETABLES,
#                                            model.VEGETABLES,
#                                            rule=abs_diff_constraint_rule_neg)


# Objective function
def production(model):
    return sum(model.scheduling[x, h, t] / model.growth[h]
               for x in model.X for h in model.VEGETABLES for t in model.TIME)


# def balanced_production(model):
#     return - sum(abs(model.expr_vegetables[v1] - model.expr_vegetables[v2]) 
#                  for v1 in model.VEGETABLES for v2 in model.VEGETABLES)

# def linear_balanced_production(model):
#     return - sum(model.abs_diff[i] for i in range(len(model.VEGETABLES)**2))


model.production = Objective(rule=production, sense=maximize)
# model.demmand = Objective(rule=linear_balanced_production, sense=maximize)

# model.demmand.deactivate()

# eps = EpsilonConstraintAugment(model, solver='cbc')
# solutions = eps.execute(time_limit=10)

# model.objective = Multi