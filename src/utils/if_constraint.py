import pyomo.environ as pyo
from common_utils import kwargs_var_and_name


def if_constraint(model: pyo.Model, left_clause, right_clause, **kwargs):
    (_, var1_name), (_, var2_name) = kwargs_var_and_name(
        model, left_clause, right_clause, **kwargs
    )

    BIG_M = 10e6
    # simulate restricted >
    EPSILON = 10e-6

    b = pyo.Var(domain=pyo.Binary)

    b_name = f"if_constraint_{var1_name}_{var2_name}"

    setattr(model, b_name, b)

    b_model = getattr(model, b_name)

    c1 = pyo.Constraint(left_clause >= right_clause + EPSILON + BIG_M * (1 - b_model))

    c2 = pyo.Constraint(left_clause <= right_clause + BIG_M * b_model)

    c1_name = f"c1_{b_name}"
    c2_name = f"c2_{b_name}"

    setattr(model, c1_name, c1)
    setattr(model, c2_name, c2)

    return b_name
