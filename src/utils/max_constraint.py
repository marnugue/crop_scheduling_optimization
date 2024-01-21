from pyomo.environ import Constraint, Model, Var, Binary, Reals
from typing import Any

from common_utils import kwargs_var_and_name


def max_constraints(model: Model, var1: str | Any, var2: str | Any, **kwargs):
    BIG_M = 10e6

    (x, var1_name), (y, var2_name) = kwargs_var_and_name(model, var1, var2, **kwargs)

    var_name_aux = f"max_var_aux_{var1_name}_{var2_name}"
    max_value_name = f"max_value_{var1_name}_{var2_name}"
    max_value_name_alternative = f"max_value_{var2_name}_{var1_name}"

    if not hasattr(model, max_value_name) and not hasattr(
        model, max_value_name_alternative
    ):
        var_aux = Var(range(2), domain=Binary)

        max_value = Var(domain=Reals)

        setattr(model, var_name_aux, var_aux)

        setattr(model, max_value_name, max_value)

        c1 = Constraint(expr=x <= max_value)
        c2 = Constraint(expr=y <= max_value)
        c3 = Constraint(expr=x >= BIG_M * (1 - getattr(model, var_name_aux)[0]))
        c4 = Constraint(expr=y >= BIG_M * (1 - getattr(model, var_name_aux)[1]))
        c5 = Constraint(
            expr=sum(getattr(model, var_name_aux)[i] for i in range(2)) >= 1
        )
        setattr(model, f"c1_{max_value_name}", c1)
        setattr(model, f"c2_{max_value_name}", c2)
        setattr(model, f"c3_{max_value_name}", c3)
        setattr(model, f"c4_{max_value_name}", c4)
        setattr(model, f"c5_{max_value_name}", c5)

    return (
        max_value_name if hasattr(model, max_value_name) else max_value_name_alternative
    )
