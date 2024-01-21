def kwargs_var_and_name(model, var1, var2, **kwargs):
    
    if kwargs:
        keys = list(kwargs.keys())
        x = var1
        y = var2
        var1_name = kwargs[keys[0]]
        var2_name = kwargs[keys[1]]
    else:
        x = getattr(model, var1)
        y = getattr(model, var2)
        var1_name = var1
        var2_name = var2
        
    return (x, var1_name), (y, var2_name)