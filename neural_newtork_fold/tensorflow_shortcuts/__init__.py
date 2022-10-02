import numpy as np
def cdata(data, type=float):
    return np.array(data, dtype=type)
def predict(model, data, Type):
    return model.predict(np.array([data], dtype=Type))
