from skopt.utils import use_named_args
from skopt.space import *
from lenskit.algorithms.als import ImplicitMF

impl = ImplicitMF

space = [
    ('features', Integer(5, 250, 'log-uniform')),
    ('ureg', Real(10e-5, 10, 'log-uniform')),
    ('ireg', Real(10e-5, 10, 'log-uniform')),
    ('weight', Real(1, 50)),
    ('epochs', Integer(5, 30)),
]

def default():
    return ImplicitMF(50)

@use_named_args(space)
def from_params(features, ureg, ireg, weight, epochs):
    return ImplicitMF(features, reg=(ureg, ireg), weight=weight, iterations=epochs)
