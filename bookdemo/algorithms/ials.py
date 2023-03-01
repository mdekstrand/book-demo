from scipy.stats import zipfian, loguniform, uniform, randint
from lenskit.algorithms.als import ImplicitMF

impl = ImplicitMF

space = [
    # log-uniform (Zipf) distribution [5, 250]
    ('features', zipfian(1, 246, loc=4)),
    ('ureg', loguniform(1.0e-5, 10)),
    ('ireg', loguniform(1.0e-5, 10)),
    ('weight', uniform(1, 50)),
    ('epochs', randint(5, 30)),
]

def default():
    return ImplicitMF(50)

def from_params(features, ureg, ireg, weight, epochs):
    return ImplicitMF(features, reg=(ureg, ireg), weight=weight, iterations=epochs)
