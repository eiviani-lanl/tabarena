from autogluon.common.space import Real, Int, Categorical, Bool
from tabarena.benchmark.models.ag.gfdl.gfdl import GFDL

from ...utils.config_utils import ConfigGenerator


name = 'GFDL'
manual_configs = [
    {},
]
search_space = {
        "hidden_layer_sizes": Categorical(
            (100,), (200,), (400,), 
            (100, 100), (200, 200), (400, 400),
            (100, 100, 100), (200, 200, 200),
            (400, 400, 400, 400, 400),
            
        ),
        "activation": Categorical(
            "identity",
            "tanh",
            "relu",
            "sigmoid",
            "softmax",
            "softmin",
            "log_sigmoid",
            "log_softmax",
        ),
        "weight_scheme": Categorical(
            "zeros",
            "uniform",
            "normal",
            "he_uniform",
            "lecun_uniform",
            "glorot_uniform",
            "he_normal",
            "lecun_normal",
            "glorot_normal",
        ),
        "direct_links": Categorical(True,False),
        "reg_alpha": Real(0.0, 500.0, default=None, log=True),
        "rtol": Real(0.0, 1e-2, default=None, log=True)
    }

gen_gfdl = ConfigGenerator(model_cls=GFDL, manual_configs=manual_configs, search_space=search_space)


def generate_configs_gfdl(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
