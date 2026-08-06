from autogluon.common.space import Real, Int, Categorical, Bool
from tabarena.benchmark.models.ag.felm.felm import FELM

from ...utils.config_utils import ConfigGenerator


name = 'FELM'
manual_configs = [
    {},
]
search_space = {
        "n_hidden": Categorical(
            (100,), (200,), (400,), 
            (100, 100), (200, 200), (400, 400),
            (100, 100, 100), (200, 200, 200),
            (400, 400, 400, 400, 400),
            
        ),
        "gamma": Real(1e-3, 10.0, default=1.0, log=True),
        "reg_alpha": Real(1e-7, 500.0, default=1.0, log=True),
    }

gen_felm = ConfigGenerator(model_cls=FELM, manual_configs=manual_configs, search_space=search_space)


def generate_configs_felm(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
