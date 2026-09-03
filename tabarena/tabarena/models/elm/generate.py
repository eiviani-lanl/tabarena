from autogluon.common.space import Real, Int, Categorical, Bool
from tabarena.benchmark.models.ag.sklearn_elm.ag_elm import SELM

from ...utils.config_utils import ConfigGenerator


name = 'sklearnELM'
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
            "logistic",
            "softmax",
            "softmin",
            "log_sigmoid",
            "log_softmax",
        ),
        "weight_init": Categorical(
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
        "direct_links": Categorical(True,False)
    }

gen_elm = ConfigGenerator(model_cls=SELM, manual_configs=manual_configs, search_space=search_space)


def generate_configs_elm(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
