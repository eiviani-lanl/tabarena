from autogluon.common.space import Real, Int, Categorical
from tabarena.benchmark.models.ag.elm.ag_elm import GFDLAG

from ...utils.config_utils import ConfigGenerator


name = 'CustomELM'
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
            "relu","tanh","sigmoid","identity","softmax","softmin","log_sigmoid","log_softmax"
        ),
        "weight_scheme": Categorical(
            "zeros","range","uniform","normal","he_uniform","lecun_uniform",
            "glorot_uniform","he_normal","lecun_normal","glorot_normal"
        ),
    }

gen_elm = ConfigGenerator(model_cls=GFDLAG, manual_configs=manual_configs, search_space=search_space)


def generate_configs_elm(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
