from __future__ import annotations
import numpy as np

from autogluon.core.models import AbstractModel
from autogluon.common.space import Categorical

from autogluon.features.generators import LabelEncoderFeatureGenerator
from sklearn.preprocessing import StandardScaler


from tabarena.benchmark.models.ag.sklearn_elm.sklearn_elm import ExtremeLearningClassifier, ExtremeLearningRegressor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class SELM(AbstractModel):
    """Minimal implementation of an ELM compatible with the scikit-learn API.
    For more details on how to implement an abstract model, see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
    and compare to implementations of models under tabarena.benchmark/models/ag/.
    """

    ag_key = "sELM"
    ag_name = "sklearnELM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        """Model-specific preprocessing of the input data."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)

        X = X.fillna(0).to_numpy(dtype=np.float32)

        if is_train:
            self.sc = StandardScaler().fit(X)

        X = self.sc.transform(X)
        return X

    def _fit(
        self,
        X: pd.DataFrame,  # training data
        y: pd.Series,  # training labels
        # X_val=None,  # val data
        # y_val=None,  # val labels
        # time_limit=None,  # time limit in seconds (ignored in tutorial)
        num_cpus: int = 1,  # number of CPUs to use for training
        # num_gpus: int = 0,  # number of GPUs to use for training
        **kwargs,  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
    ):
        # Select model class
        if self.problem_type in ["regression"]:

            model_cls = ExtremeLearningRegressor
        else:

            # case for 'binary' and 'multiclass',
            model_cls = ExtremeLearningClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        """Default parameters for the model."""
        default_params = {
            "hidden_layer_sizes": (100,),
            "activation": "identity",
            "weight_init": "uniform",
            "direct_links": False,
            "random_state": 0,
            "ridge_alpha": 0.1,
            "rtol": None,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        """Specifics allowed input data and that all other dtypes should be handled
        by the model-agnostic preprocessor.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = {
            "valid_raw_types": ["int", "float", "category"],
        }
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

