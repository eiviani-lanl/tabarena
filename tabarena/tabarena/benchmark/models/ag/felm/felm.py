from __future__ import annotations
import numpy as np

from autogluon.core.models import AbstractModel
from autogluon.common.space import Categorical

from autogluon.features.generators import LabelEncoderFeatureGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

class RidgeClassifier_(BaseEstimator, ClassifierMixin): # predict_proba() is not a function included in Ridge, and is required by TabArena's HO
    def __init__(self, alpha=1.0, fit_intercept=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.clf = RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        decision = self.clf.decision_function(X)
        if len(self.classes_) == 2:
            decision = np.column_stack([-decision, decision])
        return softmax(decision, axis=1)


class FELM(AbstractModel):
    """Minimal implementation of an ELM compatible with the scikit-learn API.
    For more details on how to implement an abstract model, see https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html
    and compare to implementations of models under tabarena.benchmark/models/ag/.
    """

    ag_key = "felm"
    ag_name = "FELM"

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

        X = X.fillna(-1).to_numpy(dtype=np.float32)

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
        

        params = self._get_model_params()
        params['reg_alpha'] = 0.0 if float(params['reg_alpha'])<=1e-6 else params['reg_alpha']

        steps = []
        for i, layer in enumerate(params['n_hidden']):
                steps.append((f"features{i}",GaussianRandomProjection(n_components = layer, random_state = params['seed']+i))) # use different seed for each hidden layer (could use rng?)
        if self.problem_type in ["regression"]:
            steps.append(("ridge", Ridge(alpha=params['reg_alpha'],fit_intercept=False)))
        else:
            steps.append(("ridge",RidgeClassifier_(alpha=params['reg_alpha'],fit_intercept=False)))

        self.model = Pipeline(steps)

        X = self.preprocess(X, is_train=True)

        self.model.fit(X, y)

    def _set_default_params(self):
        """Default parameters for the model."""
        default_params = {
            
            "n_hidden": (100,),
            "reg_alpha": 1.0,
            "seed": self.model_random_seed if hasattr(self, "model_random_seed") else 0,
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

