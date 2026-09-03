from __future__ import annotations
import pytest


def test_gfdl():
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.gfdl.gfdl import ELM
        model_cls = ELM
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.gfdl.gfdl import RVFL
        model_cls = RVFL
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
