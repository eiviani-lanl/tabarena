from __future__ import annotations
import pytest


def test_felm():
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.felm.felm import FELM
        model_cls = FELM
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
