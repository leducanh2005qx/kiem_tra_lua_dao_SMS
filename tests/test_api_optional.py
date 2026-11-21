import pytest


fastapi = pytest.importorskip("fastapi")


def test_api_module_exposes_fastapi_app_when_extra_is_installed():
    from api.main import app

    assert app is not None
