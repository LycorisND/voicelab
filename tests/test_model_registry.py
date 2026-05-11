import pytest
from voicelab.core.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a clean registry."""
    ModelRegistry.instance().clear()
    yield
    ModelRegistry.instance().clear()


def test_register_and_get():
    reg = ModelRegistry.instance()
    reg.register("dummy", lambda: {"weights": [1, 2, 3]})
    model = reg.get("dummy")
    assert model == {"weights": [1, 2, 3]}


def test_lazy_loading():
    load_count = {"n": 0}

    def loader():
        load_count["n"] += 1
        return object()

    reg = ModelRegistry.instance()
    reg.register("lazy", loader)
    assert not reg.is_loaded("lazy")
    reg.get("lazy")
    assert reg.is_loaded("lazy")
    assert load_count["n"] == 1


def test_cached_on_second_call():
    call_count = {"n": 0}

    def loader():
        call_count["n"] += 1
        return object()

    reg = ModelRegistry.instance()
    reg.register("cached", loader)
    first = reg.get("cached")
    second = reg.get("cached")
    assert first is second
    assert call_count["n"] == 1


def test_unregistered_raises():
    with pytest.raises(KeyError, match="not registered"):
        ModelRegistry.instance().get("nonexistent")


def test_singleton():
    assert ModelRegistry.instance() is ModelRegistry.instance()
