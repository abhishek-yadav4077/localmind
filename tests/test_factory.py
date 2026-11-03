import pytest
from pydantic import ValidationError

from localmind.config import Settings
from localmind.llm import ModelParams, get_chat_model


def test_factory_applies_params(settings: Settings) -> None:
    params = ModelParams(temperature=0.1, top_p=0.5, max_tokens=256, num_ctx=8192)
    llm = get_chat_model("mistral", params, settings)
    assert llm.model == "mistral"
    assert llm.temperature == 0.1
    assert llm.top_p == 0.5
    assert llm.num_predict == 256
    assert llm.num_ctx == 8192
    assert llm.base_url == settings.ollama_base_url


def test_factory_defaults_to_default_model(settings: Settings) -> None:
    assert get_chat_model(settings=settings).model == settings.default_model


@pytest.mark.parametrize("bad", [{"temperature": 3}, {"top_p": 0}, {"num_ctx": 10}])
def test_param_validation(bad: dict) -> None:
    with pytest.raises(ValidationError):
        ModelParams(**bad)
