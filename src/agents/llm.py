import logging
import os
from collections.abc import Callable, Sequence
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import stop_after_attempt

from config import LLM_CHAIN, PROVIDER_API_KEYS

logger = logging.getLogger(__name__)


@cache
def _available_models(temperature: float) -> tuple[BaseChatModel, ...]:
    """Instancia (uma única vez) só os modelos cujo provedor tem chave configurada."""
    models: list[BaseChatModel] = []
    for provider, name in LLM_CHAIN:
        key_var = PROVIDER_API_KEYS.get(provider)
        if key_var and not os.getenv(key_var):
            logger.debug("Ignorando %s/%s: %s não definida", provider, name, key_var)
            continue
        try:
            models.append(
                init_chat_model(
                    model=name,
                    model_provider=provider,
                    temperature=temperature,
                    max_retries=0,
                    timeout=45
                )
            )
        except Exception as exc:  # provedor/modelo indisponível não deve derrubar o app
            logger.warning("Falha ao inicializar %s/%s: %s", provider, name, exc)
    if not models:
        raise RuntimeError("Nenhum LLM disponível: verifique chaves de API e LLM_CHAIN.")
    return tuple(models)


def _compose(
    binder: Callable[[BaseChatModel], Runnable], temperature: float
) -> Runnable:
    runnables: list[Runnable] = []
    for model in _available_models(temperature):
        try:
            runnables.append(binder(model))
        except (NotImplementedError, ValueError) as exc:
            logger.warning("Modelo não suporta a configuração pedida: %s", exc)
    if not runnables:
        raise RuntimeError("Nenhum modelo suporta a configuração solicitada.")
    primary, *fallbacks = runnables
    chain = primary.with_fallbacks(fallbacks) if fallbacks else primary
    return chain.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )


def get_llm(temperature: float = 0.0) -> Runnable:
    return _compose(lambda m: m, temperature)


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.0) -> Runnable:
    return _compose(lambda m: m.with_structured_output(schema), temperature)


def get_llm_with_tools(tools: Sequence[BaseTool], temperature: float = 0.0) -> Runnable:
    return _compose(lambda m: m.bind_tools(list(tools)), temperature)
