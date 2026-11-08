from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["qa", "sintese", "plano", "fora_dominio"] | None
    complexity: Literal["simples", "complexa"] | None
    research_directive: str | None

    # --- Execução da pesquisa (plan-and-execute) ---
    plan_steps: list[str]  # passos de pesquisa da rodada atual
    step_index: int  # passo em execução

    # --- Geração e verificação ---
    draft: str | dict[str, Any] | None  # str (qa/síntese) ou ProjectPlan
    final_response: str | dict[str, Any] | None
    iteration_count: int
    verdict: Literal["aprovado", "reescrever", "pesquisar_mais", "reclassificar"] | None
    feedback: str | None
    warning: str | None


def initial_state(question: str) -> AgentState:

    return {
        "messages": [HumanMessage(content=question)],
        "plan_steps": [],
        "step_index": 0,
        "iteration_count": 0,
        "verdict": None,
        "feedback": None,
        "warning": None,
    }