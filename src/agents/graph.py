import logging
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from config import MAX_ITERATIONS

from .decomposer import decompositor_node
from .finalizer import finalizar_node, fora_dominio_node
from .planner import planejador_node
from .researcher import pesquisador_node
from .state import AgentState
from .supervisor import supervisor_node
from .tools import RESEARCHER_TOOLS
from .verifier import verificador_node
from .writer import redator_node

logger = logging.getLogger(__name__)

def route_from_supervisor(
    state: AgentState,
) -> Literal["fora_dominio", "decompositor", "pesquisador"]:
    if state.get("intent") == "fora_dominio":
        return "fora_dominio"
    return "decompositor" if state.get("complexity") == "complexa" else "pesquisador"


def route_after_tools(
    state: AgentState,
) -> Literal["pesquisador", "planejador", "redator"]:
    if state["step_index"] < len(state["plan_steps"]): # pyright: ignore[reportTypedDictNotRequiredAccess]
        return "pesquisador"  # faltam passos
    return "planejador" if state.get("intent") == "plano" else "redator"


def route_from_verifier(
    state: AgentState,
) -> Literal["finalizar", "supervisor", "pesquisador", "planejador", "redator"]:
    verdict = state.get("verdict")
    if verdict == "aprovado":
        return "finalizar"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        logger.warning(
            "Limite de iterações atingido; entregando a melhor resposta atual."
        )
        return "finalizar"
    if verdict == "reclassificar":
        return "supervisor"
    if verdict == "pesquisar_mais":
        return "pesquisador"
    return "planejador" if state.get("intent") == "plano" else "redator"  # "reescrever": evidências bastam, sem nova pesquisa


@lru_cache(maxsize=1)
def graph_builder() -> CompiledStateGraph:

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("fora_dominio", fora_dominio_node)
    builder.add_node("decompositor", decompositor_node)
    builder.add_node("pesquisador", pesquisador_node)
    builder.add_node("pesquisador_tools", ToolNode(RESEARCHER_TOOLS))
    builder.add_node("redator", redator_node)
    builder.add_node("planejador", planejador_node)
    builder.add_node("verificador", verificador_node)
    builder.add_node("finalizar", finalizar_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["fora_dominio", "decompositor", "pesquisador"],
    )
    builder.add_edge("decompositor", "pesquisador")

    builder.add_edge("pesquisador", "pesquisador_tools")
    builder.add_conditional_edges(
        "pesquisador_tools", route_after_tools, ["pesquisador", "planejador", "redator"]
    )

    builder.add_edge("redator", "verificador")
    builder.add_edge("planejador", "verificador")
    builder.add_conditional_edges(
        "verificador",
        route_from_verifier,
        ["finalizar", "supervisor", "pesquisador", "planejador", "redator"],
    )
    builder.add_edge("finalizar", END)
    builder.add_edge("fora_dominio", END)

    return builder.compile()
