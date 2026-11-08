from langchain_core.prompts import ChatPromptTemplate

from .llm import get_structured_llm
from .prompts import PLANNER_PROMPT
from .schemas import ProjectPlan
from .state import AgentState
from .utils import build_revision_alert, get_research_evidence, get_user_question


def planejador_node(state: AgentState) -> dict:
    """Estrutura um projeto; a saída é validada pelo schema Pydantic `ProjectPlan`."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", PLANNER_PROMPT), ("user", "Solicitação do usuário: {input}")]
    )

    projeto: ProjectPlan = (prompt | get_structured_llm(ProjectPlan)).invoke(
        {
            "contexto": get_research_evidence(state),
            "alerta": build_revision_alert(state.get("feedback"), "plano"),
            "input": get_user_question(state),
        }
    )
    return {"draft": projeto.model_dump()}