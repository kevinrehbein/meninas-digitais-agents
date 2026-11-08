import logging
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import MAX_TOOL_CALLS

from .llm import get_llm_with_tools
from .prompts import RESEARCHER_PROMPT
from .state import AgentState
from .tools import RESEARCHER_TOOLS
from .utils import get_user_question

logger = logging.getLogger(__name__)


def pesquisador_node(state: AgentState) -> dict:
    """Escolhe as ferramentas de UM passo; o ToolNode as executa."""
    steps, index = state["plan_steps"], state["step_index"] # pyright: ignore[reportTypedDictNotRequiredAccess]
    step = steps[index]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESEARCHER_PROMPT),
            ("user", "Pergunta original do usuário: {input}\nEscolha as ferramentas para o passo atual."),
        ]
    )
    response = (prompt | get_llm_with_tools(RESEARCHER_TOOLS)).invoke(
        {
            "intent": state.get("intent", "qa"),
            "step": step,
            "step_number": index + 1,
            "total_steps": len(steps),
            "max_calls": MAX_TOOL_CALLS,
            "input": get_user_question(state),
        }
    )
    # chamada genérica de ferramenta em caso de falha
    tool_calls = list(response.tool_calls)[:MAX_TOOL_CALLS] or [
        {
            "name": "pesquisar_contexto_wit",
            "args": {"pergunta": step},
            "id": f"call_{uuid4().hex}",
            "type": "tool_call",
        }
    ]
    logger.info("Pesquisador: passo %d/%d", index + 1, len(steps))

    return {
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
        "step_index": index + 1,
    }