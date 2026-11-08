import logging

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_PLAN_STEPS

from .llm import get_structured_llm
from .prompts import DECOMPOSER_PROMPT
from .schemas import TaskDecomposition
from .state import AgentState
from .utils import get_user_question

logger = logging.getLogger(__name__)

def decompositor_node(state: AgentState) -> dict:

    directive = state.get("research_directive") or get_user_question(state)
    intent = state.get("intent", "qa")
    feedback = state.get("feedback") or "Nenhuma"

    prompt = ChatPromptTemplate.from_messages(
        [("system", DECOMPOSER_PROMPT), ("user", "{input}")]
    )

    pipe = (prompt | get_structured_llm(TaskDecomposition))

    result = pipe.invoke({
        "max_steps": MAX_PLAN_STEPS,
        "intent": intent,
        "directive": directive,
        "feedback": feedback,
        "input": get_user_question(state),
    })

    # Limite aplicado em código
    steps = list(dict.fromkeys(s.strip() for s in result.steps if s.strip()))[:MAX_PLAN_STEPS]
    if not steps:
        steps = [directive]

    logger.info("Decompositor: %d passos: %s", len(steps), steps)
    return {"plan_steps": steps, "step_index": 0}