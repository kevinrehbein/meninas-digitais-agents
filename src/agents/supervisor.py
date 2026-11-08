import logging

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_structured_llm
from .prompts import SUPERVISOR_PROMPT
from .schemas import SupervisorDecision
from .state import AgentState
from .utils import get_user_question

logger = logging.getLogger(__name__)

def supervisor_node(state: AgentState) -> dict:

    iteration = state.get("iteration_count", 0)
    if iteration > 0:
        logger.info("Supervisor: reclassificação (iteração %d)", iteration)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SUPERVISOR_PROMPT), ("user", "{input}")]
    )

    pipe = prompt | get_structured_llm(SupervisorDecision)

    decision = pipe.invoke({
        "input": get_user_question(state),
        "iteracao": iteration,
        "intent": state.get("intent") or "ainda não classificada",
        "feedback": state.get("feedback") or "Nenhuma (primeira tentativa)",
    })

    logger.info(
        "Supervisor: intent=%s | complexidade=%s | diretiva=%s",
        decision.intent, decision.complexity, decision.research_directive,
    )

    estado = {
        "intent": decision.intent,
        "complexity": decision.complexity,
        "research_directive": decision.research_directive,
    }

    #inicia uma pesquisa simples, 1 passo
    if decision.intent != "fora_dominio" and decision.complexity == "simples":
        estado.update(plan_steps=[decision.research_directive], step_index=0)

    if decision.intent == "plano":
        estado.update(complexity="complexa")

    return estado
