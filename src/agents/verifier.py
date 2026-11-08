import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from config import MAX_PLAN_STEPS

from .llm import get_structured_llm
from .prompts import VERIFIER_PROMPT
from .schemas import Review
from .state import AgentState
from .utils import get_research_evidence, get_user_question

logger = logging.getLogger(__name__)

def quality_issues(draft) -> list[str]:
    if not isinstance(draft, dict):
        return []
    issues = []

    steps = draft.get("methodology", [])
    if len(steps) < 4:
        issues.append("Metodologia com menos de 4 etapas.")
    shallow = [s["etapa"] for s in steps if len(s["descricao"]) < 80]
    if shallow:
        issues.append(f"Etapas sem detalhe: {shallow}.")
    if len(draft.get("justification", "")) < 300:
        issues.append("Justificativa curta demais.")
    return issues


def verificador_node(state: AgentState) -> dict:
    """Audita o rascunho e decide o próximo passo (aprovar, reescrever, pesquisar mais)."""
    draft = state.get("draft")
    iteration = state.get("iteration_count", 0) + 1

    if not draft:
        logger.warning("Verificador: rascunho vazio.")
        return {
            "iteration_count": iteration,
            "verdict": "reescrever",
            "feedback": "O rascunho veio vazio. Produza uma resposta baseada nas evidências.",
        }

    draft_str = (
        json.dumps(draft, ensure_ascii=False, indent=2)
        if isinstance(draft, dict)
        else str(draft)
    )
    steps = state.get("plan_steps") or []

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFIER_PROMPT),
            (
                "user",
                "Pergunta do usuário: {question}\n\nRascunho a ser avaliado:\n\n{draft}",
            ),
        ]
    )

    pipe = prompt | get_structured_llm(Review)

    evidence = get_research_evidence(state)

    review: Review = pipe.invoke(
        {
            "intent": state.get("intent") or "qa",
            "passos": "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
            or "(nenhum)",
            "contexto": evidence,
            "max_steps": MAX_PLAN_STEPS,
            "question": get_user_question(state),
            "draft": draft_str,
        }
    )

    missing = [s.strip() for s in review.missing_steps if s.strip()][:MAX_PLAN_STEPS]
    verdict = review.verdict
    if verdict == "pesquisar_mais" and not missing:
        verdict = "reescrever"  # sem buscas concretas não há o que pesquisar

    feedback = review.feedback
    if verdict == "aprovado":
        problems = quality_issues(draft)
        if problems:
            verdict = "reescrever"
            feedback = " ".join(problems)
    logger.info("Verificador: iteração %d | veredito=%s", iteration, verdict)

    update: dict = {
        "iteration_count": iteration,
        "verdict": verdict,
        "feedback": None,
    }
    if verdict == "aprovado":
        return update

    update["feedback"] = feedback or "Reprovado sem detalhes; revise o grounding."
    logger.info("Verificador: crítica: %s", update["feedback"])

    if verdict == "pesquisar_mais":
        update.update(plan_steps=missing, step_index=0)  # nova rodada só com o que falta
    return update