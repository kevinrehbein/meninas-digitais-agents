import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from config import EVIDENCE_CHAR_LIMIT, TOOL_ERROR

from .state import AgentState


def message_text(message: BaseMessage) -> str:
    """Texto da mensagem; alguns provedores (ex.: Gemini) devolvem lista de blocos."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "") for part in content
    )


def get_user_question(state: AgentState) -> str:
    """Pergunta original do usuário (nunca a última mensagem, que pode ser de ferramenta)."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def get_research_evidence(state: AgentState) -> str:
    """Evidências coletadas para a pergunta atual (todos os passos e rodadas).

    Redator, planejador e verificador usam esta função, então o verificador julga
    exatamente o que o autor viu. Erros de ferramenta e resultados repetidos são
    descartados; o orçamento de caracteres é dividido igualmente entre as fontes.
    """
    messages = state.get("messages", [])
    calls = {
        c["id"]: c["args"]
        for m in messages
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)

    blocks: dict[str, str] = {}  # texto -> origem (o dict já elimina duplicatas)
    for m in messages[start + 1 :]:
        if isinstance(m, ToolMessage):
            text = message_text(m)
            if text != TOOL_ERROR:
                args = json.dumps(calls.get(m.tool_call_id, {}), ensure_ascii=False)
                blocks.setdefault(text, f"{m.name}({args})")
    if not blocks:
        return "(nenhuma evidência coletada)"

    budget = EVIDENCE_CHAR_LIMIT // len(blocks)
    return "\n\n".join(
        f"--- Origem: {origin} ---\n{text[:budget]}"
        + (" [...truncado...]" if len(text) > budget else "")
        for text, origin in blocks.items()
    )


def build_revision_alert(feedback: str | None, artifact: str) -> str:
    """Bloco de correção injetado nos prompts quando o verificador reprova."""
    if not feedback:
        return ""
    return (
        "\n\n🚨 ALERTA DE REVISÃO DO AVALIADOR 🚨\n"
        f"Sua tentativa anterior ({artifact}) foi REJEITADA pelo seguinte motivo:\n"
        f"'{feedback}'\n"
        "CORRIJA ESTE ERRO AGORA. Baseie-se apenas nas evidências fornecidas."
    )


def render_project_plan(plan: dict[str, Any]) -> str:
    steps = "\n".join(
        f"{i}. **{s['etapa']}** — {s['descricao']} "
        + (f"_(fonte: {s['fonte']})_" if s.get("fonte") else "_(proposta do planejador)_")
        for i, s in enumerate(plan["methodology"], 1)
    )
    sections = [
        ("Desafios", "challenges"),
        ("Indicadores de sucesso", "indicators"),
        ("Resultados esperados", "expected_results"),
        ("Referências", "references"),
    ]
    tail = "".join(
        f"\n\n## {title}\n" + ("\n".join(f"- {x}" for x in plan[key]) or "- (nenhuma)")
        for title, key in sections
    )
    return (
        f"# {plan['title']}\n\n"
        f"**Público-alvo:** {plan['target_audience']}\n\n"
        f"## Justificativa\n{plan['justification']}\n\n"
        f"## Metodologia\n{steps}{tail}"
    )


def format_final_response(response: str | dict[str, Any] | None) -> str:
    if not response:
        return "Não foi possível gerar uma resposta."
    if isinstance(response, dict):
        return render_project_plan(response)
    return str(response)