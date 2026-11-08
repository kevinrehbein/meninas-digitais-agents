from .prompts import OUT_OF_SCOPE_MESSAGE
from .state import AgentState


def finalizar_node(state: AgentState) -> dict:

    update = {"final_response": state.get("draft")}

    if state.get("verdict") != "aprovado":
        update["warning"] = (
            "Limite de iterações atingido: esta resposta não passou integralmente "
            "na verificação de fidelidade às fontes. Confira as informações."
        )
    return update


def fora_dominio_node(state: AgentState) -> dict:
    return {"final_response": OUT_OF_SCOPE_MESSAGE}
