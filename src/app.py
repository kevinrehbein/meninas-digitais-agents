import logging

import streamlit as st
from PIL import Image

from agents.graph import graph_builder
from agents.state import initial_state
from agents.utils import format_final_response
from config import GRAPH_RECURSION_LIMIT, MAX_QUESTION_CHARS, validate_config
from retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("menidas_digitais.app")

img_icon = Image.open("media/meninas-digitais-icon.png")

st.set_page_config(page_title="Meninas Digitais Assistente", page_icon=img_icon)

EXAMPLES = [
    "Quais artigos falam sobre mentoria para meninas?",
    "Faça um panorama sobre a inclusão de mulheres na computação.",
    "Crie um plano de projeto de oficina de programação para alunas do ensino médio.",
]

NODE_LABELS = {
    "redator": "✍️ Redator — texto gerado",
    "planejador": "🗺️ Planejador — plano estruturado",
    "finalizar": "📦 Resposta final compilada",
    "fora_dominio": "🚫 Fora do domínio do WIT",
}


@st.cache_resource(
    show_spinner="Carregando modelo de embeddings e conectando ao grafo…"
)
def load_graph():
    """Roda uma vez por processo: aquece o BGE-M3 e compila o grafo."""
    validate_config()
    HybridRetriever.get_instance().embedding_model.embed_query("ping")
    return graph_builder()


def describe(node: str, update: dict) -> str:
    """Texto curto de progresso para cada nó do grafo."""
    if node == "supervisor":
        return (
            f"🧭 **Supervisor** — intenção `{update.get('intent')}`, "
            f"complexidade `{update.get('complexity')}`"
        )
    if node == "decompositor":
        steps = update.get("plan_steps", [])
        lista = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        return f"🧩 **Decompositor** — {len(steps)} passos:\n{lista}"
    if node == "pesquisador":
        names = ", ".join(c["name"] for c in update["messages"][0].tool_calls)
        return f"🔎 **Pesquisador** — consultando: `{names}`"
    if node == "pesquisador_tools":
        return "📚 Ferramentas — evidências recebidas"
    if node == "verificador":
        feedback = update.get("feedback")
        extra = f"\n> {feedback}" if feedback else ""
        return f"🧪 **Verificador** — veredito `{update.get('verdict')}`{extra}"
    return NODE_LABELS.get(node, node)


def run_question(graph, question: str) -> dict:
    """Executa o grafo mostrando cada etapa; devolve o estado final acumulado."""
    final: dict = {}
    with st.status("Processando…", expanded=True) as status:
        for chunk in graph.stream(
            initial_state(question),
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            stream_mode="updates",  # um evento por nó, com o que ele alterou
        ):
            for node, update in chunk.items():
                if isinstance(update, dict):
                    final.update({k: v for k, v in update.items() if k != "messages"})
                    status.write(describe(node, update))
        status.update(label="Concluído", state="complete", expanded=False)
    return final


# ----------------------------------------------------------------------- UI
st.title("Assistente Meninas Digitais")
st.caption(
    "Respostas fundamentadas nos artigos do WIT. Cada pergunta é independente (sem memória de conversa)."
)

try:
    graph = load_graph()
except Exception as exc:
    logger.exception("Falha na inicialização")
    st.error(f"Não foi possível inicializar o sistema: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Exemplos")
    for example in EXAMPLES:
        st.button(
            example,
            on_click=lambda e=example: st.session_state.update(pending=e),
            use_container_width=True,
        )
    st.button(
        "Limpar conversa",
        on_click=lambda: st.session_state.update(history=[]),
        use_container_width=True,
    )

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Pergunte sobre o WIT…") or st.session_state.pop(
    "pending", None
)

if question:
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Pergunta muito longa (máx. {MAX_QUESTION_CHARS} caracteres).")
        st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            final = run_question(graph, question)
        except Exception:
            logger.exception("Falha ao processar a pergunta")
            st.error(
                "Ocorreu um erro ao processar sua pergunta. Veja os logs para detalhes."
            )
            st.stop()

        answer = format_final_response(final.get("final_response"))
        st.markdown(answer)
        if final.get("warning"):
            st.warning(final["warning"])

    st.session_state.history.append({"role": "assistant", "content": answer})
