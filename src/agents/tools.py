import functools
import json
import logging
from typing import Literal

from langchain_core.tools import BaseTool, tool

from config import TOOL_ERROR
from retrieval import queries
from retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def _safe_tool(func):
    """Falha de infraestrutura (Neo4j, embeddings) vira texto, não derruba o grafo."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Falha na ferramenta %s", func.__name__)
            return TOOL_ERROR

    return wrapper


@tool
@_safe_tool
def buscar_artigos_wit(query: str) -> str:
    """
    Lista os artigos do evento Women in Information Technology (WIT) mais relevantes
    para um texto ou pergunta, com um trecho de cada um.

    Args:
        query: O termo de busca ou pergunta (ex: "inclusão de mulheres na programação").
    """
    results = HybridRetriever.get_instance().buscar_artigos(query)
    if not results:
        return "Nenhum artigo encontrado para esta busca no WIT."
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
@_safe_tool
def pesquisar_contexto_wit(pergunta: str) -> str:
    """
    Busca informações cruciais na base de dados do evento Women in Information Technology (WIT).
    Usa GraphRAG para cruzar similaridade de artigos com relações de entidades no Neo4j.
    Sempre use esta ferramenta para coletar evidências antes de responder ou planejar.

    Args:
        pergunta: A dúvida ou tópico a ser pesquisado (ex: "projetos de mentoria para meninas").
    """
    return HybridRetriever.get_instance().recuperar_contexto(pergunta)


@tool
@_safe_tool
def buscar_projetos_semelhantes(palavras_chave: list[str]) -> str:
    """
    Encontra artigos e projetos no WIT que focam especificamente em um conjunto de palavras-chave.
    Útil para verificar lacunas na literatura ou levantar projetos que já fizeram algo similar.

    Args:
        palavras_chave: Lista de strings com os temas de interesse (ex: ["Mentoria", "Meninas na Computação"]).
    """
    results = queries.find_similar_projects_by_keyword(palavras_chave)
    if not results:
        return "Nenhum projeto estruturalmente semelhante encontrado com essas palavras-chave."
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
@_safe_tool
def perfil_entidade(nome: str, tipo: Literal["Author"] = "Author") -> str:
    """
    Busca o perfil completo de uma entidade no grafo, como um autor específico.
    Retorna o total de publicações, títulos e os principais tópicos pesquisados por essa pessoa.

    Args:
        nome: O nome exato da entidade (ex: "Renata Reiser").
        tipo: O tipo da entidade (atualmente apenas "Author").
    """
    results = queries.get_entity_profile(nome, tipo)
    if not results:
        return f"Perfil não encontrado para {nome}."
    return json.dumps(results, ensure_ascii=False, default=str)


RESEARCHER_TOOLS: list[BaseTool] = [
    buscar_artigos_wit,
    pesquisar_contexto_wit,
    buscar_projetos_semelhantes,
    perfil_entidade,
]
