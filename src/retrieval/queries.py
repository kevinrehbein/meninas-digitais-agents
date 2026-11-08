"""Consultas Cypher estruturadas usadas pelas ferramentas do pesquisador.

ASSUMPTIONS sobre o schema do grafo (ajuste se o seu for diferente):
- (:Document)-[:PARTE_DE]->(:Artigo {titulo})
- (:Document)-[:MENTIONS]->(:__Entity__ {id})
- Autores são entidades com o label `Author` (id = nome).
"""

from .connection import get_graph

# Labels não podem ser parametrizados em Cypher; usamos uma allowlist para
# impedir injeção de Cypher via argumento gerado pelo LLM.
ENTITY_LABELS: dict[str, str] = {"Author": "Author"}


def find_similar_projects_by_keyword(
    keywords: list[str], limit: int = 10
) -> list[dict]:
    cleaned = [k.strip().lower() for k in keywords if k and k.strip()]
    if not cleaned:
        return []
    return get_graph().query(
        """
        MATCH (e:__Entity__)
        WHERE any(k IN $kws WHERE toLower(e.id) CONTAINS k)
        MATCH (d:Document)-[:MENTIONS]->(e)
        MATCH (d)-[:PARTE_DE]->(a:Artigo)
        RETURN a.titulo AS titulo,
               collect(DISTINCT e.id)[..8] AS temas_em_comum,
               count(DISTINCT e) AS aderencia
        ORDER BY aderencia DESC
        LIMIT $limit
        """,
        {"kws": cleaned, "limit": limit},
    )


def get_entity_profile(name: str, entity_type: str = "Author") -> list[dict]:
    label = ENTITY_LABELS.get(entity_type)
    if label is None:
        raise ValueError(f"Tipo de entidade não suportado: {entity_type!r}")
    graph = get_graph()
    params = {"nome": name.strip()}

    publicacoes = graph.query(
        f"""
        MATCH (p:`{label}`) WHERE toLower(p.id) = toLower($nome)
        OPTIONAL MATCH (p)<-[:MENTIONS]-(:Document)-[:PARTE_DE]->(a:Artigo)
        RETURN p.id AS name,
               count(DISTINCT a) AS total_publicacoes,
               collect(DISTINCT a.titulo) AS titulos
        """,
        params,
    )
    if not publicacoes:
        return []

    topicos = graph.query(
        f"""
        MATCH (p:`{label}`) WHERE toLower(p.id) = toLower($nome)
        MATCH (p)<-[:MENTIONS]-(d:Document)-[:MENTIONS]->(t:__Entity__)
        WHERE t <> p AND NOT t:`{label}`
        RETURN t.id AS topico, count(DISTINCT d) AS frequencia
        ORDER BY frequencia DESC
        LIMIT 5
        """,
        params,
    )
    perfil = dict(publicacoes[0])
    perfil["principais_topicos"] = [t["topico"] for t in topicos]
    return [perfil]