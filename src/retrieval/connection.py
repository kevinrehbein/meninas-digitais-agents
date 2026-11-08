import threading

from langchain_neo4j import Neo4jGraph

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

_graph: Neo4jGraph | None = None
_lock = threading.Lock()


def get_graph() -> Neo4jGraph:
    """Conexão Neo4j única e compartilhada (thread-safe)."""
    global _graph
    if _graph is None:
        with _lock:
            if _graph is None:
                _graph = Neo4jGraph(
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    refresh_schema=False,  # não usamos o schema; evita custo/APOC na inicialização
                )
    return _graph