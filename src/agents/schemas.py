from typing import Literal

from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    intent: Literal["qa", "sintese", "plano", "fora_dominio"] = Field(
        ...,
        description="A intenção do usuário: 'qa', 'sintese', 'plano' ou 'fora_dominio'",
    )
    complexity: Literal["simples", "complexa"] = Field(
        default="simples",
        description="'simples' (1-2 buscas diretas) ou 'complexa' (várias etapas/aspectos independentes).",
    )
    research_directive: str = Field(
        ...,
        min_length=1,
        description="Instrução clara e direta dizendo ao pesquisador O QUE ele deve buscar na base de dados.",
    )


class TaskDecomposition(BaseModel):
    """Saída do decompositor: passos de PESQUISA independentes e autocontidos."""

    steps: list[str] = Field(
        ...,
        min_length=1,
        description="Passos de pesquisa, cada um uma busca autocontida na base do WIT.",
    )

class PlanStep(BaseModel):
    etapa: str = Field(..., description="Nome curto da etapa")
    descricao: str = Field(..., description="2-3 frases: o que fazer, como, com quem e em quanto tempo")
    fonte: str | None = Field(None, description="Título do artigo que embasa a etapa; null se for proposta sua")

class ProjectPlan(BaseModel):
    title: str = Field(..., description="Título do projeto proposto")
    target_audience: str = Field(..., description="Público alvo do projeto")
    justification: str = Field(..., description="4-6 frases citando artigos pelo título")
    methodology: list[PlanStep] = Field(..., min_length=1)
    challenges: list[str] = Field(..., min_length=1, description="Desafios prováveis")
    indicators: list[str] = Field(..., min_length=1, description="Indicadores de sucesso mensuráveis")
    expected_results: list[str] = Field(..., min_length=1, description="Resultados esperados")
    references: list[str] = Field(..., description="Artigos do WIT citados (nós do grafo)")


class Review(BaseModel):
    """Usado pelo verificador para decidir o próximo passo do fluxo."""

    verdict: Literal["aprovado", "reescrever", "pesquisar_mais", "reclassificar"] = (
        Field(
            ...,
            description=(
                "'aprovado'; 'reescrever' (evidências bastam, texto erra); "
                "'pesquisar_mais' (faltam evidências); 'reclassificar' (intenção/escopo errados)."
            ),
        )
    )
    feedback: str | None = Field(
        default=None,
        description="Se não aprovado, explicar exatamente o que corrigir.",
    )
    missing_steps: list[str] = Field(
        default_factory=list,
        description="Só para 'pesquisar_mais': buscas específicas que faltam.",
    )
