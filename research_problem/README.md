# research_problem — o problema de pesquisa na literatura

Pasta criada em 2026-10-31, na branch `docs/research-decisions`, para responder a duas perguntas
antes de seguir com o recorte em [`research_decisions/`](../research_decisions/README.md):

1. Como a literatura **define** *negative transfer* e o que chamamos de "cancelamento de pesos"?
2. Qual é o **estado da arte** de cada um (medida e mitigação)?

Material de trabalho: nada aqui muda código, configs ou artefatos congelados.

| Pasta | Conteúdo |
|---|---|
| [`negative_transfer/`](negative_transfer/README.md) | definições (Rosenstein 2005 → Wang 2019 → survey 2023 → FL/FMTL), divergências, métricas, SOTA, críticas, implicações, lacunas |
| [`cancelamento_de_pesos/`](cancelamento_de_pesos/README.md) | mapeamento do termo para os nomes usados na literatura (WUDD, client drift, objective inconsistency, gradient conflict, parameter interference, sign conflict), métricas, SOTA em FL e model merging |

Cada pasta tem `README.md` (síntese em português, com citações), `fontes.md` (tabela de fontes) e
`fichas.md` (uma ficha de leitura por artigo, com a definição citada literalmente e a seção).

## Como os dois termos se relacionam (resumo)

- **Negative transfer é um conceito de desfecho**: o modelo que usa outra fonte de dados fica pior
  que o *mesmo algoritmo* sem ela (NTG > 0; Wang et al., 2019). Em FL, isso vira "federado pior que
  local-only" (Bao et al., 2023; Yu et al., 2020). O nosso `federado < local-only` é NT no sentido
  canônico, desde que os dois braços sejam o mesmo algoritmo (mesmo budget, mesmas seeds).
- **Cancelamento é um mecanismo candidato**, não uma definição de NT. A nossa razão
  `1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖` é exatamente o `1 − N/E` de Nagura et al. (2020, FedNNNN), que chamam o
  fenômeno de *weight updating direction divergence* (WUDD). O excesso sobre a referência
  ortogonal tem o mesmo sinal do *server aggregation conflict* de Zheng et al. (2025, FedCALM).
  Essa equivalência é derivação nossa.
- **Diluição do passo** (updates quase ortogonais encolhem o agregado sem conflito) não tem nome
  consolidado como forma de NT. Os mais próximos são *gradient diversity* e o passo de servidor
  do FedExP (Jhunjhunwala et al., 2023) e o *slowdown* do FedNova (Wang et al., 2020). Separar
  diluição de conflito, como a E0 já faz, aparece nas duas sínteses como lacuna da literatura.
- Críticas fortes (Xin et al., 2022; Kurin et al., 2022) e o FMTL-Bench (Yang et al., 2024) mostram
  que métodos só de direção (PCGrad/CAGrad) raramente vencem uma escalarização bem ajustada, e que
  em FL multitarefa a cirurgia de gradiente falha. Isso é coerente com o nosso diagnóstico de
  pouco conflito, mas não o prova.

## Como foi feito

Um workflow com agentes separados por tema (2026-10-31): 3 agentes de busca por tema
(30 e 29 candidatos), **um agente por artigo** lendo o texto completo (15 por tema, priorizados
pelos que definem o termo e pelos marcos/SOTA) e um agente de síntese por tema, restrito às fichas.
Conferimos à mão as citações centrais de Nagura et al. (2020, resumo) e Wang et al. (2019, CVPR).
A citação de Zheng et al. (2025) **não** foi conferida à mão (PDF sem extrator de texto aqui).

### Limites desta passada (o que não foi lido)

O corte em 15 artigos por tema deixou de fora, entre outros:

- **negative transfer**: FedPer, FedRep, FedBN, Ditto, FedBone, MAS (já lido em
  `research_decisions/literatura_mas_fedmtl.md`), CAGrad, Nash-MTL, Aligned-MTL, FAMO, Kendall
  (uncertainty weighting), Zhao et al. (*On learning invariant representations*).
- **cancelamento de pesos**: DARE, AdaMerging, Fisher merging, model soups, Git Re-Basin, FedLAW,
  FedLWS (*layer-wise weight shrinking*), Gradient Masked Averaging, CAGrad, GEM/A-GEM, OGD.

Por isso o SOTA de mitigação 2023–2026 em MTL (Nash-MTL, Aligned-MTL, FAMO) e em FL com passo de
servidor/encolhimento (FedLAW, FedLWS) está **sub-representado**. Uma segunda passada sobre essa
lista é o próximo passo natural antes de afirmar qual é o estado da arte.

Os p-valores deste projeto continuam exploratórios
(`exploratory_only_non_independent_client_fold_pairs`); nada aqui os torna confirmatórios.
