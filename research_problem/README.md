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

Dois workflows com agentes separados por tema, em 2026-10-31. Cada artigo foi lido por **um agente
dedicado**, quase sempre em texto completo, que preencheu uma ficha. Um agente de síntese por tema
escreveu o `README.md` só a partir das fichas.

- **Primeira passada:** 3 agentes de busca por tema (30 e 29 candidatos), 15 artigos lidos por tema.
- **Segunda passada:** os artigos cortados na primeira (FedPer, FedRep, FedBN, Ditto, FedBone,
  Kendall, CAGrad, Nash-MTL, Aligned-MTL, FAMO, Zhao et al.; DARE, AdaMerging, Fisher merging,
  model soups, Git Re-Basin, FedLAW, FedLWS, Gradient Masked Averaging, OGD), mais até 6 trabalhos
  de 2023–2026 por tema, achados por um agente de busca dirigido. Entre eles: ForkMerge,
  Selective Task Group Updates, Mueller et al. (TMLR 2025), FedGain (ICML 2026), OmniFM
  (CVPR 2026); Iso-C/Iso-CTS, WUDI-Merging, EMR-Merging, Twin-Merging, FedAWA e um survey de
  model merging. A síntese de cada tema foi reescrita para incorporar esses artigos.
- **Total:** 32 fontes em *negative transfer* e 30 em cancelamento de pesos. MAS e FedMTL ficaram
  de fora porque já estão em `research_decisions/literatura_mas_fedmtl.md`.

Citações conferidas à mão:

- Nagura et al. (2020): resumo, "diminishes the norm of the update vectors".
- Wang et al. (2019): resumo e venue CVPR 2019.
- Li et al. (2023, FedLAW): Eq. 6, a decomposição de ‖Σλᵢgᵢ‖² em normas mais termos cruzados de
  cosseno.
- Liu et al. (2026, OmniFM): título, autores e CVPR 2026.

Não conferidas à mão: a citação de Zheng et al. (2025, FedCALM), porque não havia extrator de PDF
aqui, e FedGain, que fica atrás da verificação de navegador do OpenReview e já está marcado
"[só resumo]". As fichas registram outras ressalvas de venue e de citações truncadas. Confira no
PDF antes de levar uma citação para a dissertação.

Os p-valores deste projeto continuam exploratórios
(`exploratory_only_non_independent_client_fold_pairs`); nada aqui os torna confirmatórios.
