# Métodos de verificação de conflito de gradientes — fontes verificadas

Nota da etapa E0.2.1 (`plano.md`). Para cada candidato, este documento registra:
- a definição exata **conforme a fonte primária**;
- o que muda ao levá-lo para o nosso setup: FedPer, 16 clientes, stem/cabeças personalizados, trunk
  de 4,66 M parâmetros;
- o status de verificação da venue.

Verificado em 2026-10-09 pelos textos do arXiv/ar5iv.

**Legenda de status**
- **verificado**: fórmula e autoria lidas no texto da fonte primária;
- **venue não confirmada**: a venue não aparece no texto primário e só foi vista em fontes
  secundárias (citações, repositórios). Confirmar antes de citar.

## Contexto que condiciona a escolha (E0.1/E0.1b)

- A métrica atual compara **deltas de rodada** Δᵢ = θᵢ − θ (trunk devolvido − trunk enviado). Ela é
  válida, mas é de primeira ordem e só mede geometria.
- Com a seed por cliente, os updates são ortogonais até entre clientes com os mesmos dados (H3).
- Entre datasets, o cosseno é ≈ 0 por construção, já que cada dataset tem um stem diferente.
- A pergunta científica é **transferência negativa**: o efeito na loss ou no desempenho, e não apenas
  a direção do update.

---

## M1 — Cosseno e similaridade de magnitude entre gradientes (PCGrad)

- **Fonte:** Yu, Kumar, Gupta, Levine, Hausman, Finn. *Gradient Surgery for Multi-Task Learning.*
  NeurIPS 2020. arXiv:2001.06782. **Status: verificado.**
- **Definição 1 (conflito):** os gradientes gᵢ, gⱼ estão em conflito quando `cos φᵢⱼ < 0`.
- **Definição 2 (similaridade de magnitude):** `Φ(gᵢ, gⱼ) = 2‖gᵢ‖‖gⱼ‖ / (‖gᵢ‖² + ‖gⱼ‖²)`. Vale 1
  para magnitudes iguais e tende a 0 quando elas divergem.
- **"Tragic triad":** a dificuldade de otimização surge quando coexistem três condições: conflito,
  grande diferença de magnitude (uma tarefa domina) e alta curvatura.
- **Como os gradientes são obtidos:** por tarefa, nos **mesmos parâmetros compartilhados θ**, cada um
  sobre o sub-batch da sua tarefa (Algoritmo 1: `gₖ = ∇θ Lₖ(θ)`).
- **No nosso setup:**
  - "Tarefa" passa a ser "cliente".
  - O gradiente é calculado no trunk recebido, **antes** do primeiro passo local, sobre batches-sonda
    fixos. Isso é exatamente a condição `grad10` do `scripts/gradient_conflict_controls.py`.
  - Elimina o Adam e o acúmulo de steps (H1/H2).
  - A magnitude Φ é informação nova: a métrica atual normaliza a escala e não vê quando uma tarefa
    domina. Isso é relevante para seg com peso 4 e cls com peso 1.
- **Limite:** continua sendo geometria de primeira ordem. Entre datasets, o valor ≈ 0 por construção
  se mantém.

## M2 — Afinidade lookahead entre tarefas (TAG)

- **Fonte:** Fifty, Amid, Zhao, Yu, Anil, Finn. *Efficiently Identifying Task Groupings for Multi-Task
  Learning.* NeurIPS 2021 (spotlight). arXiv:2109.04617. **Status: verificado.**
- **Eq. 1:** `Z^t_{i→j} = 1 − L_j(X^t, θ^{t+1}_{s|i}, θ^t_j) / L_j(X^t, θ^t_s, θ^t_j)`, com
  `θ^{t+1}_{s|i} = θ^t_s − η ∇_{θ_s} L_i(X^t, θ^t_s, θ^t_i)`:
  - é **um passo de SGD** sobre os parâmetros compartilhados, com os parâmetros específicos da tarefa
    e o batch mantidos fixos;
  - o mesmo batch Xᵗ é usado no passo e nas duas avaliações.
- **Interpretação** (palavras da fonte): Z > 0 significa que o update compartilhado reduz a loss de j;
  Z < 0 significa que o update é "antagonistic". **É a definição funcional de transferência
  negativa.**
- **Agregação no tempo:** média sobre todos os passos, a cada n passos ou num trecho contíguo:
  `Ẑ = (1/T) Σₜ Zᵗ`. Calcular a cada 10 passos é apresentado como ganho de eficiência.
- **Treino vs validação** (Apêndice B.4): para i ≠ j, calcular no conjunto de treino aproxima a
  validação (Pearson 0.98). A aproximação falha para i = j.
- **No nosso setup:**
  - i e j são clientes. Lⱼ é avaliada com o stem/cabeças **de j** e o batch-sonda **de j**.
  - O passo de i pode ser:
    - (a) o passo de SGD do TAG, `−η gᵢ`, que reaproveita os gradientes do M1;
    - (b) o delta real da rodada Δᵢ, uma adaptação ao FL próxima do FedFomo (ver M2').
  - **Atenção:** com η pequeno, Z ≈ η gⱼ·gᵢ / Lⱼ. Ou seja, o TAG reduz-se a um **produto interno**,
    que inclui magnitude, ao contrário do cosseno. A contribuição própria do TAG (efeitos de ordem
    superior) só aparece com passos do tamanho real do update. Por isso, a variante (b) é a mais
    informativa para nós.
  - Custo: n² = 256 avaliações (só forward) por rodada amostrada. Cada cliente j precisa receber os
    16 Δᵢ, que na simulação podem ser distribuídos via disco.

## M2' — Ganho da federação por cliente (adaptação; precedente: FedFomo)

- **Precedente:** Zhang, Sapra, Fidler, Yeung, Alvarez. *Personalized Federated Learning with First
  Order Model Optimization* (FedFomo). arXiv:2012.08565. **Status:** fórmula verificada; **venue não
  confirmada** (ICLR 2021 segundo fontes secundárias).
- **Eq. 3 do FedFomo:** `w_n = (L_i(θ_i^{t−1}) − L_i(θ_n^t)) / ‖θ_n^t − θ_i^{t−1}‖`, calculada no
  **split de validação** do cliente i. Mede quanto o modelo de outro cliente reduz a loss do cliente i.
- **Nossa adaptação (não é método publicado; declarar como tal):**
  `G_j = L_j(θ + Δ_j) − L_j(θ_agg)`, na validação de j, com os parâmetros personalizados de j.
  - G_j > 0 indica que a média federada foi melhor para j do que o próprio passo de j.
  - G_j < 0 indica que a federação atrapalhou j nessa rodada. Isso inclui a **diluição** (12,5% de
    peso), que nem o cosseno nem o M1 enxergam.
  - Custo: 2 forwards por cliente por rodada amostrada; nenhum dado extra a distribuir.
- **Observação:** é a versão por rodada da comparação federado × local-only, e responde diretamente à
  pergunta sobre transferência negativa.

## M3 — Pureza de sinal do gradiente (GradDrop)

- **Fonte:** Chen, Ngiam, Huang, Luong, Kretzschmar, Chai, Anguelov. *Just Pick a Sign: Optimizing
  Deep Multitask Models with Gradient Sign Dropout.* NeurIPS 2020. arXiv:2010.06808.
  **Status: verificado.**
- **Eq. 1:** `P = ½ (1 + Σᵢ ∇Lᵢ / Σᵢ |∇Lᵢ|)`, calculada **por elemento**, somando sobre as tarefas i.
  Com batches separados por tarefa, cada Gᵢ é somado sobre o batch antes.
- **Interpretação:** P = 1 se todos os gradientes são positivos e P = 0 se todos são negativos.
  P ≈ 0.5 indica magnitudes positivas e negativas equilibradas. A leitura "0.5 = conflito" é
  **inferência nossa**; a fonte não a enuncia.
- **No nosso setup:** grátis sobre os gradientes do M1. Dá uma distribuição por coordenada/canal, que
  capta conflito localizado que o cosseno global dilui (H5).

## M4 — Similaridade por grupo de parâmetros com EMA (GradVac)

- **Fonte:** Wang, Tsvetkov, Firat, Cao. *Gradient Vaccine: Investigating and Improving Multi-task
  Optimization in Massively Multilingual Models.* arXiv:2010.05874. **Status:** fórmula verificada;
  **venue não confirmada** (ICLR 2021 segundo o registro NSF-PAR; o texto primário não declara).
- **Medida:** cosseno entre gradientes de tarefas, `φᵢⱼ = gᵢ·gⱼ / (‖gᵢ‖‖gⱼ‖)`, **por grupo de
  parâmetros k**, com alvo suavizado por EMA: `φ̂ᵢⱼₖ⁽ᵗ⁾ = (1−β) φ̂ᵢⱼₖ⁽ᵗ⁻¹⁾ + β φᵢⱼₖ⁽ᵗ⁾`, com
  `φ̂⁽⁰⁾ = 0`.
- **No nosso setup:** a métrica atual já é por bloco. O que o GradVac acrescenta é só a suavização
  temporal (reduz ruído), que pode ser aplicada na análise sem coleta nova. **Baixo ganho marginal.**

## M5 — O que a literatura de FL multitarefa mede

- **FedBone:** Chen, Zhang, Jiang, Chen, Gao, Huang. *FedBone: Towards Large-Scale Federated
  Multi-Task Learning.* arXiv:2306.17465. **Status:** fórmulas verificadas; **sem venue
  identificada**, citar como preprint.
  - Conflito: `cos ωᵢⱼ < 0` entre gradientes de clientes, no servidor (Definição 1), o mesmo critério
    do PCGrad.
  - GPAggregation: projeta `∇ᵢ' = ∇ᵢ − (∇ᵢ·∇ⱼ / ‖∇ⱼ‖²) ∇ⱼ` quando `∇ᵢ·∇ⱼ < 0`, após reescala por
    atenção.
  - Não dá limiar nem analisa a frequência de conflito.
- **FedHCA²:** Lu, Huang, Yang, Sirejiding, Ding, Lu. *FedHCA²: Towards Hetero-Client Federated
  Multi-Task Learning.* arXiv:2311.13250. **Status:** fórmulas verificadas; **venue não confirmada no
  texto primário** (CVPR 2024, pp. 5599–5609, segundo citações).
  - Usa **deltas de rodada** `Δθᵢ = θᵢ − θᵢ^{(r−1)}` do encoder, como a nossa métrica atual, e
    reconhece que é uma aproximação por cobrir várias épocas locais.
  - Agregação conflict-averse ao estilo do CAGrad: `max_Ũ min_i ⟨Δθᵢ, Ũ⟩` sujeito a
    `‖Ũ − Δθ̄‖ ≤ c‖Δθ̄‖` (Eq. 6).
- **Consequência para o artigo:** a métrica atual (cosseno entre deltas de rodada) tem precedente
  direto no FedHCA², e o critério `cos < 0` vem do PCGrad/FedBone. O que **nenhuma** dessas fontes faz
  é validar a métrica com controles, nem separar "sem conflito" de "sem interação". Esse é um ponto
  defensável do nosso trabalho.

## M6 — CKA (similaridade de representações)

- **Fonte:** Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations
  Revisited.* ICML 2019. arXiv:1905.00414. **Status: verificado.**
- X e Y são ativações (com colunas centradas) **para os mesmos n exemplos**; o CKA linear é
  `‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F)`.
- **No nosso setup:** exige as mesmas imagens, portanto **só vale dentro de um dataset**, comparando
  as features do trunk de clientes diferentes sobre as mesmas imagens. Não responde à pergunta entre
  datasets. Fica como complemento opcional.

---

## Correções em relação à tabela inicial do `plano.md`

- TAG: o título correto é "...Task Groupings **for** Multi-Task Learning", e não "in".
- GradVac, FedFomo e FedHCA²: venues **não confirmadas no texto primário**. Não citar as venues sem
  checar os proceedings.
- FedBone: preprint, sem venue.
- M2 com η pequeno reduz-se a um produto interno de primeira ordem. Usar o delta real (variante b)
  para captar efeitos além do M1.

## Leitura preliminar para a matriz de decisão (a confirmar na prototipagem E0.2.2–E0.2.3)

| | C1 transf. negativa | C2 indep. otimizador | C4 custo | C5 fonte | C6 complementar |
|---|---|---|---|---|---|
| M1 (cos + Φ do gradiente) | proxy de 1ª ordem | sim | ~2× a rodada amostrada | verificado | sim (magnitude) |
| M2 (TAG com Δ real) | **sim** (funcional) | parcial (Δ vem do otimizador real) | 256 forwards por rodada amostrada | verificado (adaptação ao FL) | sim |
| M2' (ganho da federação) | **sim**, inclui diluição | parcial | 2 forwards por cliente | precedente FedFomo; adaptação nossa | sim |
| M3 (pureza de sinal) | proxy local | sim | grátis sobre M1 | verificado | parcial |
| M4 (GradVac EMA) | não | — | grátis | venue não confirmada | baixo |
| M6 (CKA) | não | sim | médio | verificado | só intra-dataset |

## BibTeX

```bibtex
@inproceedings{yu2020gradient,
  title     = {Gradient Surgery for Multi-Task Learning},
  author    = {Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  note      = {arXiv:2001.06782}
}
@inproceedings{fifty2021efficiently,
  title     = {Efficiently Identifying Task Groupings for Multi-Task Learning},
  author    = {Fifty, Christopher and Amid, Ehsan and Zhao, Zhe and Yu, Tianhe and Anil, Rohan and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
  note      = {arXiv:2109.04617}
}
@inproceedings{chen2020just,
  title     = {Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout},
  author    = {Chen, Zhao and Ngiam, Jiquan and Huang, Yanping and Luong, Thang and Kretzschmar, Henrik and Chai, Yuning and Anguelov, Dragomir},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020},
  note      = {arXiv:2010.06808}
}
@misc{wang2020gradient,
  title         = {Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models},
  author        = {Wang, Zirui and Tsvetkov, Yulia and Firat, Orhan and Cao, Yuan},
  year          = {2020},
  eprint        = {2010.05874},
  archivePrefix = {arXiv},
  note          = {Venue (ICLR 2021) not confirmed in the primary text}
}
@misc{zhang2020personalized,
  title         = {Personalized Federated Learning with First Order Model Optimization},
  author        = {Zhang, Michael and Sapra, Karan and Fidler, Sanja and Yeung, Serena and Alvarez, Jose M.},
  year          = {2020},
  eprint        = {2012.08565},
  archivePrefix = {arXiv},
  note          = {Venue (ICLR 2021) not confirmed in the primary text}
}
@misc{chen2023fedbone,
  title         = {FedBone: Towards Large-Scale Federated Multi-Task Learning},
  author        = {Chen, Yiqiang and Zhang, Teng and Jiang, Xinlong and Chen, Qian and Gao, Chenlong and Huang, Wuliang},
  year          = {2023},
  eprint        = {2306.17465},
  archivePrefix = {arXiv}
}
@misc{lu2023fedhca2,
  title         = {FedHCA$^2$: Towards Hetero-Client Federated Multi-Task Learning},
  author        = {Lu, Yuxiang and Huang, Suizhi and Yang, Yuwen and Sirejiding, Shalayiding and Ding, Yue and Lu, Hongtao},
  year          = {2023},
  eprint        = {2311.13250},
  archivePrefix = {arXiv},
  note          = {Cited elsewhere as CVPR 2024, pp. 5599--5609; confirm in the CVF proceedings}
}
@inproceedings{kornblith2019similarity,
  title     = {Similarity of Neural Network Representations Revisited},
  author    = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {International Conference on Machine Learning},
  year      = {2019},
  note      = {arXiv:1905.00414}
}
```
