# Cancelamento de pesos ("cancelamento de pesos") — síntese da literatura

> Escopo: síntese feita **apenas** a partir das 15 avaliações de artigos fornecidas. Todas foram lidas em texto completo (`full_text`), por isso nenhuma afirmação leva a marca [só resumo]. Alguns pontos não foram verificados e aparecem sinalizados no texto:
> - a fórmula da "conflict rate" de Zheng et al. (2025) está num apêndice que não foi lido;
> - as citações de Nagura et al. (2020) vêm de uma versão HTML e são quase literais;
> - o venue de Zhang et al. (2023) não foi confirmado;
> - a publicação de Sattler et al. (2019) na TNNLS foi citada de memória.

## 1. Resumo em 5 linhas

1. "Cancelamento de pesos" não é um termo padrão. A formalização mais próxima da razão do projeto, 1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖, é o par N/E de Nagura et al. (2020), que chamam o fenômeno de "weight updating direction divergence (WUDD)".
2. A literatura divide o fenômeno em famílias:
   - **conflito direcional**, com cosseno < 0 (Yu et al., 2020; Wang et al., 2021b; Zhang et al., 2023);
   - **viés de otimização**, sob os nomes client drift e objective inconsistency (Karimireddy et al., 2020; Wang et al., 2020);
   - **interferência paramétrica** em model merging (Yadav et al., 2023; Wang et al., 2024; Gargiulo et al., 2025).
3. Zheng et al. (2025) dão a âncora formal para separar conflito de diluição. O gap ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖² é negativo exatamente quando a razão do projeto supera a sua referência ortogonal.
4. Atualizações quase ortogonais não "cancelam" no sentido de conflito, mas encolhem o passo agregado. Esse caso é o que FedExP trata com extrapolação (Jhunjhunwala et al., 2023) e o que métodos de projeção (PCGrad, FedGH, FedFV) não conseguem corrigir.
5. Baixo cancelamento não garante agregação inócua. Há degradação mesmo com suportes disjuntos (Wang et al., 2024), e a não interferência funcional depende de pré-treino (Ortiz-Jimenez et al., 2023).

## 2. Como a literatura define o termo

### 2.1 Cronologia

**2019 — Sattler et al.: "incongruence".**
- O artigo define clientes congruentes e incongruentes a partir da existência de um ótimo local comum: *"we will call a set of clients and their data generating distributions ϕ congruent (with respect to f and l) if they satisfy Assumption 1 and incongruent if they don't."* (Sec. I, p.1).
- No estacionário do FL, os gradientes ponderados se anulam enquanto cada um continua não nulo: *"Federated Learning converges to a stationary point of the FL objective where the gradients of the two clients are of positive norm and point into opposite directions."* (Fig. 2, p.3).
- O diagnóstico usa dois limiares separados, sem razão normalizada: norma do update do servidor < ε₁ e norma máxima de cliente > ε₂ (Eqs. 32–33). Usa também o cosseno entre weight-updates (Eq. 44) (Sattler et al., 2019).

**2020 — Nagura et al.: WUDD. É a definição mais próxima do "cancelamento de pesos".**
- *"averaging models from different clients significantly diminishes the norm of the update vectors, resulting in slow learning rate and low prediction accuracy."* (Abstract).
- *"the local models tend to cancel each other out, resulting in a extremely small updating vector on the server [...] We refer to this phenomenon as the weight updating direction divergence (WUDD) problem."* (Sec. 3).
- Formalmente, N = ‖Σ(n_k/n)Δw^k‖ (Eq. 4) e E = Σ(n_k/n)‖Δw^k‖ (Eq. 5), com N ≤ E (Eq. 6). **A razão do projeto é exatamente 1 − N/E** (Nagura et al., 2020).
- Ressalva: é um preprint arXiv, e as citações são quase literais (versão HTML).

**2020 — Yu et al.: "conflicting gradients" (PCGrad).**
- Definição: *"We define φ_ij as the angle between two task gradients g_i and g_j. We define the gradients as conflicting when cos φ_ij < 0."* (Sec. 2.2, Def. 1).
- Ressalva central dos próprios autores: *"conflicting gradients are not detrimental on their own. Indeed, simply averaging task gradients should provide the correct solution to descend the multi-task objective."* (Sec. 2.2).
- O dano exige a "tragic triad": conflito, diferença de magnitude (Φ, Def. 2) e alta curvatura (Def. 3) (Yu et al., 2020).

**2020 — Karimireddy et al.: "client-drift".**
- *"suffers from 'client-drift' when the data is heterogeneous (non-iid), resulting in unstable and slow convergence."* (Abstract).
- *"Even if all the clients start from the same point x, each of the y_i will move towards their client optimum x_i*."* (Sec. 3.2).
- Não há métrica escalar. A heterogeneidade entra pelas constantes (G, B) e δ (A1, A2) (Karimireddy et al., 2020).

**2020 — Wang et al.: "objective inconsistency".**
- *"Naive weighted aggregation of such models causes objective inconsistency, that is, the global model converges to a stationary point of a mismatched objective function which can be arbitrarily different from the true objective."* (Abstract).
- O artigo separa direção (wᵢ, dᵢ) de comprimento do passo (τ_eff). Chama de "slowdown" a razão τ̄/τ_eff (Sec. 4.1) (Wang et al., 2020).

**2021 — Wang et al. (GradVac): similaridade contínua.**
- *"two tasks have a positive gradient similarity of cos(θ)=ϕij"* (Sec. 3.2).
- Sobre o PCGrad: ele *"implicitly makes the assumption that any two tasks must have the same gradient similarity objective of zero."* (Sec. 3.1).
- Usa "negative interference" de forma informal (Introdução) (Wang et al., 2021a).

**2021 — Wang et al. (FedFV): conflito entre clientes.**
- *"Client i's gradient g_i ∈ R^d conflicts with client j's gradient g_j ∈ R^d iff g_i · g_j < 0."* (Sec. 3, Def. 1).
- O dano vem do conflito somado à disparidade de magnitude: *"when the conflicting gradients have large differences in their magnitudes, model accuracy can suffer significant reductions for some of the clients"* (Introdução) (Wang et al., 2021b).

**2023 — Ilharco et al.: task vectors e quase-ortogonalidade.**
- *"vectors from different tasks are typically close to orthogonal, and speculate that this enables the combination of task vectors via addition with minimal interference."* (Sec. 6, p.7).
- A ligação entre ortogonalidade e baixa interferência é declarada como especulação (Ilharco et al., 2023).

**2023 — Yadav et al. (TIES): "interference" com duas fontes.**
- As duas fontes nomeadas são *"Interference from redundant parameters"* e *"Interference from sign disagreement"* (Sec. 1).
- *"simply aggregating the values lead to interference that shrinks the parameter's value in the merged model."* (Sec. 1). Esta é a descrição coordenada a coordenada mais próxima de "cancelamento" (Yadav et al., 2023).

**2023 — Ortiz-Jimenez et al.: "weight disentanglement", uma definição funcional.**
- *"the function f can be decomposed as a sum of spatially-localized components, i.e., vanishing outside a spatial region, whose functional variation is entirely captured by each τ_t."* (Sec. 4, Property 3).
- Ressalva: o artigo **não** discute cosseno nem ortogonalidade de task vectors (Ortiz-Jimenez et al., 2023).

**2023 — Jhunjhunwala et al. (FedExP): pseudo-gradientes e passo do servidor.**
- O termo "cancelamento" não aparece. O fenômeno é descrito pela razão Σ‖Δᵢ‖² / (2M‖Δ̄‖²).
- *"we can have ‖Δ̄‖² ≈ 0 but ‖Δ_i‖² ≠ 0"* (Sec. 3.3).
- Liga essa razão a "gradient diversity" (Jhunjhunwala et al., 2023).

**2023 — Zhang et al. (FedGH): "gradient conflict" entre clientes.**
- *"We identify gradient conflict when the cosine similarity between g_t^i and g_t^j, ∀i≠j, becomes negative"* (Sec. II-B).
- Usa deltas de modelo como pseudo-gradientes (Zhang et al., 2023).

**2024 — Wang et al. (TALL/Consensus): separa "information erasure" de "task interference".**
- *"Information erasure: large amount of information specific to each task is erased when merging the multi-task vector. Task interference: the task-specific information is preserved in the multi-task vector, but can not manifest properly due to interference between the tasks."* (Sec. 3).
- Conclusão: a degradação *"is not due to information erasure during merging process, but because of task interference during evaluation."* (Sec. 7) (Wang et al., 2024).

**2025 — Gargiulo et al. (TSV): "Singular Task Interference".**
- *"Given that a formal definition of interference in model merging is not yet established, we adopt an operational definition: interference is any cross-task interaction that hinders the merging process."* (App. E.2).
- STI = ‖(UᵀU − I) Σ (VᵀV − I)‖₁ (Sec. 3.3, Eq. 6).
- O artigo chama o cosseno de *"coarse-grained"* (Sec. 1) (Gargiulo et al., 2025).

**2025 — Zheng et al. (FedCALM): "server aggregation conflicts".**
- *"when ∆θ(k)i·∆θ(k)j < 0 or ∥∆θ(k)avg∥² < Σ_{i=1}^N p_i² ∥∆θ(k)i∥², it leads to an increase in global loss. Specifically, this situation can be further detailed into two main sources: insufficient average update magnitude and directional conflicts between client updates."* (Sec. 3.3, após Eq. 4, p. 15447).
- O cenário é FedRep/FedPer (Zheng et al., 2025).

**Quem cunhou o quê:**
- "Cancel each other out" ligado a uma razão de normas: Nagura et al. (2020), sob o nome WUDD.
- "Conflicting gradients" como cosseno < 0: Yu et al. (2020) em MTL. Foi transportado para FL por Wang et al. (2021b) e Zhang et al. (2023).
- "Client-drift": Karimireddy et al. (2020).
- "Objective inconsistency": Wang et al. (2020).
- "Interference" paramétrica em merging: Yadav et al. (2023), Wang et al. (2024) e Gargiulo et al. (2025).
- Nenhum dos 15 artigos usa "weight cancellation" como termo técnico.

### 2.2 Tabela comparativa de definições

| Autor | Ano | Termo | Definição (resumida) | Nível | Como mede |
|---|---|---|---|---|---|
| Sattler et al. | 2019 | incongruence | Não existe ótimo comum. No estacionário, os updates continuam não nulos e opostos. | cliente (distribuição) | cosseno entre weight-updates (Eq. 44); ‖update servidor‖ < ε₁ e max‖Δᵢ‖ > ε₂ |
| Nagura et al. | 2020 | WUDD ("cancel each other out") | A média encolhe a norma do update agregado. | cliente → servidor (modelo inteiro) | N = ‖Σwᵢ Δᵢ‖ vs E = Σwᵢ‖Δᵢ‖ (**= 1 − razão do projeto**) |
| Yu et al. | 2020 | conflicting gradients / tragic triad | cos φᵢⱼ < 0. Só é danoso com disparidade de magnitude e curvatura alta. | tarefa (gradiente por passo) | cosseno; Φ; curvatura H |
| Karimireddy et al. | 2020 | client-drift | Passos locais derivam para o ótimo de cada cliente, o que enviesa a média. | cliente (trajetória) | constantes teóricas (G, B), δ; rodadas até a acurácia-alvo |
| Wang et al. | 2020 | objective inconsistency | A agregação otimiza Σwᵢ Fᵢ ≠ Σpᵢ Fᵢ quando o trabalho local é desigual. | cliente (pesos efetivos) | χ²_{p‖w}; slowdown τ̄/τ_eff (teórico) |
| Wang et al. (GradVac) | 2021a | gradient similarity / negative interference | Similaridade contínua. Abaixo da similaridade "natural" já há prejuízo. | tarefa, por camada | cosseno por camada e por passo |
| Wang et al. (FedFV) | 2021b | conflicting gradients (internal/external) | gᵢ·gⱼ < 0 entre clientes. Danoso com disparidade de magnitude. | cliente | teste binário de sinal |
| Ilharco et al. | 2023 | task vector / interference (informal) | τ = θ_ft − θ_pre. Vetores quase ortogonais somam com pouca interferência (especulação). | tarefa (pesos) | cosseno entre task vectors; acurácia normalizada |
| Yadav et al. | 2023 | parameter interference (redundância, conflito de sinal) | Sinais opostos e zeros redundantes encolhem o valor do parâmetro fundido. | parâmetro (coordenada) | % de parâmetros com conflito de sinal |
| Ortiz-Jimenez et al. | 2023 | weight disentanglement | O update de uma tarefa não altera a saída fora do suporte dela. | função / domínio | erro de disentanglement ξ(α₁, α₂) |
| Jhunjhunwala et al. | 2023 | pseudo-gradient / gradient diversity | O update médio é muito menor que os individuais, o que justifica passo η_g > 1. | cliente → servidor | Σ‖Δᵢ‖² / (2M‖Δ̄‖²) |
| Zhang et al. | 2023 | gradient conflict | cos(gⁱ, gʲ) < 0 entre deltas de cliente. | cliente (modelo achatado) | cosseno mínimo; fração de pares negativos |
| Wang et al. (TALL) | 2024 | information erasure vs task interference | A informação sobrevive na soma, mas não se manifesta. | parâmetro / tarefa | recuperação via máscara; perfil selfish/universal |
| Gargiulo et al. | 2025 | Singular Task Interference | Sobreposição entre subespaços singulares por camada. | camada (matriz) | STI (Eq. 6) |
| Zheng et al. | 2025 | server aggregation conflicts | Termos cruzados negativos aumentam a perda global. Duas fontes: magnitude e direção. | cliente, por camada | gap ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖²; "conflict rate" (fórmula no apêndice, não lido) |

## 3. Divergências e ambiguidades entre definições

**1. "Cancelamento de pesos" não é termo padrão.** Ele corresponde a pelo menos seis termos em uso:

| Termo usado na literatura | Fonte | O que de fato descreve |
|---|---|---|
| update cancellation / WUDD | Nagura et al. (2020) | perda de norma na média |
| client drift | Karimireddy et al. (2020) | viés de trajetória, não oposição |
| objective inconsistency | Wang et al. (2020) | viés de pesos efetivos, não perda de norma |
| gradient conflict | Yu et al. (2020); Wang et al. (2021b); Zhang et al. (2023) | sinal do cosseno |
| parameter interference | Wang et al. (2024); Gargiulo et al. (2025) | efeito funcional ou subespacial |
| sign conflict | Yadav et al. (2023) | por coordenada |

**2. Direção e magnitude.**
- Os critérios de conflito são binários e ignoram a magnitude da perda de norma (Yu et al., 2020; Wang et al., 2021b; Zhang et al., 2023).
- Nagura et al. (2020) medem só a magnitude agregada e não separam conflito de ortogonalidade.
- Zheng et al. (2025) nomeiam as duas fontes, mas na Eq. 4 a parte de magnitude aparece como subproduto dos termos cruzados. Updates ortogonais dão gap = 0 e, portanto, são tratados como inócuos.

**3. A ortogonalidade é ambígua.**
- Para Ilharco et al. (2023), ela permite "minimal interference" (especulação).
- Para Jhunjhunwala et al. (2023), é justamente o caso que pede extrapolação (η_g ≈ M/2).
- Para Ortiz-Jimenez et al. (2023), ortogonalidade geométrica não equivale a não interferência funcional (inferência a partir da definição funcional; o artigo não discute cosseno).
- Wang et al. (2024) mostram degradação com suportes disjuntos, o que dá cancelamento zero por construção.

**4. Nível de análise.** As medidas existem em cinco níveis: coordenada (Yadav et al., 2023), camada ou matriz (Gargiulo et al., 2025; Zheng et al., 2025), modelo achatado (Zhang et al., 2023; Nagura et al., 2020), função (Ortiz-Jimenez et al., 2023) e distribuição (Sattler et al., 2019). Esses níveis podem discordar entre si. Muitos conflitos de sinal por coordenada podem coexistir com baixo cancelamento em norma (Yadav et al., 2023).

**5. Objeto medido.**
- Gradientes por passo em modelo centralizado (Yu et al., 2020; Wang et al., 2021a).
- Deltas de vários passos locais (Zhang et al., 2023; Sattler et al., 2019).
- Task vectors de fine-tuning completo e merge único (Ilharco et al., 2023; Yadav et al., 2023).
- A transferência entre esses regimes não é automática.

**Qual termo casa com a razão do projeto, CR = 1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖:**
- **A métrica bruta é exatamente o 1 − N/E de Nagura et al. (2020)** ("weight updating direction divergence"). Esse é o termo a citar como definição.
- **O excesso sobre a referência ortogonal** do projeto, 1 − √(Σwᵢ²‖Δᵢ‖²) / Σwᵢ‖Δᵢ‖, tem o mesmo sinal que o "server aggregation conflict" de Zheng et al. (2025). Vale CR > referência ⇔ G = ‖Σwᵢ Δᵢ‖² − Σwᵢ²‖Δᵢ‖² < 0. Essa equivalência é uma derivação, não está no artigo, e a mesma ressalva vale para todas as derivações marcadas abaixo.
- Recomendação de nomenclatura:
  - "cancelamento", ou perda de norma da agregação (Nagura et al., 2020), para CR;
  - "conflito", na acepção de Zheng et al. (2025) e Yu et al. (2020), para o excesso sobre a referência ortogonal;
  - "diluição de passo" para a perda que existe mesmo sem conflito, ligada a "gradient diversity" e ao passo do servidor (Jhunjhunwala et al., 2023) e ao "slowdown" (Wang et al., 2020).

## 4. Como é medido (métricas)

| Métrica | Fórmula / descrição | Fonte | Observação |
|---|---|---|---|
| Razão de normas N/E | ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖ | Nagura et al. (2020) | 1 − N/E = CR do projeto. Sem referência ortogonal. |
| Gap quadrático de agregação | ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖² = Σ_{i≠j} pᵢpⱼ Δᵢ·Δⱼ | Zheng et al. (2025) | negativo ⇔ conflito. Por camada. |
| Razão FedExP | Σ‖Δᵢ‖² / (2M‖Δ̄‖²) | Jhunjhunwala et al. (2023) | versão quadrática da diluição. Pesos uniformes. |
| Cosseno par a par | Δᵢ·Δⱼ / (‖Δᵢ‖‖Δⱼ‖) | Sattler et al. (2019); Zhang et al. (2023); Wang et al. (2021a) | por camada (Wang et al., 2021a) ou modelo inteiro |
| Cosseno mínimo / fração de pares negativos | min cos; #{cos < 0} / #pares | Zhang et al. (2023) | valores relatados: −0,06 a −0,22 (rótulos não-IID) |
| Separation gap | g(α) = α_intra^min − α_cross^max | Sattler et al. (2019) | separa grupos de clientes |
| Similaridade de magnitude | Φ = 2‖gᵢ‖‖gⱼ‖ / (‖gᵢ‖² + ‖gⱼ‖²) | Yu et al. (2020) | condição da tragic triad |
| Conflito de sinal | % de coordenadas com sinais divergentes após trim top-k | Yadav et al. (2023) | chega a ~80% com pouco trim |
| Perfil de pesos | catastrophic / selfish / universal pela máscara m_t = 1{\|τ_t\| ≥ \|τ_MTL − τ_t\|} | Wang et al. (2024) | "posse" coordenada a coordenada |
| STI | ‖(UᵀU − I) Σ (VᵀV − I)‖₁ por camada | Gargiulo et al. (2025) | mede sobreposição de subespaço, não cancelamento realizado |
| Erro de disentanglement | ξ = Σ_t E[1(f(θ₀+α_tτ_t) ≠ f(θ₀+α₁τ₁+α₂τ₂))] | Ortiz-Jimenez et al. (2023) | sonda funcional |
| χ²_{p‖w}, slowdown | Σ(pᵢ − wᵢ)² / wᵢ; τ̄/τ_eff | Wang et al. (2020) | apenas teórico |

## 5. Estado da arte de mitigação (ênfase 2023–2026)

| Método | Ano | Ideia | Onde se aplica | Evidência |
|---|---|---|---|---|
| FedCALM (Zheng et al.) | 2025 | Por camada: projeção PCGrad entre clientes, depois uma QP que garante dᵀΔᵢ ≥ ε‖Δᵢ‖ para todo cliente | FL personalizado (FedRep/FedPer) | +1,6 a +9,9 pp em Flowers102 com ResNets e ganhos como plug-in. Pior que FedPAC com CNN rasa. 3 seeds, sem variância. d* pode ter norma maior que o passo FedAvg, sem controle de passo do servidor. |
| TSV-Merge (Gargiulo et al.) | 2025 | Truncar cada task matrix ao posto 1/T e ortogonalizar (Procrustes) os vetores singulares antes de somar | model merging | ViT-B/32 com 8 tarefas: 85,86 vs 70,79 (TA). A ortogonalização sozinha dá +3,6 a +7,4 pp. |
| TALL masks / Consensus (Wang et al.) | 2024 | Máscara por tarefa onde \|τ_t\| ≥ \|τ_MTL − τ_t\|; remover pesos selfish e catastrophic | model merging, compressão | >99% da acurácia com máscaras. Consensus TA: +4,9 pp (ViT-L/14, 20 tarefas). Ganho quase nulo com 8 tarefas. |
| FedExP (Jhunjhunwala et al.) | 2023 | Passo adaptativo do servidor η_g = max{1, Σ‖Δᵢ‖² / (2M(‖Δ̄‖² + ε))} | FL (modelo global) | 1,4–2,1× menos rodadas até a acurácia-alvo (EMNIST, CIFAR, CINIC). Teoria supõe minimizador comum. |
| TIES-Merging (Yadav et al.) | 2023 | Trim, eleição de sinal e média disjunta | model merging | +1,7 pp (visão) e +2,3 pp (NLP) sobre o melhor baseline. Com sinal oráculo: 72,0 vs 73,1 (multitask). |
| Tangent-space FT (Ortiz-Jimenez et al.) | 2023 | Fine-tuning linearizado para aumentar o disentanglement | merging / edição | Adição: +3,4 a +5,8 pp. Não funciona a partir de inicialização aleatória. |
| FedGH (Zhang et al.) | 2023 | PCGrad no servidor entre deltas com cos < 0 | FL | +0,5 a +5,2 pp. Benefício ~nulo com K = 5. Uma seed. |
| FedFV (Wang et al.) | 2021b | Projeção ordenada pela perda, conflitos externos, renormalização para a norma da média simples | FL (justiça) | CIFAR-10: pior 5% sobe de 19,84 para 32,24. A renormalização preserva o passo do FedAvg (não corrige diluição). |
| GradVac (Wang et al.) | 2021a | Alinhar cada par a uma similaridade-alvo EMA por camada | MTL (NMT) | +0,4 a +1,4 BLEU. Execução única. |
| FedNNNN (Nagura et al.) | 2020 | Reescalar o update agregado por β·E/N, com momentum | FL | Até +5,4 pp. Benchmarks de brinquedo, execução única, preprint. |
| FedNova (Wang et al.) | 2020 | Normalizar Δᵢ por trabalho local; τ_eff escolhido à parte | FL | +6 a +9 pp em VGG-11/CIFAR-10. Não trata desacordo direcional. |
| SCAFFOLD (Karimireddy et al.) | 2020 | Variáveis de controle (c − cᵢ) e passo global η_g | FL | Teoria robusta à heterogeneidade. EMNIST: o mais rápido. |
| PCGrad (Yu et al.) | 2020 | Projeção no plano normal quando cos < 0 | MTL | NYUv2 e Meta-World. Os autores o veem como regularizador em MTL supervisionado (App. B). |
| CFL (Sattler et al.) | 2019 | Agrupar clientes pela bipartição de cosseno quando há incongruência | FL | CIFAR-10 com rótulos permutados: ~20% → ~60%. Incongruência sintética. |

Tendências no período 2023–2026:
- **Merging** migrou de heurísticas por coordenada (Yadav et al., 2023) para estrutura de subespaço por camada (Gargiulo et al., 2025).
- **FL** passou de projeções globais (Zhang et al., 2023) para mitigação por camada voltada a FL personalizado (Zheng et al., 2025).
- **Em paralelo**, cresceu a linha que trata o problema como passo do servidor, sem corrigir direção (Jhunjhunwala et al., 2023).

## 6. Críticas e resultados negativos

1. **Conflito não basta para haver dano.**
   - Os próprios proponentes do PCGrad escrevem que *"conflicting gradients are not detrimental on their own"* (Yu et al., 2020, Sec. 2.2), e FedFV repete a ideia (Wang et al., 2021b, Introdução).
   - Em MTL supervisionado, o PCGrad atua mais como regularizador do que como melhoria de otimização (Yu et al., 2020, App. B).
2. **A projeção raramente dispara com similaridades positivas ou nulas.** *"PCGrad is not effective for positive gradient similarities"* (Wang et al., 2021a).
3. **Benefício pequeno com poucos clientes.** FedGH não mostrou ganho com K = 5 (Zhang et al., 2023).
4. **Cancelamento zero ainda degrada.** Task vectors com suportes disjuntos, sem coordenadas compartilhadas, degradam sob task arithmetic (Wang et al., 2024).
5. **Sem pré-treino não há composição.** A partir de inicialização aleatória, a adição de tarefas fica no acaso (ViT-B/32: 5,5 vs 5,3) (Ortiz-Jimenez et al., 2023).
6. **Forçar ortogonalidade piora.** A ablação CosReg, que empurra todos os cossenos para 0, fica atrás do PCGrad (Yu et al., 2020).
7. **Extrapolar não corrige direção** e pode elevar o piso de erro e fazer o iterado oscilar (Jhunjhunwala et al., 2023). FedNNNN amplifica uma direção média ruidosa quando há conflito real (Nagura et al., 2020).
8. **Métodos de projeção preservam ou reduzem a norma.** FedGH e FedFV não recuperam passo perdido na média (Zhang et al., 2023; Wang et al., 2021b).
9. **Fragilidade experimental recorrente:**
   - seed única (Zhang et al., 2023; Nagura et al., 2020; Wang et al., 2021a);
   - 3 seeds sem variância (Zheng et al., 2025);
   - heterogeneidade apenas por rótulo, sem shift de domínio, em quase todos os trabalhos de FL avaliados;
   - nenhum trabalho cobre segmentação médica entre modalidades.
10. **Definições apenas operacionais.** "Interference" é declarada sem definição formal estabelecida (Gargiulo et al., 2025, App. E.2). A ligação ortogonalidade → baixa interferência é especulativa (Ilharco et al., 2023).

## 7. O que isso implica para o nosso diagnóstico

Cautela geral: as conclusões abaixo são hipóteses de trabalho. Os p-valores do projeto são exploratórios (`exploratory_only_non_independent_client_fold_pairs`), e quase toda a evidência da literatura vem de classificação com skew de rótulo, não de FedPer multi-domínio e multi-tarefa.

1. **Nomenclatura.**
   - A razão do projeto deve ser citada como o 1 − N/E de Nagura et al. (2020).
   - O "excesso sobre a referência ortogonal" é, com sinal equivalente, o conflito de agregação de Zheng et al. (2025) (derivação nossa).
   - Baixo excesso entre datasets significa pouco **conflito** no sentido da literatura. Não significa pouca **perda de norma**.
2. **Diluição é um mecanismo distinto de conflito.**
   - Com updates quase ortogonais, a componente de cada cliente no agregado é wᵢΔᵢ. O passo efetivo de cada cliente é escalado por wᵢ, sem cancelamento algum.
   - Esse regime é exatamente o que Jhunjhunwala et al. (2023) tratam com η_g ≈ M/2. É também o regime que Zheng et al. (2025) consideram inócuo (G ≈ 0), o que marca uma lacuna entre as duas linhas.
   - O achado de passo agregado ~10× menor com pouco conflito é compatível com essa leitura. Isso não demonstra que a diluição seja a causa de federado < local-only.
3. **Verificação de consistência sugerida (derivação nossa).**
   - Para updates ortogonais, ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖ ≈ 1/√N_eff com normas iguais, onde N_eff = 1/Σwᵢ².
   - Com poucos clientes efetivos, essa razão fica bem acima de 0,1. Um encolhimento de 10× do passo agregado em relação ao passo dos clientes precisa então ser decomposto em três parcelas: ortogonalidade, pesos, e diferença de τ entre modos steps/epochs (Wang et al., 2020).
   - Se o encolhimento observado superar a referência ortogonal, há conflito residual, possivelmente intra-dataset, que precisa ser localizado.
4. **O cosseno par a par tem padrão na literatura.** Podem ser reportados a partir dos logs existentes:
   - fração de pares com cos < 0 e cosseno mínimo, separados em intra e inter (Zhang et al., 2023);
   - separation gap (Sattler et al., 2019);
   - leitura por bloco e por rodada (Wang et al., 2021a; Zheng et al., 2025);
   - Φ por par, para checar a tragic triad (Yu et al., 2020).
5. **Remédios que devem ser testados como ablações, não assumidos como solução:**
   - Passo do servidor sobre o trunk:
     - FedExP com versão ponderada Σwᵢ‖Δᵢ‖² / ‖Σwᵢ Δᵢ‖², que é extensão nossa (Jhunjhunwala et al., 2023);
     - reescala β·E/N (Nagura et al., 2020);
     - η_g > 1 (Karimireddy et al., 2020).
   - O η_g da FedExP pode ser recalculado offline a partir de aggregation_history.json.
   - Se o reescalonamento fechar o gap com local-only, a hipótese de diluição ganha força. Se não fechar ou desestabilizar, pesam mais a heterogeneidade de domínio e a interferência funcional.
   - A premissa de minimizador comum da FedExP é duvidosa entre quatro modalidades.
   - A comparação precisa manter o mesmo modo de budget, porque as normas Δᵢ dependem dele (Wang et al., 2020).
6. **Gradient surgery provavelmente rende pouco aqui.** PCGrad, FedGH e FedFV só agem com cos < 0, não restauram norma e mostraram pouco ganho com poucos clientes (Yu et al., 2020; Zhang et al., 2023; Wang et al., 2021b). FedCALM pode ajudar, mas parte do ganho pode vir de um passo efetivamente maior, e isso exige controle (Zheng et al., 2025).
7. **Baixo cancelamento não implica ausência de interferência.**
   - Suportes disjuntos ainda degradam (Wang et al., 2024).
   - Composição funcional depende de pré-treino (Ortiz-Jimenez et al., 2023). Isso reforça o valor do braço E7 (ResNet-34 ImageNet) como teste, frente ao MTnnUNet treinado do zero.
   - Uma sonda funcional ao estilo da Eq. 6 complementaria a medida geométrica: avaliar o cliente t com o trunk global versus trunk + Δ_t apenas.
8. **Profundidade do compartilhamento.** Zheng et al. (2025) associam corpos compartilhados mais profundos a mais conflito, de forma correlacional. Isso é relevante para o braço E6 (wide sharing), sem permitir extrapolação direta.
9. **Clustering por dataset** (Sattler et al., 2019) é um baseline natural contra um trunk único para todos os datasets.

## 8. Lacunas abertas

1. Não existe métrica consolidada que **separe diluição (ortogonalidade + pesos) de conflito**. Nagura et al. (2020) não usam referência ortogonal, e Zheng et al. (2025) tratam o caso ortogonal como inócuo. A decomposição CR = referência ortogonal + excesso, como o projeto já faz, é uma contribuição possível.
2. Nenhum dos trabalhos avaliados estuda cancelamento ou conflito em **FL entre domínios de imagem médica** nem em **multi-tarefa seg+cls com FedPer**.
3. A ligação entre medidas geométricas (cosseno, CR, STI) e interferência **funcional** não está estabelecida (Ortiz-Jimenez et al., 2023; Gargiulo et al., 2025).
4. Extrapolação do passo sem minimizador comum, num regime de heterogeneidade de domínio, não tem garantia (Jhunjhunwala et al., 2023).
5. Não há versão **ponderada ou hierárquica** de FedExP nem de FedNNNN. Ambas usam pesos uniformes ou n_k/n.
6. Faltam medidas por camada da perda de norma ao longo das rodadas de FL. STI é por camada, mas em merging único (Gargiulo et al., 2025). O gap de FedCALM não é reportado empiricamente (Zheng et al., 2025).
7. A maioria das definições é para gradientes de um passo ou task vectors de merge único. Falta teoria para **deltas de vários passos em agregação iterativa** com partes personalizadas.
8. Controles ausentes na literatura: métodos de direção, como FedCALM, não são comparados com baselines de passo do servidor, então não se sabe quanto do ganho é correção de diluição.

---

Fontes: [`fontes.md`](fontes.md). Fichas de leitura por artigo: [`fichas.md`](fichas.md).
