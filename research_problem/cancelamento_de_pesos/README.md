# Cancelamento de pesos ("cancelamento de pesos") — síntese da literatura

> Escopo: síntese feita **apenas** a partir de 30 avaliações de artigos, em duas passadas: 15 na primeira e 15 na segunda. Todas foram lidas em texto completo (`full_text`), por isso nenhuma afirmação leva a marca [só resumo]. Alguns pontos não foram verificados e aparecem sinalizados no texto:
> - a fórmula da "conflict rate" de Zheng et al. (2025) está num apêndice que não foi lido;
> - as citações de Nagura et al. (2020) vêm de uma versão HTML e são quase literais;
> - o venue de Zhang et al. (2023) não foi confirmado;
> - a publicação de Sattler et al. (2019) na TNNLS foi citada de memória;
> - o venue de Tenison et al. (2022) é inferido (o template da v2 sugere TMLR, mas o artigo não o nomeia);
> - o venue de Cheng et al. (2025) não foi confirmado (só aparece a palavra-chave do template ICML);
> - os venues de Yu et al. (2024) (ICML 2024) e de Huang et al. (2024) (NeurIPS 2024 Spotlight) vêm de conhecimento externo ou do enunciado da tarefa, não da página lida;
> - em Shi et al. (2025b), Yang et al. (2024b) e Lu et al. (2024), a ferramenta de leitura truncou as citações (~125 caracteres). Os trechos são literais, mas fragmentários, e a confiança nessas duas primeiras fichas é média;
> - a equação do loss barrier de Ainsworth et al. (2023) foi transcrita a partir da notação matemática, não caractere por caractere.
>
> Desambiguação de anos: Shi et al. (2025a) = FedLWS e Shi et al. (2025b) = FedAWA; Yang et al. (2024a) = AdaMerging e Yang et al. (2024b) = survey de model merging; Yu et al. (2020) = PCGrad e Yu et al. (2024) = DARE.

## 1. Resumo em 5 linhas

1. "Cancelamento de pesos" não é um termo padrão: nenhum dos 30 artigos o usa como termo técnico. A formalização mais próxima da razão do projeto, 1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖, é o par N/E de Nagura et al. (2020), que chamam o fenômeno de "weight updating direction divergence (WUDD)". A Eq. 6 de FedLAW (Li et al., 2023) dá a álgebra que liga essa razão aos cossenos par a par: ‖Σλᵢgᵢ‖² = Σλᵢ²‖gᵢ‖² + Σ_{i≠j} λᵢλⱼ cos(gᵢ, gⱼ)‖gᵢ‖‖gⱼ‖.
2. A literatura divide o fenômeno em famílias:
   - **conflito direcional**, com cosseno < 0 (Yu et al., 2020; Wang et al., 2021b; Zhang et al., 2023);
   - **coerência ou dispersão** dos updates, que é contínua e não binária (Li et al., 2023; Shi et al., 2025a; Tenison et al., 2022);
   - **viés de otimização**, sob os nomes client drift e objective inconsistency (Karimireddy et al., 2020; Wang et al., 2020);
   - **interferência paramétrica** em model merging (Yadav et al., 2023; Wang et al., 2024; Gargiulo et al., 2025), que em 2025 ganhou definições funcionais e subespaciais (Cheng et al., 2025; Marczak et al., 2025);
   - **barreira de perda** no caminho linear entre modelos (Ainsworth et al., 2023; Wortsman et al., 2022).
3. Zheng et al. (2025) dão a âncora formal para separar conflito de diluição. O gap ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖² é negativo exatamente quando a razão do projeto supera a sua referência ortogonal. Esse gap é o termo cruzado da Eq. 6 de Li et al. (2023); a identificação entre os dois é nossa.
4. Atualizações quase ortogonais não "cancelam" no sentido de conflito, mas encolhem o passo agregado. Em merging, task vectors quase ortogonais ainda perdem ~20 pontos sob soma uniforme, e boa parte disso volta só com reescala de coeficientes por tarefa e por camada (Yang et al., 2024a). Reescala ou restauração de magnitude é ingrediente central em DARE, EMR-Merging e FedExP (Yu et al., 2024; Huang et al., 2024; Jhunjhunwala et al., 2023). Mas a literatura de FL diverge sobre o que fazer quando os clientes discordam: há métodos que **encolhem** o passo (Li et al., 2023; Shi et al., 2025a; Tenison et al., 2022) e métodos que o **amplificam** (Jhunjhunwala et al., 2023; Nagura et al., 2020).
5. Baixo cancelamento não garante agregação inócua:
   - há degradação mesmo com suportes disjuntos (Wang et al., 2024);
   - a não interferência funcional depende de pré-treino (Ortiz-Jimenez et al., 2023; Huang et al., 2024);
   - o cosseno não previu a qualidade do merge por tarefa, enquanto o alinhamento de subespaço previu (Marczak et al., 2025);
   - a interferência pode ser definida no espaço de funções, como desvio da saída de cada camada sobre as entradas da própria tarefa (Cheng et al., 2025).

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

**2020 — Farajtabar et al. (OGD): "interference" em aprendizado contínuo.**
- *"In this paper, we present a system where the gradients produced on a training minibatch can avoid interfering with gradients produced on previous tasks."* (Sec. 1).
- *"moving locally along the direction of ±∇f_j(x;w) leads to the biggest change in model prediction f_j(x;w) given any sample x, while moving orthogonal to ∇f_j(x;w) leads to the least change (or no change, locally) to the prediction of x."* (Sec. 3).
- A interferência é operacional e de primeira ordem no espaço de funções: um update interfere quando tem componente no span dos gradientes de saída (logits) da tarefa anterior. Não há escalar de interferência, e o artigo mede só acurácia. Distingue gradientes de saída de gradientes de perda (Eq. 4), que se anulam em exemplos bem ajustados (Farajtabar et al., 2020).
- O cenário é sequencial, não agregação simultânea. Os termos "cancellation", "conflict" e "negative transfer" não aparecem.

**2021 — Wang et al. (GradVac): similaridade contínua.**
- *"two tasks have a positive gradient similarity of cos(θ)=ϕij"* (Sec. 3.2).
- Sobre o PCGrad: ele *"implicitly makes the assumption that any two tasks must have the same gradient similarity objective of zero."* (Sec. 3.1).
- Usa "negative interference" de forma informal (Introdução) (Wang et al., 2021a).

**2021 — Wang et al. (FedFV): conflito entre clientes.**
- *"Client i's gradient g_i ∈ R^d conflicts with client j's gradient g_j ∈ R^d iff g_i · g_j < 0."* (Sec. 3, Def. 1).
- O dano vem do conflito somado à disparidade de magnitude: *"when the conflicting gradients have large differences in their magnitudes, model accuracy can suffer significant reductions for some of the clients"* (Introdução) (Wang et al., 2021b).

**2022 — Matena & Raffel: FedAvg como "isotropic merging".**
- *"simple parameter averaging is implicitly using an isotropic Gaussian posterior approximation."* (Sec. 2.3). *"Merging is the core component of the FedAvg algorithm used in Federated Learning"* (Sec. 1).
- O merge Fisher é, por parâmetro, θ*(j) = Σᵢλᵢ Fᵢ(j) θᵢ(j) / Σᵢλᵢ Fᵢ(j) (Eq. 4). A média do FedAvg é o caso Fᵢ = I.
- A falha da média não é descrita como cancelamento de sinal, mas como desprezo da importância por parâmetro. O artigo não usa "interference", "conflict" nem "cancellation", e só funde o corpo compartilhado: as cabeças ficam sem merge, numa estrutura parecida com a do FedPer (Sec. 2.4) (Matena & Raffel, 2022).

**2022 — Wortsman et al. (Model soups): ângulo entre updates e "interpolation advantage".**
- *"we compare the accuracy of their average with the average of their accuracies, [...] which we refer to as the interpolation advantage. [...] we observe that the interpolation advantage is correlated with the angle $\phi$"* (Sec. 3.2). O ângulo é *"the angle between $\theta_1 - \theta_0$ and $\theta_2 - \theta_0$ (i.e., the initialization is treated as the origin)."* (Fig. 2, legenda).
- *"more uncorrelated solutions---models that form an angle closer to 90 degrees---may lead to higher accuracy on the linear interpolation path."* (Sec. 3.2). Aqui a quase-ortogonalidade é **boa** para a média, desde que todos os modelos fiquem na mesma bacia.
- Modo de falha: *"there can be an error barrier between fine-tuned models. We mainly observe this when fine-tuning with high learning rates"* (Sec. 3.3). O artigo não usa "cancellation" nem "interference" (Wortsman et al., 2022).

**2022 — Tenison et al. (GMA): concordância de sinal por coordenada.**
- *"averaging can result in information loss"* e *"the bias induced by dominant client gradients"* (Abstract). A máscara *"controls the parameter update based on the agreement of direction among the gradients across clients"* (Sec. 3.2).
- Formalmente, A_j = |(1/N) Σₙ sign(Δⱼⁿ)| ∈ (0, 1] (Eq. 3): vale 1 quando todos os clientes concordam no sinal da coordenada j e fica perto de 0 quando os sinais se dividem. O artigo evita os termos "cancel", "conflict" e "interfere" (Tenison et al., 2022).

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

**2023 — Li et al. (FedLAW): "local gradient coherence" e "global weight shrinking".**
- Definição 5.1: *"The local gradient coherence of two clients i and j at round t is defined by the cosine similarity of their local updates sent to the server, as c(i,j)^t = cos(g_i^t, g_j^t). The overall local gradient coherence of a cohort of clients at round t is defined by the weighted cosine similarity of all clients' local updates sent to the server, as coh^t = (1/m) Σ_{i,j,i≠j} λ_i λ_j cos(g_i^t, g_j^t)."* (Sec. 5.1).
- *"In this case, the local gradients of clients are usually almost orthogonal, which means that they have low coherence."* (Sec. 5.1).
- Eq. 6, expansão de primeira ordem da perda numa rodada: ΔL ≈ −η(Σλᵢ²‖gᵢ‖² + Σ_{i≠j} λᵢλⱼ cos(gᵢ, gⱼ)‖gᵢ‖‖gⱼ‖). A descida é o quadrado da norma do update agregado, decomposto em termo próprio e termo cruzado.
- *"The critical point is the round that the coherence is near zero. After the critical point, the test accuracy gain is marginal, and the coherence is kept negative but close to zero."* (Sec. 5.2).
- A agregação é w_g = γΣλᵢwᵢ (Eq. 3). Com γ < 1, ela equivale a weight decay no servidor (Eq. 4). O único "conflito" nomeado é entre esse termo de regularização e o pseudo-gradiente, não entre clientes (Li et al., 2023).

**2023 — Ainsworth et al. (Git Re-Basin): loss barrier e simetria de permutação.**
- *"Given two points ΘA,ΘB such that ℒ(ΘA)≈ℒ(ΘB), the loss barrier is defined as [max_{λ∈[0,1]} ℒ((1−λ)ΘA+λΘB) − ½(ℒ(ΘA)+ℒ(ΘB))] ... Loss barriers are non-negative, with zero indicating an interpolation of flat or positive curvature."* (Sec. 2; equação transcrita da notação do artigo).
- A média ingênua é *"the status quo for model combination in federated learning and distributed training."* (Sec. 5.4).
- A falha tratada é o desalinhamento de unidades ocultas entre modelos treinados **independentemente**, com inicializações diferentes. Não é conflito entre tarefas, e o artigo não usa "cancellation", "interference", "conflict" nem "negative transfer" (Ainsworth et al., 2023).

**2024 — Wang et al. (TALL/Consensus): separa "information erasure" de "task interference".**
- *"Information erasure: large amount of information specific to each task is erased when merging the multi-task vector. Task interference: the task-specific information is preserved in the multi-task vector, but can not manifest properly due to interference between the tasks."* (Sec. 3).
- Conclusão: a degradação *"is not due to information erasure during merging process, but because of task interference during evaluation."* (Sec. 7) (Wang et al., 2024).

**2024 — Yang et al. (AdaMerging): conflito de sinal e coeficiente de merge.**
- *"some parameter values in the task vector may be redundant, or the signs of the parameters may conflict, and direct merging will cause performance losses."* (Sec. 3.1).
- *"these task vectors are almost orthogonal to each other."* (App. B.2). Mesmo assim, a soma com coeficiente único perde ~20 pontos em ViT-B/32, e coeficientes por tarefa e por camada recuperam boa parte da perda.
- O artigo não usa "cancellation", "negative transfer" nem "parameter interference" como termos definidos (Yang et al., 2024a).

**2024 — Yu et al. (DARE): "parameter interference" e reescala.**
- *"we hypothesize that DARE can help address the interference of parameters when merging multiple models"* (Sec. 3.2).
- A interferência é nomeada, mas nunca definida nem medida. O que o artigo formaliza é o operador: descartar uma fração p do delta e reescalar os sobreviventes por 1/(1−p), o que preserva a saída esperada (*"By setting γ=1/(1-p), we have E[h_i]=E[ĥ_i]"*, Sec. 3.1) (Yu et al., 2024).

**2024 — Huang et al. (EMR-Merging): lacuna de representação de um único vetor.**
- *"We discover that using a single model's weights can hardly simulate all the models' performance."* (Abstract).
- O fenômeno é definido pelo objetivo Dis = (1/N) Σᵢ‖τᵢ − τ_uni‖² (Eq. 3), a distância quadrática média entre o vetor fundido e cada especialista. "Interference" e "sign conflicts" aparecem só na acepção do TIES (Huang et al., 2024).

**2024 — Lu et al. (Twin-Merging): conhecimento compartilhado vs exclusivo.**
- A parameter interference *"refers to the redundancy or sign discrepancies found in parameters located at the same position"*; a task interference *"arises from the inherent differences between tasks."* (§3.1, trechos fragmentados pela extração).
- *"The exclusive nature of task-specific knowledge hinders the effectiveness of merging methods."* (§3.2). A média preserva o componente compartilhado e dilui o exclusivo, mesmo sem conflito de sinal. A interferência só é inferida da queda de desempenho (Lu et al., 2024).

**2024 — Yang et al. (survey de model merging): vocabulário consolidado.**
- *"There are conflicts and interference among multiple models."* (§2.1). Sobre AWD: *"interference between tasks can be minimized when task vectors are mutually orthogonal."* (§2.3.4).
- Sobre FL: *"Most FL methods adopt a simple coordinate-wise average to aggregate the local models"* (§5.4).
- Não há definição formal única nem métrica padrão de interferência, e *"a unified theoretical framework for model merging has not yet emerged"* (§3.1). A palavra "cancellation" não aparece no texto lido (Yang et al., 2024b; citações truncadas em ~125 caracteres).

**2025 — Gargiulo et al. (TSV): "Singular Task Interference".**
- *"Given that a formal definition of interference in model merging is not yet established, we adopt an operational definition: interference is any cross-task interaction that hinders the merging process."* (App. E.2).
- STI = ‖(UᵀU − I) Σ (VᵀV − I)‖₁ (Sec. 3.3, Eq. 6).
- O artigo chama o cosseno de *"coarse-grained"* (Sec. 1) (Gargiulo et al., 2025).

**2025 — Zheng et al. (FedCALM): "server aggregation conflicts".**
- *"when ∆θ(k)i·∆θ(k)j < 0 or ∥∆θ(k)avg∥² < Σ_{i=1}^N p_i² ∥∆θ(k)i∥², it leads to an increase in global loss. Specifically, this situation can be further detailed into two main sources: insufficient average update magnitude and directional conflicts between client updates."* (Sec. 3.3, após Eq. 4, p. 15447).
- O cenário é FedRep/FedPer (Zheng et al., 2025).

**2025 — Shi et al. (FedLWS): "gradient variance" como medida de desacordo.**
- *"if local client models agree less with each other in terms of the gradient updates (i.e., higher variance of their gradients), the global model should be more conservative in terms of updating its weights, meaning that a stronger regularization is needed (i.e., a smaller γ_t)."* (Sec. 4.1).
- O desacordo é τ = (1/K) Σₖ‖gₖ − g_mean‖ (Eq. 4): a distância L2 média, não ao quadrado e não ponderada, dos deltas à média. O artigo não usa "cancellation", "conflict", "interference" nem "negative transfer" (Shi et al., 2025a).

**2025 — Shi et al. (FedAWA): "client vectors" e direções divergentes.**
- A heterogeneidade estaria *"causing divergent optimization directions and unstable convergence"* (Sec. 1). Pesos maiores vão para *"clients whose update directions more align with the merged global vector"* (Sec. 4.3; trechos truncados).
- O client vector é τₖ = θₖ − θ_g (Eq. 1), o mesmo objeto que o Δᵢ do projeto. Não há métrica de interferência nem de cancelamento (Shi et al., 2025b).

**2025 — Marczak et al. (Iso-C/Iso-CTS): interferência como degradação de representação.**
- *"we define the interference as the degradation in a task's internal representation due to merging"* (App. A.3), medida pelo desvio entre as ativações do modelo fundido e as do especialista, com cota ‖Δⱼ − αΔ_M‖·‖xⱼ‖ (Eq. 17).
- Sobre o cosseno: *"we observe no clear correlation (see Figure 2b)"* (Sec. 3.2). O que correlaciona com o ganho é o alinhamento de subespaço (SAR, ρ = 0,94; Sec. 3.3). Tarefas fora do subespaço dominante ficam *"underrepresented in the dominant subspace of ΔTA"* (Sec. 3.3) (Marczak et al., 2025).

**2025 — Cheng et al. (WUDI-Merging): definição funcional por camada.**
- *"for task $i$, we defined the rest of the weight in $\tau_m$ as interference vector $\delta_i$, which is formulated as: $\delta_i=\tau_m-\tau_i$"* (Sec. 3.1, Eq. 2).
- *"we can formally define the interference of the $l$-th linear linear for task $i$ as follows [...] $\mathcal{J}_i(\tau_{m,l}) = \mathbb{E}_{x_{i,l}\sim p(x_{i,l})}\|\theta_{m,l}x_{i,l}-\theta_{i,l}x_{i,l}\|_2^2$ [...] $= \mathbb{E}\|\delta_{i,l}x_{i,l}\|_2^2$"* (Sec. 3.1, Eqs. 3–5).
- Na análise por camada (Fig. 3): *"interference accumulates progressively as the depth of the network increases. Specifically, task arithmetic introduces a relative error of nearly 50% in the final output."* (Cheng et al., 2025; venue não confirmado).

**Quem cunhou o quê:**
- "Cancel each other out" ligado a uma razão de normas: Nagura et al. (2020), sob o nome WUDD.
- "Conflicting gradients" como cosseno < 0: Yu et al. (2020) em MTL. Foi transportado para FL por Wang et al. (2021b) e Zhang et al. (2023).
- "Local gradient coherence", o cosseno médio ponderado entre clientes, e "global weight shrinking": Li et al. (2023). O segundo termo foi retomado por Shi et al. (2025a).
- "Client-drift": Karimireddy et al. (2020).
- "Objective inconsistency": Wang et al. (2020).
- "Interference" paramétrica em merging: Yadav et al. (2023), Wang et al. (2024) e Gargiulo et al. (2025). Definições funcionais formais: Cheng et al. (2025) e Marczak et al. (2025), além da noção de primeira ordem de Farajtabar et al. (2020).
- Barreira de perda e conectividade linear de modos, no corpus avaliado: Ainsworth et al. (2023) e Wortsman et al. (2022).
- Nenhum dos 30 artigos usa "weight cancellation" como termo técnico.

### 2.2 Tabela comparativa de definições

| Autor | Ano | Termo | Definição (resumida) | Nível | Como mede |
|---|---|---|---|---|---|
| Sattler et al. | 2019 | incongruence | Não existe ótimo comum. No estacionário, os updates continuam não nulos e opostos. | cliente (distribuição) | cosseno entre weight-updates (Eq. 44); ‖update servidor‖ < ε₁ e max‖Δᵢ‖ > ε₂ |
| Nagura et al. | 2020 | WUDD ("cancel each other out") | A média encolhe a norma do update agregado. | cliente → servidor (modelo inteiro) | N = ‖Σwᵢ Δᵢ‖ vs E = Σwᵢ‖Δᵢ‖ (**= 1 − razão do projeto**) |
| Yu et al. | 2020 | conflicting gradients / tragic triad | cos φᵢⱼ < 0. Só é danoso com disparidade de magnitude e curvatura alta. | tarefa (gradiente por passo) | cosseno; Φ; curvatura H |
| Karimireddy et al. | 2020 | client-drift | Passos locais derivam para o ótimo de cada cliente, o que enviesa a média. | cliente (trajetória) | constantes teóricas (G, B), δ; rodadas até a acurácia-alvo |
| Wang et al. | 2020 | objective inconsistency | A agregação otimiza Σwᵢ Fᵢ ≠ Σpᵢ Fᵢ quando o trabalho local é desigual. | cliente (pesos efetivos) | χ²_{p‖w}; slowdown τ̄/τ_eff (teórico) |
| Farajtabar et al. (OGD) | 2020 | interference (aprendizado contínuo) | Um update interfere quando tem componente no span dos gradientes de saída da tarefa anterior (1ª ordem). | função (gradientes de logit) | só acurácia nas tarefas antigas |
| Wang et al. (GradVac) | 2021a | gradient similarity / negative interference | Similaridade contínua. Abaixo da similaridade "natural" já há prejuízo. | tarefa, por camada | cosseno por camada e por passo |
| Wang et al. (FedFV) | 2021b | conflicting gradients (internal/external) | gᵢ·gⱼ < 0 entre clientes. Danoso com disparidade de magnitude. | cliente | teste binário de sinal |
| Matena & Raffel | 2022 | isotropic merging | A média supõe posterior gaussiana isotrópica e ignora a importância por parâmetro. | parâmetro | só desempenho downstream |
| Wortsman et al. | 2022 | interpolation advantage / error barrier | A média de modelos de uma mesma bacia ganha mais quando os updates são mais ortogonais. | modelo (fine-tuning completo) | Acc(média) − média das Acc; ângulo φ entre θᵢ − θ₀ |
| Tenison et al. (GMA) | 2022 | sign agreement / information loss | Fração de clientes que concordam no sinal de cada coordenada. | parâmetro (coordenada) | A_j = \|média de sign(Δⱼⁿ)\| |
| Ilharco et al. | 2023 | task vector / interference (informal) | τ = θ_ft − θ_pre. Vetores quase ortogonais somam com pouca interferência (especulação). | tarefa (pesos) | cosseno entre task vectors; acurácia normalizada |
| Yadav et al. | 2023 | parameter interference (redundância, conflito de sinal) | Sinais opostos e zeros redundantes encolhem o valor do parâmetro fundido. | parâmetro (coordenada) | % de parâmetros com conflito de sinal |
| Ortiz-Jimenez et al. | 2023 | weight disentanglement | O update de uma tarefa não altera a saída fora do suporte dela. | função / domínio | erro de disentanglement ξ(α₁, α₂) |
| Jhunjhunwala et al. | 2023 | pseudo-gradient / gradient diversity | O update médio é muito menor que os individuais, o que justifica passo η_g > 1. | cliente → servidor | Σ‖Δᵢ‖² / (2M‖Δ̄‖²) |
| Zhang et al. | 2023 | gradient conflict | cos(gⁱ, gʲ) < 0 entre deltas de cliente. | cliente (modelo achatado) | cosseno mínimo; fração de pares negativos |
| Li et al. (FedLAW) | 2023 | local gradient coherence / global weight shrinking | Cosseno médio ponderado entre updates de clientes. Coerência baixa encolhe a descida (Eq. 6). | cliente → servidor | coh = (1/m) Σ_{i≠j} λᵢλⱼ cos(gᵢ, gⱼ); γ aprendido |
| Ainsworth et al. | 2023 | loss barrier / LMC | Máximo excesso de perda no caminho linear entre dois modelos, módulo permutação. | função (caminho em pesos) | barreira de perda e de acurácia |
| Wang et al. (TALL) | 2024 | information erasure vs task interference | A informação sobrevive na soma, mas não se manifesta. | parâmetro / tarefa | recuperação via máscara; perfil selfish/universal |
| Yang et al. (AdaMerging) | 2024a | conflicts (redundância, sinal) | Coeficiente de merge inadequado impede o MTL; vetores quase ortogonais ainda perdem. | tarefa, por camada | acurácia média; cosseno entre task vectors |
| Yu et al. (DARE) | 2024 | parameter interference | Nomeada, não definida. Hipótese: deltas redundantes interferem no merge. | parâmetro | só desempenho downstream |
| Huang et al. (EMR) | 2024 | weight-representation gap | Um único vetor não representa todos os especialistas. | parâmetro / tarefa | Dis = (1/N) Σ‖τᵢ − τ_uni‖²; razão de conflito de sinal; L2; cosseno |
| Lu et al. (Twin) | 2024 | parameter vs task interference; shared vs exclusive | A média preserva o compartilhado e dilui o exclusivo. | tarefa | normalized score (Eq. 4) |
| Yang et al. (survey) | 2024b | task conflict/interference | Três acepções: sinal, redundância/subespaço e não ortogonalidade (pesos ou NTK). | vários | indireto (acurácia vs especialistas) |
| Gargiulo et al. | 2025 | Singular Task Interference | Sobreposição entre subespaços singulares por camada. | camada (matriz) | STI (Eq. 6) |
| Zheng et al. | 2025 | server aggregation conflicts | Termos cruzados negativos aumentam a perda global. Duas fontes: magnitude e direção. | cliente, por camada | gap ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖²; "conflict rate" (fórmula no apêndice, não lido) |
| Shi et al. (FedLWS) | 2025a | gradient variance | Dispersão dos deltas em torno da média; mais dispersão pede passo mais conservador. | cliente, por camada | τ = (1/K) Σ‖gₖ − g_mean‖ |
| Shi et al. (FedAWA) | 2025b | divergent optimization directions | Clientes desalinhados com o vetor global consenso. | cliente, por camada (variante L) | sem métrica; distância a um vetor "ideal" |
| Marczak et al. | 2025 | task interference / underrepresentation | Degradação de representação; tarefa fora do subespaço dominante do merge. | camada (matriz) | SAR = ‖Π Δₜ‖_F / ‖Δₜ‖_F; desvio de ativações |
| Cheng et al. (WUDI) | 2025 | parameter interference (funcional) | J_i = E‖(τ_m − τ_i) x_i‖² sobre as entradas da própria tarefa. | camada linear | proxy ‖δᵢτᵢᵀ‖²_F; erro relativo por camada (Eq. 83) |

## 3. Divergências e ambiguidades entre definições

**1. "Cancelamento de pesos" não é termo padrão.** Ele corresponde a pelo menos dez termos em uso:

| Termo usado na literatura | Fonte | O que de fato descreve |
|---|---|---|
| update cancellation / WUDD | Nagura et al. (2020) | perda de norma na média |
| client drift | Karimireddy et al. (2020) | viés de trajetória, não oposição |
| objective inconsistency | Wang et al. (2020) | viés de pesos efetivos, não perda de norma |
| gradient conflict | Yu et al. (2020); Wang et al. (2021b); Zhang et al. (2023) | sinal do cosseno |
| local gradient coherence | Li et al. (2023) | cosseno médio ponderado, contínuo |
| gradient variance | Shi et al. (2025a) | dispersão dos deltas, que mistura magnitude e direção |
| sign conflict / sign agreement | Yadav et al. (2023); Tenison et al. (2022) | por coordenada |
| parameter interference | Wang et al. (2024); Gargiulo et al. (2025); Cheng et al. (2025); Marczak et al. (2025) | efeito funcional ou subespacial |
| weight-representation gap / exclusive knowledge | Huang et al. (2024); Lu et al. (2024) | distância de um vetor único a cada especialista; diluição do componente exclusivo |
| loss barrier | Ainsworth et al. (2023); Wortsman et al. (2022) | perda no caminho linear entre modelos |

**2. Direção e magnitude.**
- Os critérios de conflito são binários e ignoram a magnitude da perda de norma (Yu et al., 2020; Wang et al., 2021b; Zhang et al., 2023).
- Nagura et al. (2020) medem só a magnitude agregada e não separam conflito de ortogonalidade.
- Zheng et al. (2025) nomeiam as duas fontes, mas na Eq. 4 a parte de magnitude aparece como subproduto dos termos cruzados. Updates ortogonais dão gap = 0 e, portanto, são tratados como inócuos.
- A dispersão τ de Shi et al. (2025a) cresce tanto com direções opostas quanto com normas muito diferentes e direções alinhadas, de modo que não separa os dois efeitos.
- A concordância de sinal A_j (Tenison et al., 2022) olha só direção e ignora magnitude, por construção.
- Huang et al. (2024) tratam as duas partes de forma explícita e separada: máscara de sinal para a direção e um reescalonador escalar por tarefa para a magnitude.

**3. A ortogonalidade é ambígua.**
- Para Ilharco et al. (2023), ela permite "minimal interference" (especulação). O survey repete a leitura NTK de que vetores "approximately orthogonal" não se atrapalham (Yang et al., 2024b, §3.1).
- Para Wortsman et al. (2022), ela é **desejável**: soluções mais ortogonais dão maior interpolation advantage, desde que fiquem na mesma bacia.
- Para Li et al. (2023), é sinal de **baixa coerência**: depois do "critical point" de coerência ≈ 0, os ganhos de acurácia ficam marginais.
- Para Jhunjhunwala et al. (2023), é justamente o caso que pede extrapolação (η_g ≈ M/2).
- Para Yang et al. (2024a), task vectors quase ortogonais ainda perdem ~20 pontos sob soma uniforme. O problema ali é escala, não direção.
- Para Farajtabar et al. (2020), ortogonalidade aos gradientes de saída de outra tarefa significa não interferência, mas só localmente e em primeira ordem.
- Para Ortiz-Jimenez et al. (2023), ortogonalidade geométrica não equivale a não interferência funcional (inferência a partir da definição funcional; o artigo não discute cosseno).
- Marczak et al. (2025) mostram empiricamente que o cosseno entre task vector e vetor fundido não prevê o ganho por tarefa.
- Wang et al. (2024) mostram degradação com suportes disjuntos, o que dá cancelamento zero por construção.

**4. Nível de análise.** As medidas existem em seis níveis: coordenada (Yadav et al., 2023; Tenison et al., 2022), camada ou matriz (Gargiulo et al., 2025; Zheng et al., 2025; Marczak et al., 2025; Shi et al., 2025a), modelo achatado (Zhang et al., 2023; Nagura et al., 2020; Li et al., 2023), função ou saída por camada (Ortiz-Jimenez et al., 2023; Cheng et al., 2025; Farajtabar et al., 2020), caminho linear no espaço de pesos (Ainsworth et al., 2023; Wortsman et al., 2022) e distribuição (Sattler et al., 2019). Esses níveis podem discordar entre si. Muitos conflitos de sinal por coordenada podem coexistir com baixo cancelamento em norma (Yadav et al., 2023), e cosseno baixo pode coexistir com sub-representação de subespaço (Marczak et al., 2025).

**5. Objeto medido e regime.**
- Gradientes por passo em modelo centralizado (Yu et al., 2020; Wang et al., 2021a; Farajtabar et al., 2020).
- Deltas de vários passos locais (Zhang et al., 2023; Sattler et al., 2019; Li et al., 2023; Shi et al., 2025a, 2025b; Tenison et al., 2022).
- Task vectors de fine-tuning completo e merge único (Ilharco et al., 2023; Yadav et al., 2023; Yang et al., 2024a; Huang et al., 2024; Lu et al., 2024; Marczak et al., 2025; Cheng et al., 2025).
- Modelos de inicialização independente (Ainsworth et al., 2023), regime oposto ao do FedAvg, em que todos partem do mesmo ponto a cada rodada.
- Quase todo o merging supõe um pré-treino comum e deltas pequenos. EMR-Merging *"cannot be generalized to models trained from-scratch"* (Huang et al., 2024, App. G). DARE *"can work well when the absolute values of SFT delta parameters are relatively small (e.g., within 0.002)... Otherwise, DARE may fail"* (Yu et al., 2024, Sec. 4.6).
- A transferência entre esses regimes não é automática.

**6. O que fazer quando os clientes discordam: encolher ou amplificar.**
- **Encolher:** Li et al. (2023) aprendem γ < 1 (weight decay no servidor); Shi et al. (2025a) reduzem γ quanto maior o desacordo; Tenison et al. (2022) amortecem as coordenadas sem concordância de sinal.
- **Amplificar:** Jhunjhunwala et al. (2023) extrapolam com η_g ≥ 1 quando ‖Δ̄‖ é pequeno frente aos ‖Δᵢ‖; Nagura et al. (2020) reescalam por E/N.
- **Merging:** a soma com λ ajustado é preferida à média (Yang et al., 2024a), e magnitude é restaurada por reescala (Yu et al., 2024; Huang et al., 2024).
- Os dois grupos de FL nunca são comparados entre si nos artigos avaliados. Os ganhos de "encolher" vêm de classificação com skew de rótulo e modelo global único, e Shi et al. (2025a) os atribuem a um efeito de regularização.

**7. Espaço de parâmetros ou espaço de funções.** CR, cosseno, A_j e τ são geométricos. J_i (Cheng et al., 2025), desvio de ativação (Marczak et al., 2025), disentanglement (Ortiz-Jimenez et al., 2023) e loss barrier (Ainsworth et al., 2023) são funcionais. Sob a definição de Cheng et al. (2025), a própria diluição conta como interferência: com média ponderada, δᵢ contém −(1 − wᵢ)τᵢ, que é grande na direção de τᵢ mesmo quando os termos cruzados são nulos. Essa leitura é derivação nossa a partir da Eq. 2, não está no artigo.

**Qual termo casa com a razão do projeto, CR = 1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖:**
- **A métrica bruta é exatamente o 1 − N/E de Nagura et al. (2020)** ("weight updating direction divergence"). Esse é o termo a citar como definição.
- **O excesso sobre a referência ortogonal** do projeto, 1 − √(Σwᵢ²‖Δᵢ‖²) / Σwᵢ‖Δᵢ‖, tem o mesmo sinal que o "server aggregation conflict" de Zheng et al. (2025). Vale CR > referência ⇔ G = ‖Σwᵢ Δᵢ‖² − Σwᵢ²‖Δᵢ‖² < 0. Essa equivalência é uma derivação, não está no artigo, e a mesma ressalva vale para todas as derivações marcadas abaixo.
- **G é o termo cruzado da Eq. 6 de Li et al. (2023)**, Σ_{i≠j} wᵢwⱼ cos(Δᵢ, Δⱼ)‖Δᵢ‖‖Δⱼ‖. A matriz de cossenos do projeto e o CR são, portanto, duas leituras da mesma identidade (derivação nossa).
- **A dispersão de FedLWS**, elevada ao quadrado e com pesos uniformes, satisfaz Σₖ‖Δₖ − Δ̄‖² = Σₖ‖Δₖ‖² − K‖Δ̄‖². Ela mede a mesma distância entre média das normas e norma da média, mas mistura magnitude e direção (derivação nossa; Shi et al., 2025a).
- Recomendação de nomenclatura:
  - "cancelamento", ou perda de norma da agregação (Nagura et al., 2020), para CR;
  - "conflito", na acepção de Zheng et al. (2025) e Yu et al. (2020), para o excesso sobre a referência ortogonal;
  - "coerência" (Li et al., 2023) para o cosseno médio ponderado entre clientes;
  - "diluição de passo" para a perda que existe mesmo sem conflito, ligada a "gradient diversity" e ao passo do servidor (Jhunjhunwala et al., 2023), ao "slowdown" (Wang et al., 2020) e à diluição do conhecimento exclusivo (Lu et al., 2024);
  - "interferência" reservada para o efeito funcional, como em Cheng et al. (2025) e Marczak et al. (2025).

## 4. Como é medido (métricas)

| Métrica | Fórmula / descrição | Fonte | Observação |
|---|---|---|---|
| Razão de normas N/E | ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖ | Nagura et al. (2020) | 1 − N/E = CR do projeto. Sem referência ortogonal. |
| Gap quadrático de agregação | ‖Σpᵢ Δᵢ‖² − Σpᵢ²‖Δᵢ‖² = Σ_{i≠j} pᵢpⱼ Δᵢ·Δⱼ | Zheng et al. (2025) | negativo ⇔ conflito. Por camada. |
| Coerência de coorte | coh = (1/m) Σ_{i≠j} λᵢλⱼ cos(gᵢ, gⱼ) | Li et al. (2023) | por rodada; "critical point" quando coh ≈ 0 |
| Decomposição da descida | ‖Σλᵢgᵢ‖² = Σλᵢ²‖gᵢ‖² + Σ_{i≠j} λᵢλⱼ cosᵢⱼ‖gᵢ‖‖gⱼ‖ | Li et al. (2023), Eq. 6 | liga cosseno e norma agregada |
| Razão FedExP | Σ‖Δᵢ‖² / (2M‖Δ̄‖²) | Jhunjhunwala et al. (2023) | versão quadrática da diluição. Pesos uniformes. |
| Dispersão τ | (1/K) Σₖ‖gₖ − g_mean‖, por camada | Shi et al. (2025a) | não ao quadrado, não ponderada; mistura magnitude e direção |
| Cosseno par a par | Δᵢ·Δⱼ / (‖Δᵢ‖‖Δⱼ‖) | Sattler et al. (2019); Zhang et al. (2023); Wang et al. (2021a) | por camada (Wang et al., 2021a) ou modelo inteiro |
| Cosseno mínimo / fração de pares negativos | min cos; #{cos < 0} / #pares | Zhang et al. (2023) | valores relatados: −0,06 a −0,22 (rótulos não-IID) |
| Separation gap | g(α) = α_intra^min − α_cross^max | Sattler et al. (2019) | separa grupos de clientes |
| Similaridade de magnitude | Φ = 2‖gᵢ‖‖gⱼ‖ / (‖gᵢ‖² + ‖gⱼ‖²) | Yu et al. (2020) | condição da tragic triad |
| Conflito de sinal | % de coordenadas com sinais divergentes após trim top-k | Yadav et al. (2023) | chega a ~80% com pouco trim |
| Concordância de sinal | A_j = \|(1/N) Σₙ sign(Δⱼⁿ)\| | Tenison et al. (2022) | por coordenada; só direção |
| Razão de conflito de sinal, L2 e cosseno vs especialista | fração de elementos com sinal divergente entre o vetor fundido e cada especialista | Huang et al. (2024), App. F | por tarefa |
| Reescalonador de magnitude | λᵢ = Σ\|τᵢ\| / Σ\|Mᵢ ⊙ τ_uni\| | Huang et al. (2024) | fator pelo qual a magnitude da tarefa encolheu no merge |
| Distância ao especialista | Dis = (1/N) Σᵢ‖τᵢ − τ_uni‖² | Huang et al. (2024), Eq. 3 | objetivo do merge |
| Perfil de pesos | catastrophic / selfish / universal pela máscara m_t = 1{\|τ_t\| ≥ \|τ_MTL − τ_t\|} | Wang et al. (2024) | "posse" coordenada a coordenada |
| STI | ‖(UᵀU − I) Σ (VᵀV − I)‖₁ por camada | Gargiulo et al. (2025) | mede sobreposição de subespaço, não cancelamento realizado |
| SAR | ‖Π_{k_M} Δₜ‖_F / ‖Δₜ‖_F, com k_M para ε = 0,05 | Marczak et al. (2025), Eqs. 5–6 | ρ = 0,94 com o ganho por tarefa; o cosseno não correlacionou |
| Interferência funcional | J_i = E‖δᵢ xᵢ‖²; proxy sem dados ‖δᵢτᵢᵀ‖²_F / ‖τᵢ‖²_F | Cheng et al. (2025), Eqs. 3–5 e 21 | só camadas lineares |
| Erro relativo por camada | (1/N) Σ‖f(x; θ+τ_m) − f(x; θ+τ_i)‖ / ‖f(x; θ+τ_i)‖ | Cheng et al. (2025), Eq. 83 | precisa de dados da tarefa |
| Erro de disentanglement | ξ = Σ_t E[1(f(θ₀+α_tτ_t) ≠ f(θ₀+α₁τ₁+α₂τ₂))] | Ortiz-Jimenez et al. (2023) | sonda funcional |
| Loss barrier | max_λ L((1−λ)Θ_A + λΘ_B) − ½[L(Θ_A) + L(Θ_B)] | Ainsworth et al. (2023) | função; ≥ 0 |
| Interpolation advantage e ângulo | Acc(θ₁/2 + θ₂/2) − ½(Acc(θ₁) + Acc(θ₂)); φ = ∠(θ₁ − θ₀, θ₂ − θ₀) | Wortsman et al. (2022) | origem na inicialização comum |
| Fisher diagonal | média de (∇ log p(y\|x))², y amostrado do modelo | Matena & Raffel (2022) | importância por parâmetro, não interferência |
| Normalized score | (1/T) Σₜ Score(merge) / Score(especialista) | Lu et al. (2024) | interferência inferida por desempenho |
| χ²_{p‖w}, slowdown | Σ(pᵢ − wᵢ)² / wᵢ; τ̄/τ_eff | Wang et al. (2020) | apenas teórico |

## 5. Estado da arte de mitigação (2019–2025, ênfase 2023–2025)

Cobertura: depois da segunda passada, 19 das 30 avaliações são de 2023–2025, e a lacuna de cobertura recente apontada na primeira passada está fechada para model merging (TSV, Iso-CTS, WUDI, EMR, Twin, AdaMerging) e para agregação em FL (FedCALM, FedLWS, FedAWA). Restam duas ressalvas: nenhum trabalho de 2026 foi avaliado, e a linha específica de FL continua menor que a de merging (três artigos de 2025).

| Método | Ano | Ideia | Onde se aplica | Evidência |
|---|---|---|---|---|
| Iso-C / Iso-CTS (Marczak et al.) | 2025 | Achatar o espectro singular do update somado (todos os valores singulares iguais à média). Iso-CTS ainda acrescenta direções específicas de cada tarefa ao subespaço comum. | model merging | ViT-B/16, 8 tarefas: TA 75,4 → Iso-C 90,6, Iso-CTS 91,1. ViT-L/14, 20 tarefas: TA 74,0 → Iso-CTS 90,1. α ajustado em validação; pré-treino CLIP. |
| WUDI-Merging (Cheng et al.) | 2025 | Por camada linear e sem dados, minimizar Σᵢ(1/‖τᵢ‖²_F)‖(τ_m − τᵢ)τᵢᵀ‖²_F, partindo da soma TA | model merging | ViT-B/32, 8 tarefas: 85,2 vs TA 69,1 e AdaMerging++ 81,1. ViT-L/14: 92,6. Só camadas lineares; venue não confirmado. |
| FedCALM (Zheng et al.) | 2025 | Por camada: projeção PCGrad entre clientes, depois uma QP que garante dᵀΔᵢ ≥ ε‖Δᵢ‖ para todo cliente | FL personalizado (FedRep/FedPer) | +1,6 a +9,9 pp em Flowers102 com ResNets e ganhos como plug-in. Pior que FedPAC com CNN rasa. 3 seeds, sem variância. d* pode ter norma maior que o passo FedAvg, sem controle de passo do servidor. |
| FedLWS (Shi et al.) | 2025a | Após qualquer agregador, multiplicar cada camada por γ_l = ‖w_l‖ / (βτ_l‖ηg_l‖ + ‖w_l‖) < 1 | FL (modelo global) | FedAvg 58,02 → 59,30 (média de 12 cenários). CIFAR-10 com α = 0,1: 61,04 → 64,08. Encolhe o passo; γ ≈ 0,99 com pré-treino. Escalar só o passo (only-opt) perde ~2,8 pp. |
| FedAWA (Shi et al.) | 2025b | Pesos de agregação otimizados: min Σλₖ‖τₖ − τ_g‖ com ‖λ‖₁ = 1 e regularizador de cosseno; variante por camada | FL (modelo global) | Média geral: FedAvg 47,88 → FedAWA 49,48, FedAWA-L 49,98. Combinação convexa, sem alterar a norma do passo. FedAWA-L custa ~150× o tempo de servidor do FedAvg. Confiança média. |
| TSV-Merge (Gargiulo et al.) | 2025 | Truncar cada task matrix ao posto 1/T e ortogonalizar (Procrustes) os vetores singulares antes de somar | model merging | ViT-B/32 com 8 tarefas: 85,86 vs 70,79 (TA). A ortogonalização sozinha dá +3,6 a +7,4 pp. |
| EMR-Merging (Huang et al.) | 2024 | Eleger um vetor unificado (sinal majoritário, maior magnitude do mesmo sinal), depois máscara de sinal e reescalonador escalar por tarefa que restaura a massa L1 | model merging | ViT-B/32: 88,7 vs TA 70,1, AdaMerging++ 81,1 e MTL 88,9. Sem ajuste nem dados. Não generaliza para treino do zero (App. G). |
| Twin-Merging (Lu et al.) | 2024 | Especialista compartilhado (TA) mais resíduos exclusivos comprimidos por SVD e combinados por um roteador por entrada | model merging | RoBERTa/GLUE (normalized score): 96,14 vs TA 67,80. ViT-B/32: 95,33 vs AdaMerging 88,50. Roteador treinado em validação. |
| AdaMerging (Yang et al.) | 2024a | Coeficientes por tarefa ou por camada aprendidos por minimização de entropia em dados de teste não rotulados | model merging | ViT-B/32: TA 69,1 → 80,1 (por camada) e 81,1 (++). Transdutivo; a entropia só serve a cabeças de classificação. |
| DARE (Yu et al.) | 2024 | Descartar uma fração p do delta e reescalar os sobreviventes por 1/(1−p), antes de qualquer merge | model merging (LLMs) | Ganhos modestos e inconsistentes (ex.: +3,18% em GSM8K). Falha com deltas grandes (~0,03). A reescala é essencial (Sec. 4.4). |
| TALL masks / Consensus (Wang et al.) | 2024 | Máscara por tarefa onde \|τ_t\| ≥ \|τ_MTL − τ_t\|; remover pesos selfish e catastrophic | model merging, compressão | >99% da acurácia com máscaras. Consensus TA: +4,9 pp (ViT-L/14, 20 tarefas). Ganho quase nulo com 8 tarefas. |
| FedExP (Jhunjhunwala et al.) | 2023 | Passo adaptativo do servidor η_g = max{1, Σ‖Δᵢ‖² / (2M(‖Δ̄‖² + ε))} | FL (modelo global) | 1,4–2,1× menos rodadas até a acurácia-alvo (EMNIST, CIFAR, CINIC). Teoria supõe minimizador comum. |
| FedLAW (Li et al.) | 2023 | Aprender no servidor o fator γ (soma dos pesos) e os pesos λ num dataset proxy | FL (modelo global) | CIFAR-10, ResNet20, E = 3, IID: 68,04 → 78,88. γ aprendido ≈ 0,95; γ > 1 nunca estudado. Exige dataset proxy rotulado. |
| Git Re-Basin (Ainsworth et al.) | 2023 | Permutar as unidades ocultas de um modelo para alinhar ao outro, depois fazer a média | merging de modelos com inicialização independente | Barreira zero com ResNet20 larga em CIFAR-10; ImageNet: −67% de barreira, não zero. Sem experimentos de FL. |
| TIES-Merging (Yadav et al.) | 2023 | Trim, eleição de sinal e média disjunta | model merging | +1,7 pp (visão) e +2,3 pp (NLP) sobre o melhor baseline. Com sinal oráculo: 72,0 vs 73,1 (multitask). |
| Tangent-space FT (Ortiz-Jimenez et al.) | 2023 | Fine-tuning linearizado para aumentar o disentanglement | merging / edição | Adição: +3,4 a +5,8 pp. Não funciona a partir de inicialização aleatória. |
| FedGH (Zhang et al.) | 2023 | PCGrad no servidor entre deltas com cos < 0 | FL | +0,5 a +5,2 pp. Benefício ~nulo com K = 5. Uma seed. |
| GMA (Tenison et al.) | 2022 | Máscara suave: coordenadas com A_j < τ são escaladas por A_j (τ = 0,4) | FL (modelo global) | +1 a +3 pp (ex.: CIFAR-10 73,38 → 74,44; FEMNIST 90,28 → 93,02). Encolhe o passo; fraco com baixa participação. |
| Fisher merging (Matena & Raffel) | 2022 | Média por parâmetro ponderada pela Fisher diagonal; cabeças não fundidas | model merging | Supera a média isotrópica em todos os casos de ensembling. Mesma inicialização; quase sempre pares de modelos. |
| Model soups (Wortsman et al.) | 2022 | Média uniforme, gulosa ou aprendida de modelos fine-tuned a partir do mesmo pré-treino | merging (mesma tarefa) | ViT-G/14 (soup gulosa): 90,94% top-1 no ImageNet vs 90,72% do melhor modelo. Barreira de erro com LR alto. |
| GradVac (Wang et al.) | 2021a | Alinhar cada par a uma similaridade-alvo EMA por camada | MTL (NMT) | +0,4 a +1,4 BLEU. Execução única. |
| FedFV (Wang et al.) | 2021b | Projeção ordenada pela perda, conflitos externos, renormalização para a norma da média simples | FL (justiça) | CIFAR-10: pior 5% sobe de 19,84 para 32,24. A renormalização preserva o passo do FedAvg (não corrige diluição). |
| FedNNNN (Nagura et al.) | 2020 | Reescalar o update agregado por β·E/N, com momentum | FL | Até +5,4 pp. Benchmarks de brinquedo, execução única, preprint. |
| FedNova (Wang et al.) | 2020 | Normalizar Δᵢ por trabalho local; τ_eff escolhido à parte | FL | +6 a +9 pp em VGG-11/CIFAR-10. Não trata desacordo direcional. |
| SCAFFOLD (Karimireddy et al.) | 2020 | Variáveis de controle (c − cᵢ) e passo global η_g | FL | Teoria robusta à heterogeneidade. EMNIST: o mais rápido. |
| PCGrad (Yu et al.) | 2020 | Projeção no plano normal quando cos < 0 | MTL | NYUv2 e Meta-World. Os autores o veem como regularizador em MTL supervisionado (App. B). |
| OGD (Farajtabar et al.) | 2020 | Projetar o gradiente no complemento ortogonal dos gradientes de logit das tarefas anteriores | aprendizado contínuo | MNIST permutado/rotacionado/split, em pé de igualdade com A-GEM. Reduz a norma do passo. Todos os métodos falham com tarefas dissimilares. |
| CFL (Sattler et al.) | 2019 | Agrupar clientes pela bipartição de cosseno quando há incongruência | FL | CIFAR-10 com rótulos permutados: ~20% → ~60%. Incongruência sintética. |

Candidatos a estado da arte, segundo as avaliações: Iso-CTS, WUDI-Merging, EMR-Merging, Twin-Merging e AdaMerging em merging; FedAWA em ponderação de agregação; FedCALM em FL personalizado. Nenhum deles foi validado em FL entre domínios médicos.

Tendências no período 2023–2025:
- **Merging** migrou de heurísticas por coordenada (Yadav et al., 2023) para estrutura de subespaço por camada (Gargiulo et al., 2025; Marczak et al., 2025) e para objetivos funcionais sem dados (Cheng et al., 2025).
- **Em merging, a lacuna passou a ser atribuída também a escala e diluição**, não só a conflito de sinal: coeficientes aprendidos (Yang et al., 2024a), reescala de magnitude (Yu et al., 2024; Huang et al., 2024), espectro isotrópico (Marczak et al., 2025) e preservação do componente exclusivo (Lu et al., 2024). O survey destaca que DARE *"highlights the importance of further performing rescaling on sparse models"* (Yang et al., 2024b).
- **FL** passou de projeções globais (Zhang et al., 2023) para mitigação por camada voltada a FL personalizado (Zheng et al., 2025) e para pesos de agregação adaptativos (Shi et al., 2025b).
- **Em paralelo, coexistem duas respostas opostas ao desacordo:** encolher o passo (Li et al., 2023; Shi et al., 2025a; Tenison et al., 2022) ou amplificá-lo (Jhunjhunwala et al., 2023; Nagura et al., 2020). Nenhum artigo avaliado as compara sob o mesmo protocolo.

## 6. Críticas e resultados negativos

1. **Conflito não basta para haver dano.**
   - Os próprios proponentes do PCGrad escrevem que *"conflicting gradients are not detrimental on their own"* (Yu et al., 2020, Sec. 2.2), e FedFV repete a ideia (Wang et al., 2021b, Introdução).
   - Em MTL supervisionado, o PCGrad atua mais como regularizador do que como melhoria de otimização (Yu et al., 2020, App. B).
2. **A projeção raramente dispara com similaridades positivas ou nulas.** *"PCGrad is not effective for positive gradient similarities"* (Wang et al., 2021a).
3. **Benefício pequeno com poucos clientes.** FedGH não mostrou ganho com K = 5 (Zhang et al., 2023).
4. **Cancelamento zero ainda degrada.** Task vectors com suportes disjuntos, sem coordenadas compartilhadas, degradam sob task arithmetic (Wang et al., 2024). Task vectors quase ortogonais perdem ~20 pontos sob soma uniforme (Yang et al., 2024a).
5. **O cosseno não prevê a qualidade do merge.** Marczak et al. (2025) não encontraram correlação clara entre cosseno e ganho por tarefa (Sec. 3.2, Fig. 2b); o SAR correlacionou (ρ = 0,94). Gargiulo et al. (2025) também chamam o cosseno de *"coarse-grained"*.
6. **Sem pré-treino não há composição.**
   - A partir de inicialização aleatória, a adição de tarefas fica no acaso (ViT-B/32: 5,5 vs 5,3) (Ortiz-Jimenez et al., 2023).
   - EMR-Merging *"cannot be generalized to models trained from-scratch"* (Huang et al., 2024, App. G).
   - DARE falha quando os deltas passam de ~0,002, e após pré-treino continuado eles chegam a ~0,03 (Yu et al., 2024, Sec. 1 e 4.6).
   - A conectividade linear de modos está ausente no início do treino e em modelos estreitos (Ainsworth et al., 2023). Há barreira de erro entre modelos fine-tuned com LR alto (Wortsman et al., 2022, Sec. 3.3).
7. **Forçar ortogonalidade piora.** A ablação CosReg, que empurra todos os cossenos para 0, fica atrás do PCGrad (Yu et al., 2020).
8. **Extrapolar não corrige direção** e pode elevar o piso de erro e fazer o iterado oscilar (Jhunjhunwala et al., 2023). FedNNNN amplifica uma direção média ruidosa quando há conflito real (Nagura et al., 2020).
9. **Métodos de projeção, máscara e ponderação convexa preservam ou reduzem a norma.**
   - FedGH e FedFV não recuperam passo perdido na média (Zhang et al., 2023; Wang et al., 2021b).
   - OGD satisfaz ‖g̃‖ ≤ ‖g‖ por construção, e o artigo não mede essa perda (Farajtabar et al., 2020).
   - GMA encolhe as coordenadas sem consenso. Com baixa participação (2 de 10 clientes), a maioria dos parâmetros quase não se move (Tenison et al., 2022).
   - FedAWA reponderá dentro de uma combinação convexa e, por construção, não restaura norma. Rebaixar clientes desalinhados do consenso pode marginalizar domínios minoritários. Esta última é observação da avaliação, não dos autores (Shi et al., 2025b).
10. **Encolher funciona por regularização, e só em parte.** Em FedLWS, escalar só o termo de otimização (only-opt, equivalente a um LR de servidor menor) perde ~2,8 pp em média frente ao método completo (Shi et al., 2025a, Tabela 10). Em FedLAW, γ < 1 fixo atrapalha no fim do treino (Li et al., 2023).
11. **Todos os métodos falham com tarefas dissimilares.** Em OGD, *"fail[s] considerably when the tasks are dissimilar (for example rotations larger than 90 degrees)"* (Farajtabar et al., 2020, Sec. 6).
12. **Evidência central apoiada em procedimentos frágeis ou com dados privilegiados.**
    - A ligação entre τ e o γ ótimo de FedLWS se apoia em um Pearson de 0,86 com γ ajustado **no conjunto de teste**, para demonstração (Shi et al., 2025a, Fig. 1a).
    - FedLAW exige dataset proxy rotulado no servidor (Li et al., 2023).
    - AdaMerging usa dados de teste não rotulados no momento do merge (Yang et al., 2024a).
    - Twin-Merging treina o roteador em validação, e Iso-C/Iso-CTS ajustam α em validação (Lu et al., 2024; Marczak et al., 2025).
    - A teoria de GMA é declarada apenas como sanity check (Tenison et al., 2022).
13. **Fragilidade experimental recorrente:**
    - seed única (Zhang et al., 2023; Nagura et al., 2020; Wang et al., 2021a), resultados únicos sem estatística inferencial (Huang et al., 2024) e tabelas principais sem número de seeds nem intervalos (Shi et al., 2025a);
    - 3 seeds sem variância (Zheng et al., 2025);
    - heterogeneidade apenas por rótulo, sem shift de domínio, em quase todos os trabalhos de FL avaliados (Li et al., 2023; Shi et al., 2025a, 2025b; Tenison et al., 2022). A exceção parcial é uma tabela de dígitos multi-domínio em FedAWA, em que dois dos quatro domínios pioram (Shi et al., 2025b, App. A.4);
    - os trabalhos de merging usam só imagens naturais e NLP, com especialistas fine-tuned a partir do mesmo modelo pré-treinado;
    - nenhum trabalho cobre segmentação médica entre modalidades.
14. **Definições apenas operacionais.**
    - "Interference" é declarada sem definição formal estabelecida (Gargiulo et al., 2025, App. E.2). A ligação ortogonalidade → baixa interferência é especulativa (Ilharco et al., 2023).
    - DARE, AdaMerging e Twin-Merging nomeiam a interferência, mas só a inferem por queda de desempenho (Yu et al., 2024; Yang et al., 2024a; Lu et al., 2024).
    - O survey reconhece que os métodos *"are largely empirical and lack sufficient theoretical guarantees"* (§6) e que a teoria cobre mal tarefas e modalidades heterogêneas (§3.1) (Yang et al., 2024b).
    - As exceções com definição formal (Cheng et al., 2025; Marczak et al., 2025) dependem de aproximações: passos pequenos e poucos, entradas quase fixas, só camadas lineares ou matrizes 2D.

## 7. O que isso implica para o nosso diagnóstico

Cautela geral: as conclusões abaixo são hipóteses de trabalho. Os p-valores do projeto são exploratórios (`exploratory_only_non_independent_client_fold_pairs`). Quase toda a evidência da literatura vem de classificação com skew de rótulo ou de merging único a partir de pré-treino, não de FedPer multi-domínio e multi-tarefa treinado do zero.

1. **Nomenclatura.**
   - A razão do projeto deve ser citada como o 1 − N/E de Nagura et al. (2020).
   - O "excesso sobre a referência ortogonal" é, com sinal equivalente, o conflito de agregação de Zheng et al. (2025) (derivação nossa).
   - A Eq. 6 de Li et al. (2023) permite apresentar a matriz de cossenos e o CR como uma só identidade: o termo cruzado ponderado pelos cossenos é o que separa ‖Σwᵢ Δᵢ‖² da referência ortogonal (identificação nossa). "Coerência" (Li et al., 2023) é o nome estabelecido para o cosseno médio ponderado.
   - Baixo excesso entre datasets significa pouco **conflito** no sentido da literatura. Não significa pouca **perda de norma** nem pouca **interferência funcional** (Cheng et al., 2025; Marczak et al., 2025).
2. **Diluição é um mecanismo distinto de conflito.**
   - Com updates quase ortogonais, a componente de cada cliente no agregado é wᵢΔᵢ. O passo efetivo de cada cliente é escalado por wᵢ, sem cancelamento algum.
   - Esse regime é exatamente o que Jhunjhunwala et al. (2023) tratam com η_g ≈ M/2. É também o regime que Zheng et al. (2025) consideram inócuo (G ≈ 0), o que marca uma lacuna entre as duas linhas. Em Li et al. (2023), corresponde à coerência ≈ 0 depois do "critical point", quando os ganhos se tornam marginais.
   - O merging oferece um paralelo independente. Task vectors quase ortogonais ainda perdem ~20 pontos sob soma uniforme, e coeficientes por tarefa e por camada recuperam boa parte disso (Yang et al., 2024a). O ótimo de Task Arithmetic com 8 tarefas é λ = 0,3, cerca de 2,4× o 1/8 da média simples (razão calculada por nós). EMR-Merging conserva a magnitude máxima, não a média, e reescala por tarefa (Huang et al., 2024). Twin-Merging atribui a lacuna à diluição do conhecimento exclusivo (Lu et al., 2024).
   - O achado de passo agregado ~10× menor com pouco conflito é compatível com essa leitura. Isso não demonstra que a diluição seja a causa de federado < local-only. Os paralelos de merging vêm de merge único a partir de pré-treino, não de rodadas iterativas do zero.
3. **Verificação de consistência sugerida (derivação nossa).**
   - Para updates ortogonais, ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖ ≈ 1/√N_eff com normas iguais, onde N_eff = 1/Σwᵢ². Li et al. (2023) e Wortsman et al. (2022) apoiam a premissa de quase-ortogonalidade de updates heterogêneos de vários passos.
   - Com poucos clientes efetivos, essa razão fica bem acima de 0,1. Um encolhimento de 10× do passo agregado em relação ao passo dos clientes precisa então ser decomposto em três parcelas: ortogonalidade, pesos, e diferença de τ entre modos steps/epochs (Wang et al., 2020).
   - Se o encolhimento observado superar a referência ortogonal, há conflito residual, possivelmente intra-dataset, que precisa ser localizado.
4. **O cosseno par a par tem padrão na literatura.** Podem ser reportados a partir dos logs existentes:
   - fração de pares com cos < 0 e cosseno mínimo, separados em intra e inter (Zhang et al., 2023);
   - coerência ponderada por rodada, com a rodada do "critical point" (Li et al., 2023);
   - separation gap (Sattler et al., 2019);
   - leitura por bloco e por rodada (Wang et al., 2021a; Zheng et al., 2025);
   - Φ por par, para checar a tragic triad (Yu et al., 2020).
   - Ressalva: o cosseno sozinho não previu a qualidade do merge em Marczak et al. (2025), então não deve ser a única sonda.
5. **Sondas adicionais baratas, a partir dos deltas que o servidor já recebe.** O projeto não persiste o trunk pós-fit de cada cliente, então elas precisam ser calculadas no servidor durante a rodada, como já ocorre com o cosseno.
   - **Diluição por cliente:** o reescalonador λᵢ = Σ\|Δᵢ\| / Σ\|Mᵢ ⊙ Δ_agg\| e a concordância de sinal Mᵢ entre cada delta e o agregado (Huang et al., 2024). Com Mᵢ alto e λᵢ ≈ 10, o achado seria diluição, não cancelamento (leitura nossa).
   - **Interferência funcional sem dados:** ‖(Δ_agg − Δᵢ)Δᵢᵀ‖²_F / ‖Δᵢ‖²_F por camada, com convoluções reorganizadas como out × in·k·k (Cheng et al., 2025). Pode ser decomposta num termo de encolhimento ao longo de Δᵢ e num termo cruzado entre clientes (decomposição nossa).
   - **Sub-representação:** SAR por cliente e por bloco do trunk, separado por dataset (Marczak et al., 2025).
   - **Concordância de sinal por coordenada:** distribuição de A_j, intra e inter (Tenison et al., 2022). Para updates quase ortogonais em alta dimensão, A_j baixo é o análogo, em sinal, da referência ortogonal.
   - **Dispersão por bloco:** τ_l (Shi et al., 2025a), ao lado dos cossenos por bloco.
   - **Sonda de caminho:** perda de cada cliente no segmento entre seu trunk local e o trunk agregado, ao estilo de loss barrier ou interpolation advantage (Ainsworth et al., 2023; Wortsman et al., 2022). Mostraria se o passo diluído cai numa região plana ou atrás de uma barreira.
6. **Remédios que devem ser testados como ablações, não assumidos como solução:**
   - Passo do servidor sobre o trunk:
     - FedExP com versão ponderada Σwᵢ‖Δᵢ‖² / ‖Σwᵢ Δᵢ‖², que é extensão nossa (Jhunjhunwala et al., 2023);
     - reescala β·E/N (Nagura et al., 2020);
     - η_g > 1 (Karimireddy et al., 2020), ou γ > 1 no esquema de Li et al. (2023), que o artigo permite mas nunca estudou.
   - O η_g da FedExP pode ser recalculado offline a partir de aggregation_history.json.
   - Variantes inspiradas em merging, todas sem validação em FL: coeficientes de agregação por cliente e por camada, à moda do AdaMerging por camada (Yang et al., 2024a; a entropia só serviria às cabeças de classificação); ponderação por Fisher diagonal (Matena & Raffel, 2022); achatamento do espectro por camada de Σwᵢ Δᵢ, à moda do Iso-C (Marczak et al., 2025); e uma solução WUDI por camada sobre os deltas (Cheng et al., 2025).
   - Se o reescalonamento fechar o gap com local-only, a hipótese de diluição ganha força. Se não fechar ou desestabilizar, pesam mais a heterogeneidade de domínio e a interferência funcional.
   - A premissa de minimizador comum da FedExP é duvidosa entre quatro modalidades.
   - A comparação precisa manter o mesmo modo de budget, porque as normas Δᵢ dependem dele (Wang et al., 2020).
7. **Remédios provavelmente contraindicados para o diagnóstico de diluição.**
   - FedLWS, GMA e FedLAW com γ < 1 encolhem o passo quando os clientes discordam (Shi et al., 2025a; Tenison et al., 2022; Li et al., 2023). Num regime em que o passo agregado já é ~10× menor, tendem a agravar a diluição. Seus ganhos vêm de classificação com skew de rótulo e modelo global único, e com pré-treino γ ≈ 0,99 (Shi et al., 2025a).
   - FedAWA não altera a norma do passo e pode rebaixar o dataset minoritário. Se for testado, deve ficar dentro de cada dataset no modo hierárquico (Shi et al., 2025b).
   - Projeções (PCGrad, FedGH, FedFV, OGD) só reduzem ou preservam a norma (item 8).
   - Essas expectativas são inferências nossas e só se confirmam com ablação pareada.
8. **Gradient surgery provavelmente rende pouco aqui.** PCGrad, FedGH e FedFV só agem com cos < 0, não restauram norma e mostraram pouco ganho com poucos clientes (Yu et al., 2020; Zhang et al., 2023; Wang et al., 2021b). FedCALM pode ajudar, mas parte do ganho pode vir de um passo efetivamente maior, e isso exige controle (Zheng et al., 2025).
9. **Baixo cancelamento não implica ausência de interferência.**
   - Suportes disjuntos ainda degradam (Wang et al., 2024), e um cliente pode ficar sub-representado no subespaço dominante do agregado mesmo com cossenos baixos (Marczak et al., 2025).
   - Composição funcional depende de pré-treino (Ortiz-Jimenez et al., 2023; Huang et al., 2024; Wortsman et al., 2022). Isso reforça o valor do braço E7 (ResNet-34 ImageNet) como teste, frente ao MTnnUNet treinado do zero.
   - Matena & Raffel (2022) apontam que fundir só o corpo, deixando as cabeças sem merge, causa *"a distribution shift in the classification head inputs"* (Sec. 2.4). Num desenho FedPer, esse é um mecanismo plausível para federado < local-only, independente de cancelamento (hipótese nossa).
   - Uma sonda funcional ao estilo da Eq. 6 de Ortiz-Jimenez et al. (2023), ou do erro relativo de Cheng et al. (2025), complementaria a medida geométrica: avaliar o cliente t com o trunk global versus trunk + Δ_t apenas.
10. **Profundidade e forma do compartilhamento.**
    - Zheng et al. (2025) associam corpos compartilhados mais profundos a mais conflito, de forma correlacional. Isso é relevante para o braço E6 (wide sharing), sem permitir extrapolação direta.
    - Twin-Merging e EMR-Merging sugerem federar só o componente compartilhado e personalizar o exclusivo dentro do próprio trunk, com um resíduo de posto baixo ou uma máscara mais um escalar por cliente (Lu et al., 2024; Huang et al., 2024). É uma extensão natural do FedPer, mas EMR não se aplica a treino do zero, então precisaria ser validada.
11. **Clustering por dataset** (Sattler et al., 2019) é um baseline natural contra um trunk único para todos os datasets.
12. **Alinhamento por permutação provavelmente é desnecessário.** Git Re-Basin trata modelos de inicialização independente (Ainsworth et al., 2023). No FedPer, todos os clientes partem do mesmo trunk global a cada rodada e dão poucos passos locais, o que os mantém na mesma bacia e com a mesma ordem de unidades (inferência nossa). Isso é compatível com o baixo cancelamento entre datasets observado.

## 8. Lacunas abertas

1. Não existe métrica consolidada que **separe diluição (ortogonalidade + pesos) de conflito**. Nagura et al. (2020) não usam referência ortogonal, Zheng et al. (2025) tratam o caso ortogonal como inócuo, e τ de Shi et al. (2025a) mistura os dois. Há candidatas parciais em merging (λᵢ de Huang et al., 2024; proxy de Cheng et al., 2025; SAR de Marczak et al., 2025), nenhuma testada em FL. A decomposição CR = referência ortogonal + excesso, como o projeto já faz, é uma contribuição possível.
2. Nenhum dos 30 trabalhos avaliados estuda cancelamento ou conflito em **FL entre domínios de imagem médica** nem em **multi-tarefa seg+cls com FedPer**.
3. A ligação entre medidas geométricas (cosseno, CR, STI) e interferência **funcional** só começou a ser medida, e apenas em merging único: SAR prevê o ganho e o cosseno não (Marczak et al., 2025), e o erro relativo cresce com a profundidade (Cheng et al., 2025). Para rodadas iterativas de FL, segue não estabelecida (Ortiz-Jimenez et al., 2023; Gargiulo et al., 2025).
4. Extrapolação do passo sem minimizador comum, num regime de heterogeneidade de domínio, não tem garantia (Jhunjhunwala et al., 2023).
5. Não há versão **ponderada ou hierárquica** de FedExP nem de FedNNNN. Ambas usam pesos uniformes ou n_k/n. FedLAW permite γ > 1, mas nunca o estudou (Li et al., 2023).
6. Faltam medidas por camada da perda de norma ao longo das rodadas de FL. STI e SAR são por camada, mas em merging único (Gargiulo et al., 2025; Marczak et al., 2025). FedLWS reporta γ_l e τ_l por camada, mas não a razão de normas (Shi et al., 2025a). O gap de FedCALM não é reportado empiricamente (Zheng et al., 2025).
7. A maioria das definições é para gradientes de um passo ou task vectors de merge único. Falta teoria para **deltas de vários passos em agregação iterativa** com partes personalizadas. A teoria de merging supõe inicialização pré-treinada comum, redes largas ou linearização, o que deixa de fora um trunk treinado do zero entre modalidades (Yang et al., 2024b, §3.1; Huang et al., 2024; Yu et al., 2024).
8. Controles ausentes na literatura:
   - métodos de direção, como FedCALM, não são comparados com baselines de passo do servidor, então não se sabe quanto do ganho é correção de diluição;
   - as linhas que encolhem o passo (FedLAW, FedLWS, GMA) e as que o amplificam (FedExP, FedNNNN) nunca são comparadas sob o mesmo budget. A única ablação parcial é o only-opt de FedLWS (Shi et al., 2025a).
9. O estado da arte de merging (Iso-CTS, WUDI, EMR, Twin, AdaMerging) não foi validado em FL iterativo. Há também obstáculos técnicos: WUDI só trata camadas lineares, e os métodos espectrais exigem reorganizar convoluções em matrizes 2D (Cheng et al., 2025; Marczak et al., 2025).
10. O corpus não inclui trabalhos de 2026. Avanços recentes, sobretudo em FL, podem não estar refletidos aqui.

---

Fontes: [`fontes.md`](fontes.md). Fichas de leitura por artigo: [`fichas.md`](fichas.md).
