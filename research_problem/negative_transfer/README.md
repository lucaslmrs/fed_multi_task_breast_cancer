# Negative transfer: síntese da literatura

> Escopo: esta síntese reúne **32 avaliações de artigos feitas em duas passagens**. A primeira passagem trouxe 15 fontes (1–15) e a segunda, 17 novas (16–32). A segunda passagem entregou 18 fichas, mas duas eram do mesmo texto (Perkins & Salomon, 1992) e foram fundidas. Só uma avaliação é `abstract_only`: (Zhang et al., 2026, FedGain), lida pelo resumo e pelo pôster oficial do ICML. As afirmações tiradas dela levam a marca **"[só resumo]"**. Todas as outras foram feitas com texto completo.
>
> Confiança média, com citações a conferir no PDF antes de entrar na dissertação: (Lu et al., 2024), cujas citações vieram de uma ferramenta de busca que resume o conteúdo; (Zhang et al., 2026) [só resumo]; e (Liu et al., 2026, OmniFM), com trechos encurtados pela ferramenta e inconsistências numéricas internas. A ferramenta de HTML também encurtou citações de (Senushkin et al., 2023) e (Liu et al., 2023). Em (Mueller et al., 2025), os símbolos entre colchetes foram reconstruídos.
>
> Ressalvas de venue: TMLR para (Mueller et al., 2025) e CVPR para (Senushkin et al., 2023) não aparecem no texto do arXiv e não foram conferidos. (Arivazhagan et al., 2019) e (Chen et al., 2023) são preprints sem revisão por pares. As páginas da edição impressa de (Perkins & Salomon, 1992) não foram conferidas: o texto lido é o manuscrito de 1992, e a enciclopédia impressa costuma ser datada de 1994.
>
> Desambiguação de autores:
> - **(Yu et al., 2020a)** é Tao Yu et al., *Salvaging FL by Local Adaptation*. **(Yu et al., 2020b)** é Tianhe Yu et al., *PCGrad*.
> - **(Li et al., 2021a)** é Xiaoxiao Li et al., *FedBN*. **(Li et al., 2021b)** é Tian Li et al., *Ditto*.
> - **(Liu et al., 2021)** é CAGrad, **(Liu et al., 2023)** é FAMO (os dois de Bo Liu et al.) e **(Liu et al., 2026)** é OmniFM (Meilin Liu et al.).
> - **(Zhang et al., 2023)** é o survey de negative transfer. **(Zhang et al., 2026)** é FedGain.

---

## 1. Resumo em 5 linhas

1. *Negative transfer* (NT) é um conceito **de desfecho** desde a origem na psicologia da educação: aprender num contexto "undermines" o desempenho em outro (Perkins & Salomon, 1992). Em ML, o desempenho com outra fonte de dados ou outra tarefa fica abaixo do **mesmo algoritmo treinado sem ela** (Rosenstein et al., 2005; Wang et al., 2019; Zhang et al., 2023; Jiang et al., 2023; Mueller et al., 2025).
2. Em aprendizado federado (FL), o termo vira "o modelo federado é pior que o modelo treinado só com dados locais" (Bao et al., 2023; Yu et al., 2020a). Um trabalho recente chama isso de "negative gain" e conta a fração de clientes afetados (Zhang et al., 2026) [só resumo]. É exatamente a comparação federado × local-only do projeto.
3. "Conflito de gradientes" é uma **hipótese de mecanismo** com duas componentes: direção (cosseno < 0) e magnitude (dominância ou desbalanço de normas) (Yu et al., 2020b; Liu et al., 2021; Navon et al., 2022; Senushkin et al., 2023). Dois estudos empíricos mostram que conflito e NT se desacoplam: conflito piora a otimização, mas **não prevê** a generalização (Mueller et al., 2025; Jiang et al., 2023).
4. O SOTA de MTL centralizado em 2022–2025 (Nash-MTL, Aligned-MTL, FAMO, ForkMerge, atualização seletiva por grupos) ainda deixa Δm negativo em vários benchmarks. O SOTA de FL multitarefa e de FL médico cross-modality mede NT como Δ% contra o local, sem medir o conflito entre clientes, ou apenas o define sem medir (Yang et al., 2024; Lu et al., 2024; Chen et al., 2023; Liu et al., 2026).
5. Críticas fortes mostram que métodos "anticonflito" raramente superam uma ponderação fixa bem ajustada (Xin et al., 2022; Kurin et al., 2022). O conflito também não prevê a transferência (Mueller et al., 2025). Isso sustenta procurar a causa da NT do projeto em otimização e magnitude (diluição do passo), e não em conflito direcional.

---

## 2. Como a literatura define o termo

### 2.1 Linha do tempo

**1901/1992: origem na psicologia da educação (Perkins & Salomon, 1992).** O verbete de enciclopédia é a fonte mais antiga entre as avaliações e dá a definição canônica de transferência positiva e negativa:
> "Transfer of learning occurs when learning in one context enhances (positive transfer) or undermines (negative transfer) a related performance in another context." (Abstract, manuscrito p. 2)
> "Negative transfer occurs when learning in one context impacts negatively on performance in another. [...] Negative transfer typically causes trouble only in the early stages of learning a new domain. With experience, learners correct for the effects of negative transfer." (Sec. 1, "Positive versus negative transfer", manuscrito pp. 3–4)

A definição é contrastiva, o que exige um baseline sem a aprendizagem-fonte: "talk of transfer is always at least implicitly contrastive" (p. 3). Os autores separam três coisas: transferência **nula** (a maior parte da evidência histórica, desde Thorndike & Woodworth, 1901), transferência **negativa** (dano) e transferência *near* × *far*. Sobre esta última, avisam que "near" e "far" "do not imply any strictly defined metric of 'closeness'" (p. 4). O artigo de Thorndike & Woodworth não foi lido. Ele é citado aqui só pelo resumo de Perkins & Salomon.

**2005: origem operacional em ML (Rosenstein et al., 2005).** Entre as avaliações, este workshop do NIPS é o uso mais antigo do termo em ML. O trabalho de 2019 trata essa formulação como "widely accepted" (Wang et al., 2019). Essas fontes não bastam para afirmar que Rosenstein *cunhou* o termo em ML, que já existia na psicologia (Perkins & Salomon, 1992). O que dá para dizer é que ele está entre as formulações fundadoras em ML.
> "transfer learning may actually hinder performance if the tasks are too dissimilar. [...] one challenge for transfer learning research is to develop approaches that detect and avoid negative transfer using very little data from the target task." (Abstract, p. 1)

A operacionalização compara o modelo com transferência a um baseline "B-only" (só a tarefa-alvo) por teste de significância (p<0,05):
> "Black and gray respectively denote positive and negative transfer, and white indicates no statistically significant difference." (legenda da Fig. 2, p. 4)

**2010: base teórica sem usar o termo (Ben-David et al., 2010).** O artigo não diz "negative transfer". Ele descreve uma transição de fase a partir da qual dados da fonte prejudicam o alvo:
> "if there are enough target data (specifically, if mT ≥ D^2 = d/A^2) then no source data are needed, and in fact using any source data will yield suboptimal performance." (Sec. 6, p. 161)

**2018: visão de escala e ponderação (Kendall et al., 2018, CVPR).** O artigo não usa "negative transfer", "interference" nem "conflict". Ele atribui a queda do modelo multitarefa abaixo do single-task à **ponderação das perdas**, que depende da escala de medida e do ruído de cada tarefa:
> "model performance is extremely sensitive to weight selection, w_i, as illustrated in Figure 2." (Sec. 3)
> "Using a uniform weighting results in poor performance, in some cases not even improving on the results from the single task model." (Sec. 5.1)

**2019: formalização (Wang et al., 2019, CVPR).** Parte da descrição informal ("transferring knowledge from the source can have a negative impact on the target learner", Sec. 3) e propõe:
> "we define the negative transfer condition (NTC) for any algorithm A as R_{P_T}(A(S,T)) > R_{P_T}(A(∅,T)). [...] we also define the negative transfer gap (NTG) [...] and we say that negative transfer occurs if the negative transfer gap is positive" (Sec. 3, Eqs. 3–4)

O ponto central é que a NT é **relativa a um algoritmo fixo**. Os autores rejeitam duas alternativas: comparar com o *melhor* algoritmo target-only, que não é mensurável, e a versão independente de algoritmo, que nunca se verifica.

**2019: "invariância pode prejudicar" em adaptação de domínio (Zhao et al., 2019, ICML).** O artigo não usa a expressão "negative transfer" (busca no texto completo). Ele mostra, por cotas de risco, que alinhar features e reduzir o erro na fonte pode **aumentar** o erro no alvo quando as distribuições de rótulos ou as funções de rotulação diferem:
> "if the marginal label distributions are significantly different between source and target domains, then minimizing dJS(D^Z_S, D^Z_T) and the source error εS(h∘g) will only increase the target error." (Remark após o Teorema 4.3, Sec. 4.3, p. 11)

**2019: análogo federado em nível de objetivo (Sattler et al., 2019).** O artigo não usa "negative transfer". Ele define clientes **incongruentes** como aqueles para os quais não existe um θ* que seja ótimo local para todos ao mesmo tempo:
> "we will call a set of clients and their data generating distributions φ congruent [...] if they satisfy Assumption 1 and incongruent if they don't." (Sec. I)

**2019: FedPer, o esquema do projeto (Arivazhagan et al., 2019).** O artigo não usa "negative transfer", "interference", "conflict" nem "cancellation". Ele fala nos "ill-effects of statistical heterogeneity" e justifica a divisão base + personalização pela analogia com MTL:
> "Intuitively, the base layers can act as the shared layers in a multi-task learning model and capture the wisdom of the crowd, while the personalization layers can act as the task specific layers of the same multi-task learning model to capture user specific aspects." (Sec. 2)

**2020: MTL (Standley et al., 2020, ICML).**
> "the quality of predictions are often observed to suffer when a network is tasked with making multiple predictions due to a phenomenon called negative transfer. In fact, multi-task performance can suffer so much that smaller independent networks are often superior." (Sec. 1, p. 1)

As causas aparecem como hipóteses: taxas de aprendizado diferentes, dominância de uma tarefa, interferência de gradientes e uma paisagem de otimização mais difícil. A medida é a variação relativa da perda com **custo computacional igualado**.

**2020: conflito de gradientes, que não é a mesma coisa que NT (Yu et al., 2020b, NeurIPS).** O artigo do PCGrad não usa "negative transfer".
> "We define ϕ_ij as the angle between two task gradients g_i and g_j. We define the gradients as conflicting when cos ϕ_ij < 0." (Sec. 2.2, Def. 1)
> "conflicting gradients are not detrimental on their own. Indeed, simply averaging task gradients should provide the correct solution to descend the multi-task objective." (Sec. 2.2)

**2020: FL, critério de incentivo (Yu et al., 2020a).** Também não usa o termo.
> "many participants gain no benefit from FL because the federated model is less accurate on their data than the models they can train locally on their own." (Abstract)

**2021: conflito em relação à direção da atualização (Liu et al., 2021, CAGrad, NeurIPS).** O artigo não usa "negative transfer" nem "interference". Ele define o conflito contra o gradiente médio, não entre pares, e decompõe a causa em escala e direção:
> "gradients from different tasks 1) may have varying scales with the largest gradient dominating the update, and 2) may point in different directions so that directly optimizing the average loss can be quite detrimental to a specific task's performance." (Sec. 1)
> "However, $g_{0}$ may conflict with individual gradients, i.e. $\exists~~i,~\langle g_{i},g_{0}\rangle<0$. When this conflict is large, following $g_{0}$ will decrease the performance on task $i$." (Sec. 2.1)

**2021: afinidade entre tarefas por *lookahead* (Fifty et al., 2021, NeurIPS).** Não usa "negative transfer". O sinal negativo da afinidade indica dano:
> "a negative value of [Z] indicates that the shared parameter update is antagonistic for this task's performance." (Sec. 4.1, Eq. 1)

**2021: FL personalizado sob heterogeneidade (Collins et al., 2021; Li et al., 2021a; Li et al., 2021b).** Nenhum dos três usa "negative transfer", "interference", "conflict" ou "cancellation". Os três descrevem o mesmo desfecho: o modelo compartilhado fica abaixo do treino local.
- FedRep: "for some clients, it might be better to simply use their own local data (even if it is small) to train a local model" (Collins et al., 2021, Sec. 1).
- FedBN define *feature shift* como "1) covariate shift: the marginal distributions P_i(x) varies across clients, even if P_i(y|x) is the same for all client; and 2) concept shift: the conditional distribution P_i(x|y) varies across clients and P(y) is the same" (Li et al., 2021a, Sec. 3). Também observa que as estratégias de FL são "even worse than using single client data for training for most of the clients" (Sec. 5.2).
- Ditto modela a heterogeneidade como parentesco τ entre dispositivos: "When $1/\tau$ is small (less related), local outperforms global; when $1/\tau$ is large (more related), global is better than local." (Li et al., 2021b, legenda da Fig. 4)

**2021/2023: survey (Zhang et al., 2023).** Consolida a definição pelo NTG:
> "NT happens when ϵ𝒯(θ(𝒮,𝒯)) > ϵ𝒯(θ(∅,𝒯))" ... "Obviously, NT occurs if the NTG is positive." (Sec. II-C, Def. 1)
> "conflicting gradients among different tasks may induce NT (also known as negative interference)." (Sec. VIII-A)

Ressalva de venue: a avaliação leu a versão arXiv v4 (2021). O venue IEEE/CAA JAS 2023 veio da memória do avaliador e não foi conferido.

**2022: conflito como direção + dominância por magnitude (Navon et al., 2022, Nash-MTL, ICML).** A expressão "negative transfer" não aparece no artigo (0 ocorrências). A degradação é atribuída ao conflito, com duas componentes:
> "These per-task gradients may have conflicting directions or a large difference in magnitudes, with the largest gradient dominating the update direction." (Sec. 1)
> "If it is positive there is a positive interaction and the other gradients aid the i'th task, and if it is negative they hamper it." (Sec. 3.1, após a Eq. 2, sobre o termo (Σ_{j≠i} α_j g_j)^T g_i)

**2023: NT fraca × forte (Jiang et al., 2023, ForkMerge, NeurIPS).** É a definição de desfecho mais refinada entre as avaliações:
> "Occasionally, learning multiple tasks simultaneously results in lower accuracy than learning only the target task, which is known as negative transfer." (Abstract)
> "For some ATL algorithm A with weighting hyper-parameter λ, weak negative transfer occurs if TG(λ, A) < 0." (Sec. 3, Def. 3.2)
> "For some ATL algorithm A, strong negative transfer occurs if max_{λ>0} TG(λ, A) < 0." (Sec. 3, Def. 3.3)
> "Negative transfer is not necessarily caused by gradient conflicts and gradient conflicts do not necessarily lead to negative transfer." (Sec. 3.1, Finding 1)

**2023: conflito como progresso negativo de uma tarefa (Liu et al., 2023, FAMO, NeurIPS).** O artigo não usa "negative transfer" como termo formal. O conflito é definido pelo aumento da perda de alguma tarefa no passo, e só é danoso quando persiste. As citações abaixo são fragmentos curtos, por limite da ferramenta:
> CG ocorre quando "∃i, ℓ_i(θ_{t+1}) − ℓ_i(θ_t) ≈ −α∇ℓ_i(θ_t)^T d_t > 0", isto é, "certain task's loss is increasing." O conflito é "not inherently detrimental", mas se torna "undesirable when a subset of tasks persistently undergoes under-optimization due to CG." (Sec. 2, "Optimization Challenge")

O artigo também define um "generalized CG": todas as perdas caem, mas algumas muito mais devagar que as outras.

**2023: estabilidade pelo número de condição (Senushkin et al., 2023, Aligned-MTL).** O termo "negative transfer" aparece só de passagem, citando trabalhos anteriores. As citações devem ser conferidas no PDF porque a ferramenta as encurtou:
> "Conflicting gradients having opposing directions often induce a negative transfer." (Sec. 2)
> "The stability criterion reaches its minimum iff both issues are solved jointly and gradients are orthogonal." (Sec. 4.2)

**2023: FL, "negative transfer problem" (Bao et al., 2023).**
> "the global FL model may even perform worse than the models trained with local data only." (Abstract)
> "the global model significantly degrades, and even performs worse than local training [...] which we refer to as the negative transfer problem." (Sec. 3.1; o trecho do meio não foi reproduzido literalmente pela ferramenta)

**2023: FMTL com conflito entre clientes (Chen et al., 2023, FedBone, preprint).** Adota a definição do PCGrad no nível do cliente:
> "Define the angle between client i graidents ∇_i and client j gradients ∇_j as ω_ij, ∇_i and ∇_j are conflicting gradients when cos ω_ij < 0." (Sec. 3.4, Def. 1; erro de grafia do original)
> "direct aggregation will cause a reduction in final gradients." (Sec. 3.4)

O artigo afirma essa "redução" mas não a mede.

**2024: teoria em regressão de alta dimensão (Tahir et al., 2024, preprint).**
> "We refer to the situation 𝒯<0 as negative transfer, since pretraining leads to degradation of the generalization error." (Sec. 2)

**2024: FMTL, sem o termo (Yang et al., 2024; Lu et al., 2024).** Nenhum dos dois usa "negative transfer". Eles falam em "gradient conflict", "task heterogeneity" (Yang et al., 2024) e "model incongruity" / "conflicting parameter updates" (Lu et al., 2024). Na prática, a NT aparece como um Δ% ou Δm negativo em relação ao treino local.

**2025: SOTA de MTO (Jeong & Yoon, 2025, ICLR).**
> "training all tasks together in a single architecture can lead to performance degradation, known as negative transfer" (Abstract)

A NT fica informal. O que se formaliza é o instrumento de medida, a afinidade proximal B por *lookahead* (Sec. 3.2, Def. 2).

**2025: transferência como desfecho, conflito como fator de otimização (Mueller et al., 2025, TMLR, venue não conferido).**
> "The UMTG is thought to frequently result in poor performance for multi-task learning, often yielding worse generalization than single-task learning, a phenomenon named negative transfer (Zhang et al., 2023)." (Sec. 2.1)
> "directional conflict occurs when the angle between task gradients is high [...] separately, magnitude conflict can arise when the magnitude of task gradients are disparate, resulting in the under-optimization of certain tasks" (Sec. 2.1)

**2026: "negative gain" em FL (Zhang et al., 2026, FedGain, ICML) [só resumo].**
> "Data heterogeneity is a fundamental challenge in Federated Learning (FL), where induced model drift often results in "negative gains" for global models on data-abundant clients, with performance falling below that of local training." (Abstract) [só resumo]

**2026: FL médico cross-modality (Liu et al., 2026, OmniFM, CVPR).** Não usa "negative transfer", "interference", "gradient conflict" nem "cancellation". O fenômeno aparece como deriva qualitativa entre bacias de perda, sem medida:
> "Consequently, local models optimize against diverging loss landscapes; during aggregation [...] the global model is pulled toward conflicting minima." (§1; trecho encurtado pela ferramenta)

### 2.2 Tabela comparativa

| Autor | Ano | Termo | Definição (resumo) | Nível | Como mede |
|---|---|---|---|---|---|
| Perkins & Salomon | 1992 | negative transfer (× positive, null; near × far) | aprender em A "undermines" o desempenho em B relacionado | aprendiz/contexto | nenhuma métrica; comparação implícita com quem não aprendeu A |
| Rosenstein et al. | 2005 | negative transfer | modelo com fonte auxiliar significativamente pior que o B-only no alvo | tarefa | acurácia no holdout do alvo, teste p<0,05 por par A→B |
| Ben-David et al. | 2010 | (sem o termo) "suboptimal performance" | com m_T ≥ d/A², qualquer peso na fonte piora o erro no alvo | domínio | cota teórica: d_{HΔH}, λ, α* ótimo |
| Kendall et al. | 2018 | (sem o termo) sensibilidade à ponderação | modelo conjunto pior que o single-task por má ponderação (escala/ruído) | tarefa | métricas de tarefa contra single-task; varredura de w |
| Wang et al. | 2019 | NTC / NTG | R_T(A(S,T)) > R_T(A(∅,T)) para um algoritmo A fixo | domínio | NTG = diferença de acurácia entre o target-only e o mesmo algoritmo com a fonte |
| Zhao et al. | 2019 | (sem o termo) representação invariante "hurtful" | invariância + erro baixo na fonte pode aumentar o erro no alvo | domínio | cotas: shift condicional; JS entre marginais de rótulos |
| Sattler et al. | 2019 | incongruência | não existe θ* que seja ótimo local para todos os clientes | cliente | cosseno entre atualizações; critério de norma (média pequena, máximo individual grande) |
| Arivazhagan et al. (FedPer) | 2019 | "ill-effects of statistical heterogeneity" | FedAvg degradado por heterogeneidade; sem definição formal | cliente | acurácia média por cliente contra FedAvg |
| Standley et al. | 2020 | negative transfer | co-treino pior que a tarefa sozinha com compute igualado | tarefa | variação relativa da perda; afinidade simetrizada |
| Yu et al. (PCGrad) | 2020b | conflicting gradients | cos φ_ij < 0 (mecanismo, não desfecho) | parâmetro/gradiente | cosseno por batch; Φ; curvatura |
| Yu et al. (Salvaging) | 2020a | (sem o termo) falta de incentivo | modelo federado pior que o local no teste do próprio cliente | cliente | acurácia por participante contra o local |
| Liu et al. (CAGrad) | 2021 | conflicting gradients | ∃i: ⟨g_i, g_0⟩ < 0; causas: escala e direção | parâmetro/gradiente | R(θ,d) ≈ −min_i⟨g_i,d⟩ (só dentro do otimizador); Δm |
| Fifty et al. | 2021 | inter-task affinity | Z_{i→j} < 0: o passo de i aumenta a perda de j | tarefa (via parâmetro) | razão de perdas antes/depois de um passo de *lookahead* |
| Collins et al. (FedRep) | 2021 | (sem o termo) heterogeneidade de dados | modelo único pior que o local; custo fechado da média de heads (caso linear) | cliente | acurácia local média; distância de ângulo principal |
| Li et al. (FedBN) | 2021a | feature shift non-iid | P_i(x) ou P_i(x|y) variam entre clientes (só a condição de dados) | cliente | acurácia por cliente contra SingleSet (local) |
| Li et al. (Ditto) | 2021b | (sem o termo) parentesco τ, "representation disparity" | local supera global quando τ é grande | cliente | média e desvio-padrão da acurácia por dispositivo |
| Navon et al. (Nash-MTL) | 2022 | gradients conflict | direção conflitante ou dominância de magnitude; interação (Σ_{j≠i}α_j g_j)^T g_i | parâmetro/gradiente | Δm, MR; não mede o conflito |
| Zhang et al. | 2023 | NT / NTG | igual a Wang et al. | domínio | NTG |
| Bao et al. | 2023 | negative transfer problem | modelo FL pior que o modelo local, por cliente | cliente | IPR, RSD |
| Jiang et al. (ForkMerge) | 2023 | weak / strong negative transfer | TG(λ) < 0 (fraca); max_λ TG < 0 (forte) | tarefa | TG; GCS como diagnóstico; CSD |
| Liu et al. (FAMO) | 2023 | conflicting gradients, under-optimization | ∇ℓ_i · d < 0, danoso só se persistente | tarefa/gradiente | taxa de melhoria r_i; Δm |
| Senushkin et al. (Aligned-MTL) | 2023 | conflict, dominance, instability | conflito = cosseno < 0; dominância = razão de normas; estabilidade = κ(G) | parâmetro/gradiente | κ(G), cosseno, razão de normas; Δm |
| Chen et al. (FedBone) | 2023 | conflicting gradients (entre clientes) | cos(∇_i, ∇_j) < 0 entre clientes | cliente/parâmetro | não mede; o teste só dispara a projeção |
| Tahir et al. | 2024 | negative transfer / transferability | 𝒯 = E[R_sc − R_tx] < 0 | domínio/tarefa | MSE esperado contra o modelo treinado do zero |
| Yang et al. (FMTL-Bench) | 2024 | (sem o termo) gradient conflict, heterogeneidade | implícita: Δ% < 0 contra o Local | cliente × tarefa | Δ%, ΔG |
| Lu et al. (FedHCA2) | 2024 | model incongruity, conflicting updates | implícita: produto interno baixo entre deltas; Δm < 0 | cliente/parâmetro | Δm contra o local; conflito só na teoria |
| Jeong & Yoon | 2025 | negative transfer; proximal inter-task affinity | degradação informal; B_{G→k} < 0 indica dano | tarefa (via parâmetro + head) | B por *lookahead* com EMA; Δm |
| Mueller et al. | 2025 | negative transfer; directional / magnitude conflict | ΔE_k = E_k(MT) − E_k(ST) < 0 | tarefa | ΔE; cosseno alvo × UMTG; fatores de trajetória |
| Zhang et al. (FedGain) [só resumo] | 2026 | negative gain | colaborativo pior que o local num cliente | cliente | NGP (provável % de clientes com ganho negativo), RSD |
| Liu et al. (OmniFM) | 2026 | modality-induced drift, conflicting minima | qualitativa (bacias de perda divergentes) | cliente | não mede; só métricas de tarefa |

---

## 3. Divergências e ambiguidades entre definições

1. **Qual é o baseline.** As definições usam baselines diferentes:
   - target-only com o *mesmo algoritmo* (Wang et al., 2019; Zhang et al., 2023);
   - B-only (Rosenstein et al., 2005);
   - tarefa sozinha com *compute igualado* (Standley et al., 2020);
   - modelo treinado do zero (Tahir et al., 2024);
   - modelo single-task (Jiang et al., 2023; Mueller et al., 2025; Kendall et al., 2018);
   - modelo local, por cliente (Bao et al., 2023; Yu et al., 2020a; Yang et al., 2024; Li et al., 2021a; Li et al., 2021b; Zhang et al., 2026 [só resumo]);
   - FedAvg em vez do local, no artigo do FedPer (Arivazhagan et al., 2019), que usa o treino local só de forma indireta (Sec. 4.4).

   Wang et al. (2019) alertam que comparar com o *melhor* target-only é imensurável e classifica como NT até um algoritmo que nunca usa a fonte. Para o projeto, o baseline adequado é o local-only pareado (mesmo algoritmo, budget e seeds).
2. **Desfecho × mecanismo.** NTG, TG, ΔE, Δ%, IPR, NGP e 𝒯 medem desempenho. Cosseno < 0 (Yu et al., 2020b; Chen et al., 2023), incongruência (Sattler et al., 2019), produto interno baixo entre deltas (Lu et al., 2024), κ(G) (Senushkin et al., 2023) e R(θ,d) (Liu et al., 2021) descrevem geometria de otimização. O survey trata o conflito como *possível causa* de NT, não como definição (Zhang et al., 2023). O próprio PCGrad diz que o conflito só prejudica junto com grande diferença de magnitude e alta curvatura (Yu et al., 2020b). Dois trabalhos testam essa ligação empiricamente e a encontram fraca (Jiang et al., 2023, Finding 1; Mueller et al., 2025, Sec. 5.2).
3. **O que é "conflito".** Há pelo menos quatro operacionalizações diferentes:
   - par a par entre gradientes, cos < 0 (Yu et al., 2020b; Chen et al., 2023);
   - cada tarefa contra a direção da atualização, ⟨g_i, d⟩ < 0 (Liu et al., 2021; Liu et al., 2023);
   - interação de uma tarefa com a soma ponderada das outras (Navon et al., 2022, Eq. 2);
   - número de condição da matriz de gradientes, em que κ = 1 sse os gradientes são ortogonais e de norma igual (Senushkin et al., 2023, Eq. 2).

   Elas não são equivalentes. Uma atualização pode não ter nenhum par com cos < 0 e ainda assim ser mal condicionada, porque um participante domina pela norma.
4. **Direção × magnitude.** Navon et al. (2022), Liu et al. (2021), Senushkin et al. (2023) e Mueller et al. (2025) separam conflito direcional de conflito ou dominância por magnitude. Kendall et al. (2018) vão além e explicam a NT só pela escala e pelo ruído das tarefas, sem conflito direcional. O rótulo único "conflicting gradients" esconde essa distinção (Liu et al., 2021).
5. **Assimetria.** A transferência é assimétrica (Fifty et al., 2021; Standley et al., 2020; Jeong & Yoon, 2025). No experimento de Standley et al., a tarefa auxiliar Normals melhorou a parceira em 15 de 16 modelos e ainda assim piorou ela mesma. Definições por par simetrizado escondem isso.
6. **Binário × graduado × dependente de hiperparâmetro.** A incongruência é binária (Sattler et al., 2019). O NTG, o TG e o Δ% são contínuos. Rosenstein et al. (2005) classificam em três categorias por significância. Jiang et al. (2023) separam NT **fraca** (uma ponderação melhor resolve) de NT **forte** (nenhuma ponderação resolve). Isso torna explícito que a mesma observação pode ser NT ou não, conforme o λ escolhido.
7. **Nula × negativa.** Na psicologia, "negative" é usado tanto para transferência ausente quanto para transferência danosa, e os dois devem ser separados (Perkins & Salomon, 1992). "Federado não ajuda" (Δ ≈ 0) e "federado prejudica" (Δ < 0) são achados diferentes.
8. **Dependência de contexto.** A NT muda de sinal com a quantidade de dados rotulados no alvo (Wang et al., 2019; Rosenstein et al., 2005; Ben-David et al., 2010; Tahir et al., 2024), com a capacidade e com o tamanho do dataset (Standley et al., 2020), com hiperparâmetros (Fifty et al., 2021, Tabela 3; Jiang et al., 2023) e com o parentesco entre clientes (Li et al., 2021b). Por isso a NT não é uma propriedade fixa de um par de domínios. Perkins & Salomon (1992) acrescentam que, em humanos, a NT costuma aparecer só no início da aprendizagem.
9. **Proximidade de domínio não prevê NT, mas a forma da divergência importa.** A afinidade multitarefa não se correlaciona com a afinidade de *transfer* da Taskonomy (Standley et al., 2020). Distâncias KL e Wasserstein podem ser arbitrariamente grandes com transferência positiva (Tahir et al., 2024). Isso diverge da visão de que "a divergência entre domínios é a raiz da NT" (Zhang et al., 2023; Wang et al., 2019). Zhao et al. (2019) refinam o ponto: o que gera dano é o shift condicional ou de marginais de rótulos, não a distância das marginais de features. ForkMerge associa a NT ao aumento do shift treino/teste do alvo (Jiang et al., 2023, Finding 2). FedGain associa a NT ao balanço entre distância de distribuição e ganho de escala de dados (Zhang et al., 2026) [só resumo].
10. **Terminologia dispersa.** Os mesmos fenômenos aparecem como "task interference" (Xin et al., 2022), "negative interference" (Zhang et al., 2023), "incongruence" (Sattler et al., 2019), "model incongruity" (Lu et al., 2024), "negative gain" (Zhang et al., 2026) [só resumo] e "modality-induced drift" (Liu et al., 2026). Vários trabalhos de referência simplesmente não usam o termo "negative transfer": (Yang et al., 2024; Lu et al., 2024; Kurin et al., 2022; Xin et al., 2022; Yu et al., 2020b; Fifty et al., 2021; Liu et al., 2021; Navon et al., 2022; Kendall et al., 2018; Zhao et al., 2019; Arivazhagan et al., 2019; Collins et al., 2021; Li et al., 2021a; Li et al., 2021b; Chen et al., 2023; Liu et al., 2026).

---

## 4. Como é medido (métricas)

**Métricas de desfecho (definem NT)**
- **NTG** = erro_com_fonte − erro_sem_fonte; NT ocorre se NTG > 0 (Wang et al., 2019; Zhang et al., 2023). O survey observa que o NTG "may not always be computable", por exemplo em adaptação de domínio não supervisionada.
- **Transfer gain TG(λ, A)** = P(conjunto) − P(target-only), com NT fraca se TG(λ) < 0 e forte se max_λ TG < 0 (Jiang et al., 2023, Defs. 3.1–3.3). **ΔE_k** = E_k(MT) − E_k(ST) é a mesma ideia (Mueller et al., 2025, Sec. 2.1).
- **Δ% / Δm**: média entre tarefas da variação relativa contra o baseline local ou single-task, com o sinal corrigido para métricas em que menor é melhor (Yang et al., 2024, Eq. 5; Lu et al., 2024; Liu et al., 2021, Sec. 5.2; Navon et al., 2022; Liu et al., 2023; Senushkin et al., 2023; Jeong & Yoon, 2025). Há uma variante ponderada por tarefa e outra por métrica (Senushkin et al., 2023). Atenção ao sinal: em uns trabalhos Δm negativo significa *pior* que o single-task (Jeong & Yoon, 2025), em outros significa *melhor* (Navon et al., 2022; Liu et al., 2023).
- **Mean Rank (MR)** entre métricas de tarefa (Navon et al., 2022; Liu et al., 2023).
- **IPR**, a fração de clientes em que o modelo FL supera o local, e **RSD**, o desvio-padrão dos ganhos por cliente (Bao et al., 2023).
- **NGP**, quase certamente a porcentagem de clientes com ganho negativo contra o local (Zhang et al., 2026) [só resumo]. A sigla não é definida no material acessível.
- **Desvio-padrão da acurácia entre clientes** como medida de equidade (Li et al., 2021b, Def. 2).
- **Transferabilidade 𝒯** = E[R_sc − R_tx] (Tahir et al., 2024).
- **Afinidade por par com compute igualado** (Standley et al., 2020).
- **Contagem de participantes prejudicados**: número de clientes com acurácia federada < acurácia local (Yu et al., 2020a).
- **Posição em relação à frente de Pareto** da escalarização, obtida por varredura de pesos (Xin et al., 2022).

**Métricas de mecanismo (diagnosticam causas candidatas)**
- **Cosseno entre gradientes**, com a porcentagem de iterações com cosseno positivo; **similaridade de magnitude Φ**; **curvatura multitarefa** H (Yu et al., 2020b).
- **Cosseno do gradiente da tarefa-alvo com o gradiente multitarefa (UMTG)**, médio em 200 batches (Mueller et al., 2025, Eq. 6). Também **sharpness**, **coerência de gradiente** e **traço da Fisher no início do treino** como fatores de trajetória (Eqs. 3–5).
- **Número de condição κ(G) = σ_max/σ_min** da matriz de gradientes das tarefas, e **razão máxima entre normas** como medida de dominância (Senushkin et al., 2023, Eq. 2 e Fig. 4). Execuções com κ ≤ 10 tendem a ter melhor desempenho final (Sec. 6.4).
- **Medida de conflito R(θ,d) ≈ −min_i⟨g_i, d⟩**, a pior variação de primeira ordem entre tarefas para a direção d (Liu et al., 2021, Eq. 2). É usada dentro do otimizador e não é registrada como diagnóstico.
- **Termo de interação** (Σ_{j≠i} α_j g_j)^T g_i por tarefa (Navon et al., 2022, Eq. 2), mais o menor valor singular de G^T G para checar independência linear (App. D.5).
- **Taxa de melhoria por tarefa** r_i = (ℓ_{i,t} − ℓ_{i,t+1}) / ℓ_{i,t} (Liu et al., 2023, Eq. 3).
- **Cosseno entre atualizações de clientes**; α_intra^min, α_cross^max e **gap de separação** g(α); critério de norma ‖média Δ‖ < ε1 com max‖Δ_i‖ > ε2 (Sattler et al., 2019).
- **Afinidade por lookahead** Z_{i→j}. Agrupar tarefas por cosseno (CS) foi pior que agrupar por Z (Fifty et al., 2021). A **afinidade proximal** B_{G→k} também atualiza os parâmetros específicos da tarefa k e sai de graça das atualizações sequenciais (Jeong & Yoon, 2025, Def. 2).
- **Similaridade de cosseno entre gradientes (GCS)** como diagnóstico, que se mostrou pouco correlacionado com TG, e **discrepância de confiança (CSD)** como proxy do shift treino/teste (Jiang et al., 2023, Defs. 3.4–3.5).
- **Norma da atualização somada** ‖Σ_i ∇L_i‖ nos parâmetros compartilhados, usada como diagnóstico de sub-otimização (Kurin et al., 2022, §5.3).
- **"Positive sign purity"** do GradDrop, a pureza de sinal por coordenada (descrita em Kurin et al., 2022).
- **H-divergência empírica** estimada com um classificador de domínio (Ben-David et al., 2010, Lema 2), ou com um discriminador federado entre pares de clientes (Bao et al., 2023). **Distância JS entre marginais de rótulos** e **shift da função de rotulação** E[|f_S − f_T|] (Zhao et al., 2019, Teoremas 4.1 e 4.3).
- **Distância de cosseno entre embeddings médios dos clientes**, num extrator InfoNCE compartilhado, e não entre deltas (Zhang et al., 2026) [só resumo].
- **Distância de ângulo principal** entre a representação aprendida e a verdadeira (Collins et al., 2021, Def. 1, Eq. 9).
- **Menor autovalor do Gram NTK** como medida teórica da velocidade de convergência (Li et al., 2021a, Teorema 4.4).

**Observação.** Nenhum dos 32 trabalhos usa a razão de cancelamento agregada 1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖ com separação intra/inter-dataset e referência ortogonal. As métricas mais próximas são o critério de norma de Sattler et al. (2019) e a norma da atualização somada de Kurin et al. (2022). FedBone afirma que o conflito causa "a reduction in final gradients", mas não mede essa redução (Chen et al., 2023). OmniFM afirma deriva entre modalidades sem medir cosseno nem normas (Liu et al., 2026).

---

## 5. Estado da arte de mitigação

A segunda passagem fechou boa parte da lacuna de 2021–2026 apontada na primeira. Agora há métodos recentes de otimização multitarefa (Nash-MTL, ForkMerge, FAMO, Aligned-MTL, atualização seletiva por grupos), FL personalizado (FedRep, FedBN, Ditto, FedGain) e FL multitarefa e médico (FedBone, OmniFM). Duas ressalvas continuam valendo:
- quase todo o SOTA de MTO é centralizado, com todas as tarefas rotuladas em todas as imagens;
- nenhum método avaliado combina FL cross-modality, clientes multitarefa seg + cls e medida do conflito entre clientes.

| Método | Ano | Ideia | Onde se aplica | Evidência |
|---|---|---|---|---|
| FedGain (Zhang et al., 2026) [só resumo] | 2026 | FL clusterizado que escolhe coalizões minimizando uma cota: viés por distância de distribuição + termo de escala em lei de potência ("Effective Federated Capacity") | FL global/personalizado | FedAvg: NGP 10/55/50% → FedAvg+FedGain: 0/0/0 em CIFAR-10/100/FMNIST. Agrupar só por quantidade (NGP 0/10/10) supera agrupar só por distância (5/60/60). Parte do ganho pode vir de *não* federar. Tarefa única, mudança de distribuição sintética [só resumo] |
| OmniFM (Liu et al., 2026) | 2026 | compartilha só conhecimento espectral de baixa frequência (banco global + prompts), com termo proximal SPAlign | FL médico cross-modality, uma tarefa por federação | Segmentação no FeTS2022: Dice 77,41 contra FedPer 73,86, FedAvg 44,16 e Local 44,23. Uma seed, inconsistências entre tabelas, ganho do SPAlign de ~1 ponto. Nunca mistura tarefas numa federação |
| Atualização seletiva por grupos (Jeong & Yoon, 2025) | 2025 | atualiza grupos de tarefas em sequência no mesmo batch, agrupando por afinidade proximal | MTL centralizado | Taskonomy Δm −1,42 contra Aligned-MTL −6,22 e GD −8,65. A maior parte do ganho vem da atualização sequencial; o agrupamento acrescenta pouco. ~3× mais lento por batch que GD. Δm ainda negativo em todos os benchmarks |
| FedHCA2: agregação *conflict-averse* + cross-attention + pesos aprendíveis (Lu et al., 2024) | 2024 | CAGrad aplicado aos deltas do encoder dos clientes; atenção sobre os deltas dos decoders; α_i/β_i aprendíveis | FL multitarefa (encoder compartilhado, heads locais) | Δm +2,18% (FedAvg −11,76%) e +0,75% (FedPer −1,11% e −0,16%). Ganhos pequenos e uma seed. A restrição c·‖média‖ não recupera magnitude perdida |
| FMTL-Bench (baselines) (Yang et al., 2024) | 2024 | benchmark de FedAvg, FedProx, FedAMP, FedRep, PCGrad, CAGrad, MaT-FL, FedMTL × arquiteturas SD/MD/TC | FL multitarefa | Nenhum método vence sempre. PCGrad "generally underperforms". Compartilhar só o encoder e usar decoder TC suavizam o cenário cross-domain (NIID-7: FedAvg-SD −27,21%, FedAvg-TC −4,07%) |
| ForkMerge (Jiang et al., 2023) | 2023 | bifurca os parâmetros em ramo só-alvo e ramo conjunto; a cada Δt interpola com λ* escolhido na validação do alvo | aprendizado com tarefas auxiliares (centralizado) | DomainNet: NT em 23/30 pares com pesos iguais, evitada em 30/30. NYUv2 Δm +4,03% contra NashMTL 1,11% e CAGrad 1,41%. Exige validação do alvo. É o único método avaliado próximo de *model merging* (interpolação de parâmetros) |
| Aligned-MTL (Senushkin et al., 2023) | 2023 | substitui G pela matriz mais próxima com colunas ortogonais e de norma igual (κ = 1), reescalada por σ_min | MTL centralizado | NYUv2/MTAN Δm −8,40% contra Nash-MTL −7,56% e Uniform +5,46%. MT10 0,97 contra Nash-MTL 0,91. Custo O(T). O avaliador observa que reescalar por σ_min pode encolher o passo quando G é mal condicionada |
| FAMO (Liu et al., 2023) | 2023 | ponderação dinâmica O(1) pelo histórico de log-perdas, para que todas as perdas caiam à mesma taxa | MTL centralizado | NYUv2 Δm −4,10 contra NashMTL −4,04. QM9 58,5 contra 62,0, mas MR pior. Overhead desprezível em relação a LS. Competitivo e barato, não uniformemente melhor. Depende de γ |
| FedBone (Chen et al., 2023) | 2023 | FMTL em *split learning*: modelo geral grande no servidor, adaptadores e heads personalizados, agregação com projeção PCGrad entre clientes | FL multitarefa (inclui oftalmologia) | NYUDv2: melhor nas 4 tarefas contra FedAvg/FedProx/pFedMe/FedEM. A ablação mostra que o módulo de adaptação personalizado ajuda mais que a agregação anticonflito. Sem baseline local, uma seed, preprint |
| Nash-MTL (Navon et al., 2022) | 2022 | solução de barganha de Nash: resolve G^T G α = 1/α; invariante à escala; norma da atualização sempre √K | MTL centralizado | NYUv2 Δm −4,04% (único claramente melhor que STL na tabela). QM9 62,0 contra SI 77,8 e LS 177,6. MT10 0,91 ≈ STL. Custa K backward por passo; o substituto no nível de features falha (QM9 179,2) |
| CAGrad (Liu et al., 2021) | 2021 | maximiza a melhora da pior tarefa numa bola de raio c‖g_0‖ em torno do gradiente médio | MTL centralizado | NYUv2 Δm 0,20 contra PCGrad 3,97. CityScapes 11,64 contra PCGrad 18,29. Δm ainda positivo (pior que STL) nos dois. Dentro do desvio do CARE no Meta-World |
| FedRep (Collins et al., 2021) | 2021 | como o FedPer, mas alterna: ajusta a head até quase o ótimo com o corpo congelado, depois atualiza o corpo | FL personalizado | Margem pequena sobre o FedPer (0,3–3 pontos). O Local vence no cenário mais heterogêneo e rico em dados (CIFAR10 (100,2): 89,79 contra 87,70). Só heterogeneidade de rótulos, mesma modalidade |
| FedBN (Li et al., 2021a) | 2021 | FedAvg sem agregar camadas BatchNorm | FL com feature shift (um domínio por cliente) | Dígitos SVHN: FedAvg 62,86 < SingleSet 65,25 < FedBN 71,04. Ainda abaixo do SingleSet em DomainNet Quickdraw e ABIDE UM. Só classificação |
| Ditto (Li et al., 2021b) | 2021 | modelo pessoal regularizado em L2 rumo ao global, com λ por dispositivo (λ → 0 = local) | FL personalizado | Fashion-MNIST limpo: Ditto 0,943 contra global 0,911 e local 0,876. λ* ótimo de Bayes no caso linear. Tarefa única, modalidade única, cross-device |
| TAG (Fifty et al., 2021) | 2021 | afinidade por lookahead durante uma execução, depois agrupamento ótimo de tarefas | MTL centralizado | Taskonomy −10,0% de perda contra MTL, 11,6× mais rápido que HOA; grupos > UW/GradNorm/PCGrad |
| Agrupamento ótimo + HOA (Standley et al., 2020) | 2020 | treina subconjuntos e escolhe grupos sob um orçamento de inferência | MTL centralizado | vence a rede única, as redes independentes, MGDA e GradNorm em 4 configurações; custo exponencial |
| PCGrad (Yu et al., 2020b) | 2020 | projeta gradientes conflitantes no plano normal | MTL centralizado | Meta-World MT10 resolvido; NYUv2 melhor em 8/9 métricas. Não confirmado por (Kurin et al., 2022) nem por (Yang et al., 2024) |
| Adaptação local pós-hoc (Yu et al., 2020a) | 2020 | fine-tuning, freeze-base, EWC ou KD do modelo global em cada cliente | FL | usuários do Reddit prejudicados (BASIC) caem de 9,22% para 0,04%. Mitigação de desfecho, não de causa |
| Clustered FL (Sattler et al., 2019) | 2019 | bipartição recursiva pelo cosseno quando a federação estagna | FL | CIFAR com rótulos permutados: ~20% → ~60%. Conflitos sintéticos com cosseno ≈ −1 |
| FedPer (Arivazhagan et al., 2019) | 2019 | camadas base federadas e camadas finais personalizadas | FL personalizado | Supera o FedAvg em k=4 no CIFAR-10 e converge para o FedAvg quando os dados ficam IID. Só curvas, sem seeds repetidas |
| Discriminator gate (Wang et al., 2019) | 2019 | reponderação por densidade P_T/P_S com discriminador conjunto | adaptação de domínio | DANN: NTG +3,0 → −7,4 (+10,4 pp). Fonte e alvo únicos, classificação |
| Ponderação por incerteza (Kendall et al., 2018) | 2018 | pesos de tarefa aprendidos por incerteza homocedástica: Σ 1/(2σ_t²) L_t + log σ_t | MTL centralizado | Tiny CityScapes: pesos uniformes 50,1% IoU (−9,3 contra single-task 59,4); incerteza 63,4%. Uma seed. Já implementado no projeto (`loss.task_weighting: uncertainty`) |
| Escalarização bem ajustada / regularização (Xin et al., 2022; Kurin et al., 2022) | 2022 | pesos fixos ajustados por grid, mais dropout/ℓ2/early stopping | MTL centralizado | iguala ou supera MGDA, GradNorm, PCGrad, IMTL e RLW com custo muito menor |

*Model merging* propriamente dito continua fora das avaliações. A única entrada próxima é a interpolação de parâmetros validada do ForkMerge (Jiang et al., 2023).

---

## 6. Críticas e resultados negativos

- **Métodos anticonflito não superam uma escalarização ajustada.** "all of the MTO algorithms in our study simply yield performance trade-off points on the scalarization Pareto front" (Xin et al., 2022). "the effect of tuning hyper-parameters is much more significant than the effect of the MTO algorithm choice" (Xin et al., 2022). O próprio Nash-MTL reconhece que a maioria dos métodos de MTL "fall short compared to the simple scale-invariant baseline, which ignores gradient interaction" no QM9 (Navon et al., 2022).
- **A hipótese do conflito perde força.** Um GradDrop "agnóstico de sinal" iguala o GradDrop original, o que "seriously undermines the conflicting gradient hypothesis" (Kurin et al., 2022, App. B.4). Segundo os autores, MGDA, IMTL e PCGrad funcionam em parte como regularizadores ou por sub-otimização (Kurin et al., 2022). Eles atribuem a dificuldade da MTL a "incorporating adequate regularization" (§5.1) e reconhecem que isso é "only a partial explanation".
- **Conflito e NT se desacoplam empiricamente.** "high amounts of gradient conflict are highly correlated with [worse optimization], while having almost no correlation with task generalization" (Mueller et al., 2025, Sec. 6). No CIFAR-100, a configuração auxiliar com menor similaridade de gradiente gerou a *melhor* generalização para a tarefa Fish (Sec. 5.2). A transferência aparece como um gap de generalização cedo no treino, não na minimização final da perda (Sec. 3.1). ForkMerge encontra o mesmo desacoplamento: a NT pode ser mais grave com gradientes muito consistentes, e um termo ℓ2 conflita em gradiente e mesmo assim ajuda (Jiang et al., 2023, Finding 1). Os dois estudos são centralizados.
- **Em FL multitarefa, a cirurgia de gradiente falha.** PCGrad "generally underperforms" e só FedAvg e FedProx são robustos quando os clientes ficam menores e mais numerosos (Yang et al., 2024). O benchmark não explica por quê.
- **O SOTA de MTO ainda não elimina a NT.** Mesmo o melhor método mantém Δm negativo (pior que o single-task) em Taskonomy, NYUD-v2 e PASCAL-Context (Jeong & Yoon, 2025). O CAGrad fica acima do STL em Δm no NYUv2 e no CityScapes (Liu et al., 2021). Margens entre métodos recentes costumam ser pequenas e às vezes ficam dentro do desvio (Liu et al., 2023; Liu et al., 2021).
- **Ganhos do SOTA sobre o local-only são marginais.** Ficam na faixa de 1–2% de Δm, e algumas tarefas ainda perdem para o local (Lu et al., 2024). O FedRep ganha 0,3–3 pontos do FedPer e perde para o Local no cenário mais heterogêneo (Collins et al., 2021). O FedBN continua abaixo do SingleSet em alguns clientes (Li et al., 2021a). O FedGain zera a NGP em parte por não colaborar quando a melhor coalizão de um cliente é ele mesmo (Zhang et al., 2026) [só resumo].
- **Conflito afirmado, mas não medido, em FMTL e FL médico.** FedBone define o conflito entre clientes e não mede a sua frequência nem o efeito da projeção. A ablação mistura projeção e reescala por atenção e não tem baseline local (Chen et al., 2023). OmniFM atribui a deriva às modalidades sem nenhuma medida de cosseno, norma ou paisagem de perda (Liu et al., 2026).
- **Conflito não é condição suficiente** (Yu et al., 2020b, Sec. 2.2), e **cosseno é um proxy imperfeito** de transferência (Fifty et al., 2021; Mueller et al., 2025; Jiang et al., 2023).
- **Invariância pode prejudicar sem conflito visível.** Quando as marginais de rótulos diferem, forçar representação invariante cria um piso de erro conjunto, e encoders separados por domínio não escapam disso (Zhao et al., 2019, Teorema 4.3 e Corolário 4.1). A cota só vale para um espaço de rótulos compartilhado e é solta (Sec. 4.3).
- **Ponderação adaptativa também tem custo.** A incerteza que cai durante o treino aumenta a taxa de aprendizado efetiva e exige *annealing* (Kendall et al., 2018, App. B). Trabalhos posteriores não confirmam ganho sobre escalarização ajustada (Xin et al., 2022).
- **Avaliação frágil.** "rerunning multiple seeds is insufficient" quando os hiperparâmetros vêm de um grid esparso (Xin et al., 2022). Vários trabalhos centrais usam uma única seed ou não reportam variância (Yang et al., 2024; Lu et al., 2024; Standley et al., 2020; Yu et al., 2020a; Kendall et al., 2018; Chen et al., 2023; Liu et al., 2026; Arivazhagan et al., 2019; Collins et al., 2021).
- **A NT é relativa ao baseline.** Com mais dados rotulados no alvo, o baseline melhora e a NT aumenta (Wang et al., 2019; Ben-David et al., 2010; Tahir et al., 2024). A regularização ridge aumenta a região de NT (Tahir et al., 2024). O argumento do FedPer de que as camadas base ajudam depende de os clientes serem pequenos demais para treinar sozinhos (Arivazhagan et al., 2019, Sec. 4.4).
- **As medidas de distância entre domínios são insuficientes** para prever o sucesso da transferência (Tahir et al., 2024; Standley et al., 2020).

---

## 7. O que isso implica para o nosso diagnóstico

Cautela geral: os p-valores do projeto vêm de Wilcoxon pareado sobre 8 pares cliente × fold não independentes, com uma única seed. São **exploratórios** e não sustentam afirmações de significância. Os itens marcados **(derivação nossa)** não estão em nenhuma fonte e devem aparecer na tese como hipótese ou cálculo do projeto.

1. **"Federado < local-only" é negative transfer no sentido canônico, com uma condição.** Por cliente e tarefa, NTG = erro_FedPer − erro_local-only (Wang et al., 2019; Zhang et al., 2023). Em notação de desfecho, TG = P_FedPer − P_local (Jiang et al., 2023) ou ΔE_k (Mueller et al., 2025). O enquadramento federado coincide (Bao et al., 2023; Yu et al., 2020a; Zhang et al., 2026 [só resumo]), e a âncora histórica é a definição contrastiva de Perkins & Salomon (1992).

   A condição é que os dois braços sejam o *mesmo algoritmo*: mesmo modo de budget, seeds pareadas e mesma seleção de modelo. Wang et al. (2019) alertam que mudar o budget ou a agregação muda A e, portanto, o NTG.

   A tese pode relatar, a partir dos `*_test_results.csv` já existentes:
   - Δ% por dataset (Yang et al., 2024);
   - IPR/RSD por cliente × tarefa (Bao et al., 2023);
   - uma fração no estilo NGP de células cliente × tarefa × fold em que o FedPer perde para o local (Zhang et al., 2026) [só resumo].

   Também deve separar "sem benefício" (Δ ≈ 0) de "dano" (Δ < 0) (Perkins & Salomon, 1992).
2. **A razão de cancelamento e o cosseno não medem NT. Eles testam um mecanismo candidato.** Cancelamento inter-dataset baixo, perto da referência ortogonal, não é evidência de conflito no sentido do PCGrad (cos < 0; Yu et al., 2020b) nem de incongruência no sentido do CFL, em que o cancelamento vem de atualizações antiparalelas (Sattler et al., 2019). A leitura prudente é: **NT observada; conflito direcional como causa principal pouco sustentado pelos dados atuais.**

   Isso é coerente com a literatura que desacopla conflito e generalização (Mueller et al., 2025; Jiang et al., 2023). A ressalva é que essa evidência é centralizada e serve por analogia, não como evidência federada direta.

   O resultado também não elimina o conflito como contribuição secundária. O cosseno global pode esconder conflito por bloco ou em fases específicas do treino, e as garantias do CFL valem só perto de pontos estacionários. Na taxonomia de Mueller et al. (2025), a razão de cancelamento e o cosseno par a par são medidas de **nível 1** (conflito). Diferem do original por usarem deltas multi-step no momento da agregação, e não gradientes por batch.
3. **Diluição do passo como problema de magnitude, não de direção.** Com atualizações quase ortogonais, a média já encolhe o passo. Para normas iguais e pesos uniformes, ‖média‖ ≈ ‖Δ‖/√n, e é isso que a referência ortogonal captura **(derivação nossa)**. Vale verificar se o fator ~10× medido é compatível com essa referência, dados o número de clientes e os pesos reais. Se o fator for maior, há outro componente: pesos assimétricos ou normas desiguais (o Φ de Yu et al., 2020b).

   Na literatura, isso se aproxima de:
   - o **"magnitude conflict"** que leva à sub-otimização de tarefas (Mueller et al., 2025, Sec. 2.1) e a dominância por norma (Navon et al., 2022; Liu et al., 2021; Senushkin et al., 2023);
   - o "generalized CG" de FAMO, em que todas as perdas caem, mas devagar demais (Liu et al., 2023). A diluição apareceria como r_i uniformemente pequeno, não negativo **(derivação nossa, a partir da definição de FAMO)**;
   - a leitura de Kurin et al. (2022), em que mudança de magnitude age como sub-otimização ou regularização implícita do tronco, diagnosticada pela norma da atualização;
   - o argumento do Nash-MTL de que, com gradientes ortogonais, a solução é a **soma** dos gradientes normalizados, com norma sempre √K, e não a média (Navon et al., 2022, App. A). Em termos de princípio, isso sustenta que o passo agregado não deveria encolher quando as atualizações não conflitam. A extensão para deltas multi-step de clientes é nossa e não está coberta pela teoria do artigo;
   - o "fator algoritmo" de Wang et al. (2019) e Zhang et al. (2023), que não separam otimização de divergência;
   - a restrição da contribuição de cada participante em Yu et al. (2020a), com clipping DP e mediana.

   O FedPer prescreve exatamente a média ponderada por n_j que gera essa diluição, e não a analisa (Arivazhagan et al., 2019). Nenhum trabalho avaliado define "diluição do passo" como forma de NT. A tese deve apresentá-la como **hipótese de mecanismo própria do projeto**.
4. **Previsões coerentes com o diagnóstico, a testar (não confirmadas):**
   - Se o conflito não é o custo dominante, métodos que só ajustam a direção (PCGrad, CAGrad, o HCA do FedHCA2, a projeção do FedBone) não devem fechar o gap. Isso é consistente com (Yang et al., 2024) e (Kurin et al., 2022). A restrição c·‖média‖ do HCA não restaura magnitude (Lu et al., 2024), e o CAGrad ancora o passo deliberadamente na média (Liu et al., 2021). Uma projeção só remove componentes, então nunca aumenta a norma agregada, e com deltas quase ortogonais ela raramente dispara **(derivação nossa, a partir de Chen et al., 2023)**. A reescala por σ_min do Aligned-MTL pode até encolher o passo (observação do avaliador sobre Senushkin et al., 2023).
   - Ajustar o tamanho efetivo do passo do servidor antes de afirmar NT segue a recomendação de "tune before claiming" (Xin et al., 2022). Em termos do ForkMerge, se uma interpolação ou reescala do passo agregado fecha o gap, a NT é **fraca**; se nenhum peso fecha, é **forte** (Jiang et al., 2023, Defs. 3.2–3.3).
   - Clientes grandes (ISIC, SIIM, TCGA) devem ter NTG maior e o BUSI, pequeno, é onde a transferência positiva é mais plausível (Ben-David et al., 2010; Wang et al., 2019; Tahir et al., 2024; Bao et al., 2023; Li et al., 2021b). FedGain descreve as "negative gains" justamente em "data-abundant clients" (Zhang et al., 2026) [só resumo]. Basta reportar o NTG em função do tamanho do cliente.
   - O gap federado × local pode já aparecer nas primeiras rodadas, como o gap de generalização precoce de Mueller et al. (2025, Sec. 3.1) e a curva com pico precoce do DANN (Zhao et al., 2019, Fig. 3). Vale plotar o gap de validação por rodada. A sugestão de Perkins & Salomon (1992) de que a NT encolhe com a experiência é só uma analogia.
5. **Diagnósticos complementares sugeridos pela literatura:**
   - Afinidade federada no estilo TAG. Comparar Z_{j→j} (delta próprio) com Z_{agg→j} (delta agregado) separa a perda por diluição do dano direcional (Z_{i→j} < 0), coisa que a razão de cancelamento não consegue. É uma adaptação multi-step, não a Eq. 1 exata (Fifty et al., 2021). A variante proximal reavalia a perda depois de re-adaptar a head personalizada do cliente k (Jeong & Yoon, 2025, Def. 2).
   - κ da matriz de deltas do tronco por rodada, via SVD da Gram K × K. Separa "pouco conflito mas forte dominância por norma" de ortogonalidade verdadeira (Senushkin et al., 2023). A adaptação a deltas de clientes é nossa.
   - Por cliente, min_i⟨Δ_i, Δ_agg⟩/‖Δ_agg‖ e a fração de clientes com ⟨Δ_i, Δ_agg⟩ < 0 (Liu et al., 2021, Eq. 2). Também o termo de interação (Σ_{j≠i} w_j Δ_j)^T Δ_i (Navon et al., 2022, Eq. 2). Os dois respondem se algum cliente é empurrado "morro acima", algo que um cancelamento baixo não descarta. As duas adaptações a deltas são nossas.
   - Taxa de melhoria por cliente r_i depois de aplicar o delta agregado, para detectar sub-otimização persistente sem sinal negativo (Liu et al., 2023). A adaptação é nossa.
   - Gap de separação entre cossenos intra e inter-dataset, para decidir se agrupar por dataset se justifica (Sattler et al., 2019).
   - Braço de federação só intra-dataset ou por coalizão (Bao et al., 2023; Sattler et al., 2019). FedGain sugere considerar a quantidade de dados, e não só a distância, ao formar grupos (Zhang et al., 2026) [só resumo].
   - Braço FedPer + fine-tuning pós-hoc. Se esse braço ainda perder para o local-only, o gap não se explica por falta de personalização (Yu et al., 2020a). Uma variante é o proximal do Ditto, que regulariza uma cópia personalizada do tronco rumo ao `global_shared.pt` com λ por dataset, de modo que λ → 0 recupera o local-only (Li et al., 2021b).
   - Interpolação validada por cliente, θ = (1−λ*)·θ_local + λ*·θ_agregado, com λ* escolhido no slice de validação do cliente, sem tocar o teste (Jiang et al., 2023). λ* = 0 indica NT forte. λ* > 1, ou um passo de servidor reescalado que vence, indica que o agregado é só curto demais. A adaptação federada é nossa.
   - Alternância à la FedRep (head até quase o ótimo, depois o tronco) ou mais épocas de tronco por rodada, como ablação barata para um tronco possivelmente subtreinado (Collins et al., 2021). O ganho esperado é pequeno, porque a margem do FedRep sobre o FedPer é de 0,3–3 pontos.
   - Ablação no estilo FedBN: o MTnnUNet usa InstanceNorm, sem estatísticas correntes, e o MTResNet34UNet usa FrozenBatchNorm, idêntico em todos os clientes. Os dois já evitam a principal falha que o FedBN corrige (Li et al., 2021a). A ablação análoga seria personalizar os parâmetros afins das InstanceNorm do tronco **(derivação nossa)**.
   - H-divergência entre features do tronco como covariável descritiva (Ben-David et al., 2010), com a ressalva de que distância não prevê transferência (Tahir et al., 2024). Zhao et al. (2019) indicam que o shift de rótulos e de funções de rotulação importa mais que a distância das features. A cota deles, porém, supõe um espaço de rótulos compartilhado, o que não vale entre BUSI, ISIC, TCGA-LGG e SIIM-ACR.
6. **Confundidores de desenho a declarar.**
   - O FedPer original personaliza as camadas *finais* e compartilha a base (Arivazhagan et al., 2019). O projeto personaliza também o stem de entrada (`encoder1`) e compartilha o tronco do meio, uma extensão que o FedPer não testou. A heterogeneidade do FedPer original é só de rótulos, num único domínio.
   - A ponderação por incerteza (`loss.task_weighting: uncertainty`) segue Kendall et al. (2018). Como a incerteza cai, a taxa de aprendizado efetiva sobe (App. B), o que muda a norma dos deltas e, portanto, as medidas de cancelamento e diluição. Braços com e sem incerteza devem ser comparados sabendo disso. Com s_t personalizado, ela só reequilibra o objetivo local e não age sobre a diluição na agregação **(derivação nossa)**.
7. **Coerência com o SOTA.** No FedHCA2, o FedPer fica abaixo do local nos dois cenários (Δm −1,11% e −0,16%; Lu et al., 2024). No cenário cross-domain com clientes de tarefa única, o FedAvg perde muito para o local (Yang et al., 2024). Em FL com feature shift, o FedAvg fica abaixo do SingleSet em vários clientes (Li et al., 2021a). O resultado do projeto não é anômalo.

   O OmniFM reporta o contrário (FedPer 73,86 contra Local 44,23 de Dice), mas com segmentação de uma única modalidade (FeTS T1ce), budgets diferentes e um Local que parece subtreinado (Liu et al., 2026). A comparação exige ressalva. Esses benchmarks usam imagens naturais ou federações de tarefa única e não informam diretamente sobre o cenário seg + cls cross-modality do projeto.

---

## 8. Lacunas abertas

1. **Nenhuma definição de NT para FL multitarefa com compartilhamento parcial** (tronco compartilhado com stems e heads personalizados). As definições formais são de fonte e alvo únicos (Wang et al., 2019; Zhang et al., 2023), de aprendizado com tarefas auxiliares (Jiang et al., 2023) ou de MTL centralizado (Mueller et al., 2025). O FMTL usa um Δ% implícito (Yang et al., 2024; Lu et al., 2024), e o "negative gain" é de FL com tarefa única (Zhang et al., 2026) [só resumo].
2. **Nenhuma decomposição do NTG em mecanismos** (conflito direcional × magnitude/diluição × divergência de distribuição × regularização). Zhang et al. (2023) listam a identificação dos fatores exatos da NT como direção futura. Mueller et al. (2025) concluem que prever a transferência só pela otimização "is currently not feasible" (Sec. 6).
3. **Nenhuma medida empírica de conflito entre clientes no SOTA de FMTL.** FedHCA2 invoca o conflito só na teoria (Lu et al., 2024), FedBone o define sem medir (Chen et al., 2023), OmniFM o afirma sem medir (Liu et al., 2026) e o FMTL-Bench não o mede (Yang et al., 2024). A razão de cancelamento com referência ortogonal parece não ter equivalente nas 32 fontes avaliadas.
4. **"Magnitude conflict" não foi estudado em deltas multi-step agregados.** O conceito existe para gradientes por passo em MTL centralizado (Mueller et al., 2025; Navon et al., 2022; Senushkin et al., 2023). Nenhuma fonte o liga à diluição do passo na média do FedAvg/FedPer, que o próprio FedPer não analisa (Arivazhagan et al., 2019).
5. **Falta explicar por que a cirurgia de gradiente falha em FL** (Yang et al., 2024). O desacoplamento conflito × generalização (Mueller et al., 2025; Jiang et al., 2023) é uma explicação candidata, mas só foi mostrado em cenários centralizados.
6. **Domínios médicos cross-modality e pares segmentação + classificação.** O OmniFM é o trabalho mais próximo (FL médico cross-modality com FedPer como baseline forte), mas cada federação tem uma única tarefa e a segmentação tem uma única modalidade (Liu et al., 2026). FedBone tem tarefas oftalmológicas de um único domínio de imagem (Chen et al., 2023). Nenhum trabalho avaliado mistura seg e cls de modalidades diferentes numa mesma federação.
7. **Rigor estatístico.** Faltam múltiplas seeds e variância de hiperparâmetros (Xin et al., 2022) na maior parte dos trabalhos de FMTL e de FL médico (Yang et al., 2024; Lu et al., 2024; Chen et al., 2023; Liu et al., 2026).
8. **Cobertura desta busca.** A segunda passagem cobriu métodos de MTO de 2021–2025 e FL personalizado e médico até 2026. Restam lacunas da própria síntese, não necessariamente da literatura:
   - *model merging* (só a interpolação do ForkMerge aparece);
   - extensões federadas de Nash-MTL, CAGrad ou FAMO (o Nash-MTL cita só um uso em FL, Kim, 2021, não avaliado aqui);
   - o texto completo do FedGain, que não pôde ser lido [só resumo];
   - a fonte primária de Thorndike & Woodworth (1901).

---

---

Fontes: [`fontes.md`](fontes.md). Fichas de leitura por artigo: [`fichas.md`](fichas.md).
