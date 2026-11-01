# Negative transfer: síntese da literatura

> Escopo: esta síntese usa só as 15 avaliações de artigos fornecidas. Todas foram feitas com texto completo (`full_text`), por isso nenhuma afirmação leva a marca "[só resumo]". A avaliação de (Lu et al., 2024) tem confiança média: as citações literais dela vieram de uma ferramenta de busca que resume o conteúdo e devem ser conferidas no PDF antes de entrar na dissertação. Dois autores "Yu" publicaram em 2020. **(Yu et al., 2020a)** é Tao Yu et al., *Salvaging FL by Local Adaptation*. **(Yu et al., 2020b)** é Tianhe Yu et al., *PCGrad*.

---

## 1. Resumo em 5 linhas

1. Na literatura canônica, *negative transfer* (NT) é um conceito **de desfecho**: usar outra fonte de dados ou conhecimento piora o desempenho no alvo em comparação com o **mesmo algoritmo treinado sem essa fonte** (Rosenstein et al., 2005; Wang et al., 2019; Zhang et al., 2023).
2. Em aprendizado federado (FL), o termo vira "o modelo federado é pior que o modelo treinado só com dados locais" (Bao et al., 2023; Yu et al., 2020a). Essa é exatamente a comparação federado × local-only do projeto.
3. "Conflito de gradientes" (cosseno < 0) é uma **hipótese de mecanismo** para NT. Não é a definição de NT. Até o artigo que formalizou o conflito diz que ele, sozinho, não basta para causar dano (Yu et al., 2020b).
4. O estado da arte em FL multitarefa mede NT como Δ% em relação ao baseline local (Yang et al., 2024; Lu et al., 2024). Nenhum dos trabalhos avaliados mede de fato o conflito entre as atualizações dos clientes.
5. Críticas fortes mostram que métodos "anticonflito" raramente superam uma ponderação fixa bem ajustada (Xin et al., 2022; Kurin et al., 2022). Isso sustenta procurar a causa da NT do projeto em otimização (diluição do passo), e não em conflito.

---

## 2. Como a literatura define o termo

### 2.1 Linha do tempo

**2005: origem operacional (Rosenstein et al., 2005).** Entre as avaliações, este workshop do NIPS é o uso mais antigo do termo. O trabalho de 2019 trata essa formulação como "widely accepted" (Wang et al., 2019). Essas duas fontes não bastam para afirmar que Rosenstein *cunhou* o termo. O que dá para dizer é que ele está entre as formulações fundadoras.
> "transfer learning may actually hinder performance if the tasks are too dissimilar. [...] one challenge for transfer learning research is to develop approaches that detect and avoid negative transfer using very little data from the target task." (Abstract, p. 1)

A operacionalização compara o modelo com transferência a um baseline "B-only" (só a tarefa-alvo) por teste de significância (p<0,05):
> "Black and gray respectively denote positive and negative transfer, and white indicates no statistically significant difference." (legenda da Fig. 2, p. 4)

**2010: base teórica sem usar o termo (Ben-David et al., 2010).** O artigo não diz "negative transfer". Ele descreve uma transição de fase a partir da qual dados da fonte prejudicam o alvo:
> "if there are enough target data (specifically, if mT ≥ D^2 = d/A^2) then no source data are needed, and in fact using any source data will yield suboptimal performance." (Sec. 6, p. 161)

**2019: formalização (Wang et al., 2019, CVPR).** Parte da descrição informal ("transferring knowledge from the source can have a negative impact on the target learner", Sec. 3) e propõe:
> "we define the negative transfer condition (NTC) for any algorithm A as R_{P_T}(A(S,T)) > R_{P_T}(A(∅,T)). [...] we also define the negative transfer gap (NTG) [...] and we say that negative transfer occurs if the negative transfer gap is positive" (Sec. 3, Eqs. 3–4)

O ponto central é que a NT é **relativa a um algoritmo fixo**. Os autores rejeitam duas alternativas: comparar com o *melhor* algoritmo target-only, que não é mensurável, e a versão independente de algoritmo, que nunca se verifica.

**2019: análogo federado em nível de objetivo (Sattler et al., 2019).** O artigo não usa "negative transfer". Ele define clientes **incongruentes** como aqueles para os quais não existe um θ* que seja ótimo local para todos ao mesmo tempo:
> "we will call a set of clients and their data generating distributions φ congruent [...] if they satisfy Assumption 1 and incongruent if they don't." (Sec. I)

**2020: MTL (Standley et al., 2020, ICML).**
> "the quality of predictions are often observed to suffer when a network is tasked with making multiple predictions due to a phenomenon called negative transfer. In fact, multi-task performance can suffer so much that smaller independent networks are often superior." (Sec. 1, p. 1)

As causas aparecem como hipóteses: taxas de aprendizado diferentes, dominância de uma tarefa, interferência de gradientes e uma paisagem de otimização mais difícil. A medida é a variação relativa da perda com **custo computacional igualado**.

**2020: conflito de gradientes, que não é a mesma coisa que NT (Yu et al., 2020b, NeurIPS).** O artigo do PCGrad não usa "negative transfer".
> "We define ϕ_ij as the angle between two task gradients g_i and g_j. We define the gradients as conflicting when cos ϕ_ij < 0." (Sec. 2.2, Def. 1)
> "conflicting gradients are not detrimental on their own. Indeed, simply averaging task gradients should provide the correct solution to descend the multi-task objective." (Sec. 2.2)

**2020: FL, critério de incentivo (Yu et al., 2020a).** Também não usa o termo.
> "many participants gain no benefit from FL because the federated model is less accurate on their data than the models they can train locally on their own." (Abstract)

**2021: afinidade entre tarefas por *lookahead* (Fifty et al., 2021, NeurIPS).** Não usa "negative transfer". O sinal negativo da afinidade indica dano:
> "a negative value of [Z] indicates that the shared parameter update is antagonistic for this task's performance." (Sec. 4.1, Eq. 1)

**2021/2023: survey (Zhang et al., 2023).** Consolida a definição pelo NTG:
> "NT happens when ϵ𝒯(θ(𝒮,𝒯)) > ϵ𝒯(θ(∅,𝒯))" ... "Obviously, NT occurs if the NTG is positive." (Sec. II-C, Def. 1)
> "conflicting gradients among different tasks may induce NT (also known as negative interference)." (Sec. VIII-A)

Ressalva de venue: a avaliação leu a versão arXiv v4 (2021). O venue IEEE/CAA JAS 2023 veio da memória do avaliador e não foi conferido.

**2023: FL (Bao et al., 2023).**
> "the global FL model may even perform worse than the models trained with local data only." (Abstract)
> "the global model significantly degrades, and even performs worse than local training [...] which we refer to as the negative transfer problem." (Sec. 3.1; o trecho do meio não foi reproduzido literalmente pela ferramenta)

**2024: teoria em regressão de alta dimensão (Tahir et al., 2024, preprint).**
> "We refer to the situation 𝒯<0 as negative transfer, since pretraining leads to degradation of the generalization error." (Sec. 2)

**2024: FMTL, sem o termo (Yang et al., 2024; Lu et al., 2024).** Nenhum dos dois usa "negative transfer". Eles falam em "gradient conflict", "task heterogeneity" (Yang et al., 2024) e "model incongruity" / "conflicting parameter updates" (Lu et al., 2024). Na prática, a NT aparece como um Δ% ou Δm negativo em relação ao treino local.

### 2.2 Tabela comparativa

| Autor | Ano | Termo | Definição (resumo) | Nível | Como mede |
|---|---|---|---|---|---|
| Rosenstein et al. | 2005 | negative transfer | modelo com fonte auxiliar significativamente pior que o B-only no alvo | tarefa | acurácia no holdout do alvo, teste p<0,05 por par A→B |
| Ben-David et al. | 2010 | (sem o termo) "suboptimal performance" | com m_T ≥ d/A², qualquer peso na fonte piora o erro no alvo | domínio | cota teórica: d_{HΔH}, λ, α* ótimo |
| Wang et al. | 2019 | NTC / NTG | R_T(A(S,T)) > R_T(A(∅,T)) para um algoritmo A fixo | domínio | NTG = diferença de acurácia entre o target-only e o mesmo algoritmo com a fonte |
| Sattler et al. | 2019 | incongruência | não existe θ* que seja ótimo local para todos os clientes | cliente | cosseno entre atualizações; critério de norma (média pequena, máximo individual grande) |
| Standley et al. | 2020 | negative transfer | co-treino pior que a tarefa sozinha com compute igualado | tarefa | variação relativa da perda; afinidade simetrizada |
| Yu et al. (PCGrad) | 2020b | conflicting gradients | cos φ_ij < 0 (mecanismo, não desfecho) | parâmetro/gradiente | cosseno por batch; Φ; curvatura |
| Yu et al. (Salvaging) | 2020a | (sem o termo) falta de incentivo | modelo federado pior que o local no teste do próprio cliente | cliente | acurácia por participante contra o local |
| Fifty et al. | 2021 | inter-task affinity | Z_{i→j} < 0: o passo de i aumenta a perda de j | tarefa (via parâmetro) | razão de perdas antes/depois de um passo de *lookahead* |
| Zhang et al. | 2023 | NT / NTG | igual a Wang et al. | domínio | NTG |
| Bao et al. | 2023 | negative transfer problem | modelo FL pior que o modelo local, por cliente | cliente | IPR, RSD |
| Tahir et al. | 2024 | negative transfer / transferability | 𝒯 = E[R_sc − R_tx] < 0 | domínio/tarefa | MSE esperado contra o modelo treinado do zero |
| Yang et al. (FMTL-Bench) | 2024 | (sem o termo) gradient conflict, heterogeneidade | implícita: Δ% < 0 contra o Local | cliente × tarefa | Δ%, ΔG |
| Lu et al. (FedHCA2) | 2024 | model incongruity, conflicting updates | implícita: produto interno baixo entre deltas; Δm < 0 | cliente/parâmetro | Δm contra o local; conflito só na teoria |

---

## 3. Divergências e ambiguidades entre definições

1. **Qual é o baseline.** As definições usam baselines diferentes:
   - target-only com o *mesmo algoritmo* (Wang et al., 2019; Zhang et al., 2023);
   - B-only (Rosenstein et al., 2005);
   - tarefa sozinha com *compute igualado* (Standley et al., 2020);
   - modelo treinado do zero (Tahir et al., 2024);
   - modelo local, por cliente (Bao et al., 2023; Yu et al., 2020a; Yang et al., 2024).

   Wang et al. (2019) alertam que comparar com o *melhor* target-only é imensurável e classifica como NT até um algoritmo que nunca usa a fonte. Para o projeto, o baseline adequado é o local-only pareado (mesmo algoritmo, budget e seeds).
2. **Desfecho × mecanismo.** NTG, Δ%, IPR e 𝒯 medem desempenho. Cosseno < 0 (Yu et al., 2020b), incongruência (Sattler et al., 2019) e produto interno baixo entre deltas (Lu et al., 2024) descrevem geometria de otimização. O survey trata o conflito como *possível causa* de NT, não como definição (Zhang et al., 2023). O próprio PCGrad diz que o conflito só prejudica junto com grande diferença de magnitude e alta curvatura (Yu et al., 2020b).
3. **Assimetria.** A transferência é assimétrica (Fifty et al., 2021; Standley et al., 2020). No experimento de Standley et al., a tarefa auxiliar Normals melhorou a parceira em 15 de 16 modelos e ainda assim piorou ela mesma. Definições por par simetrizado escondem isso.
4. **Binário × graduado.** A incongruência é binária (Sattler et al., 2019). O NTG e o Δ% são contínuos. Rosenstein et al. (2005) classificam em três categorias por significância.
5. **Dependência de contexto.** A NT muda de sinal com a quantidade de dados rotulados no alvo (Wang et al., 2019; Rosenstein et al., 2005; Ben-David et al., 2010; Tahir et al., 2024), com a capacidade e com o tamanho do dataset (Standley et al., 2020), e com hiperparâmetros (Fifty et al., 2021, Tabela 3). Por isso a NT não é uma propriedade fixa de um par de domínios.
6. **Proximidade de domínio não prevê NT.** A afinidade multitarefa não se correlaciona com a afinidade de *transfer* da Taskonomy (Standley et al., 2020). Distâncias KL e Wasserstein podem ser arbitrariamente grandes com transferência positiva (Tahir et al., 2024). Isso diverge da visão de que "a divergência entre domínios é a raiz da NT" (Zhang et al., 2023; Wang et al., 2019).
7. **Terminologia dispersa.** Os mesmos fenômenos aparecem como "task interference" (Xin et al., 2022), "negative interference" (Zhang et al., 2023), "incongruence" (Sattler et al., 2019) e "model incongruity" (Lu et al., 2024). Vários trabalhos de referência simplesmente não usam o termo "negative transfer": (Yang et al., 2024; Lu et al., 2024; Kurin et al., 2022; Xin et al., 2022; Yu et al., 2020b; Fifty et al., 2021).

---

## 4. Como é medido (métricas)

**Métricas de desfecho (definem NT)**
- **NTG** = erro_com_fonte − erro_sem_fonte; NT ocorre se NTG > 0 (Wang et al., 2019; Zhang et al., 2023). O survey observa que o NTG "may not always be computable", por exemplo em adaptação de domínio não supervisionada.
- **Δ% / Δm**: média entre tarefas da variação relativa contra o baseline local, com o sinal corrigido para métricas em que menor é melhor (Yang et al., 2024, Eq. 5; Lu et al., 2024).
- **IPR**, a fração de clientes em que o modelo FL supera o local, e **RSD**, o desvio-padrão dos ganhos por cliente (Bao et al., 2023).
- **Transferabilidade 𝒯** = E[R_sc − R_tx] (Tahir et al., 2024).
- **Afinidade por par com compute igualado** (Standley et al., 2020).
- **Contagem de participantes prejudicados**: número de clientes com acurácia federada < acurácia local (Yu et al., 2020a).
- **Posição em relação à frente de Pareto** da escalarização, obtida por varredura de pesos (Xin et al., 2022).

**Métricas de mecanismo (diagnosticam causas candidatas)**
- **Cosseno entre gradientes**, com a porcentagem de iterações com cosseno positivo; **similaridade de magnitude Φ**; **curvatura multitarefa** H (Yu et al., 2020b).
- **Cosseno entre atualizações de clientes**; α_intra^min, α_cross^max e **gap de separação** g(α); critério de norma ‖média Δ‖ < ε1 com max‖Δ_i‖ > ε2 (Sattler et al., 2019).
- **Afinidade por lookahead** Z_{i→j}. Agrupar tarefas por cosseno (CS) foi pior que agrupar por Z (Fifty et al., 2021).
- **Norma da atualização somada** ‖Σ_i ∇L_i‖ nos parâmetros compartilhados, usada como diagnóstico de sub-otimização (Kurin et al., 2022, §5.3).
- **"Positive sign purity"** do GradDrop, a pureza de sinal por coordenada (descrita em Kurin et al., 2022).
- **H-divergência empírica** estimada com um classificador de domínio (Ben-David et al., 2010, Lema 2), ou com um discriminador federado entre pares de clientes (Bao et al., 2023).

**Observação.** Nenhum dos 15 trabalhos usa a razão de cancelamento agregada 1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖ com separação intra/inter-dataset e referência ortogonal. A métrica mais próxima é o critério de norma de Sattler et al. (2019).

---

## 5. Estado da arte de mitigação

As avaliações incluem poucos métodos de mitigação publicados entre 2023 e 2026: só FedHCA2 e FedCollab, além dos baselines do FMTL-Bench. Isso limita o que se pode afirmar como estado da arte recente (ver seção 8).

| Método | Ano | Ideia | Onde se aplica | Evidência |
|---|---|---|---|---|
| FedHCA2: agregação *conflict-averse* + cross-attention + pesos aprendíveis (Lu et al., 2024) | 2024 | CAGrad aplicado aos deltas do encoder dos clientes; atenção sobre os deltas dos decoders; α_i/β_i aprendíveis | FL multitarefa (encoder compartilhado, heads locais) | Δm +2,18% (FedAvg −11,76%) e +0,75% (FedPer −0,16%). Ganhos pequenos e uma seed. A restrição c·‖média‖ não recupera magnitude perdida |
| FMTL-Bench (baselines) (Yang et al., 2024) | 2024 | benchmark de FedAvg, FedProx, FedAMP, FedRep, PCGrad, CAGrad, MaT-FL, FedMTL × arquiteturas SD/MD/TC | FL multitarefa | Nenhum método vence sempre. PCGrad "generally underperforms". Compartilhar só o encoder e usar decoder TC suavizam o cenário cross-domain (NIID-7: FedAvg-SD −27,21%, FedAvg-TC −4,07%) |
| FedCollab (Bao et al., 2023) | 2023 | escolhe coalizões de clientes minimizando uma cota (distância entre distribuições + quantidade) | FL global/personalizado | IPR → ~100% em 3 cenários sintéticos; supera IFCA e KMeans. Compartilha o modelo inteiro (sem FedPer) e tarefa única |
| TAG (Fifty et al., 2021) | 2021 | afinidade por lookahead durante uma execução, depois agrupamento ótimo de tarefas | MTL centralizado | Taskonomy −10,0% de perda contra MTL, 11,6× mais rápido que HOA; grupos > UW/GradNorm/PCGrad |
| Agrupamento ótimo + HOA (Standley et al., 2020) | 2020 | treina subconjuntos e escolhe grupos sob um orçamento de inferência | MTL centralizado | vence a rede única, as redes independentes, MGDA e GradNorm em 4 configurações; custo exponencial |
| PCGrad (Yu et al., 2020b) | 2020 | projeta gradientes conflitantes no plano normal | MTL centralizado | Meta-World MT10 resolvido; NYUv2 melhor em 8/9 métricas. Não confirmado por (Kurin et al., 2022) nem por (Yang et al., 2024) |
| Adaptação local pós-hoc (Yu et al., 2020a) | 2020 | fine-tuning, freeze-base, EWC ou KD do modelo global em cada cliente | FL | usuários do Reddit prejudicados (BASIC) caem de 9,22% para 0,04%. Mitigação de desfecho, não de causa |
| Clustered FL (Sattler et al., 2019) | 2019 | bipartição recursiva pelo cosseno quando a federação estagna | FL | CIFAR com rótulos permutados: ~20% → ~60%. Conflitos sintéticos com cosseno ≈ −1 |
| Discriminator gate (Wang et al., 2019) | 2019 | reponderação por densidade P_T/P_S com discriminador conjunto | adaptação de domínio | DANN: NTG +3,0 → −7,4 (+10,4 pp). Fonte e alvo únicos, classificação |
| Escalarização bem ajustada / regularização (Xin et al., 2022; Kurin et al., 2022) | 2022 | pesos fixos ajustados por grid, mais dropout/ℓ2/early stopping | MTL centralizado | iguala ou supera MGDA, GradNorm, PCGrad, IMTL e RLW com custo muito menor |

Nenhuma das avaliações cobre model merging. A coluna "merging" do formato pedido fica sem entradas.

---

## 6. Críticas e resultados negativos

- **Métodos anticonflito não superam uma escalarização ajustada.** "all of the MTO algorithms in our study simply yield performance trade-off points on the scalarization Pareto front" (Xin et al., 2022). "the effect of tuning hyper-parameters is much more significant than the effect of the MTO algorithm choice" (Xin et al., 2022).
- **A hipótese do conflito perde força.** Um GradDrop "agnóstico de sinal" iguala o GradDrop original, o que "seriously undermines the conflicting gradient hypothesis" (Kurin et al., 2022, App. B.4). Segundo os autores, MGDA, IMTL e PCGrad funcionam em parte como regularizadores ou por sub-otimização (Kurin et al., 2022). Eles atribuem a dificuldade da MTL a "incorporating adequate regularization" (§5.1) e reconhecem que isso é "only a partial explanation".
- **Em FL multitarefa, a cirurgia de gradiente falha.** PCGrad "generally underperforms" e só FedAvg e FedProx são robustos quando os clientes ficam menores e mais numerosos (Yang et al., 2024). O benchmark não explica por quê.
- **Ganhos do SOTA sobre o local-only são marginais.** Ficam na faixa de 1–2% de Δm, e algumas tarefas ainda perdem para o local (Lu et al., 2024).
- **Conflito não é condição suficiente** (Yu et al., 2020b, Sec. 2.2), e **cosseno é um proxy imperfeito** de transferência (Fifty et al., 2021).
- **Avaliação frágil.** "rerunning multiple seeds is insufficient" quando os hiperparâmetros vêm de um grid esparso (Xin et al., 2022). Vários trabalhos centrais usam uma única seed (Yang et al., 2024; Lu et al., 2024; Standley et al., 2020; Yu et al., 2020a).
- **A NT é relativa ao baseline.** Com mais dados rotulados no alvo, o baseline melhora e a NT aumenta (Wang et al., 2019; Ben-David et al., 2010; Tahir et al., 2024). A regularização ridge aumenta a região de NT (Tahir et al., 2024).
- **As medidas de distância entre domínios são insuficientes** para prever o sucesso da transferência (Tahir et al., 2024; Standley et al., 2020).

---

## 7. O que isso implica para o nosso diagnóstico

Cautela geral: os p-valores do projeto vêm de Wilcoxon pareado sobre 8 pares cliente × fold não independentes, com uma única seed. São **exploratórios** e não sustentam afirmações de significância.

1. **"Federado < local-only" é negative transfer no sentido canônico, com uma condição.** Por cliente e tarefa, NTG = erro_FedPer − erro_local-only (Wang et al., 2019; Zhang et al., 2023). O enquadramento federado coincide (Bao et al., 2023; Yu et al., 2020a). A condição é que os dois braços sejam o *mesmo algoritmo*: mesmo modo de budget, seeds pareadas e mesma seleção de modelo. Wang et al. (2019) alertam que mudar o budget ou a agregação muda A e, portanto, o NTG. A tese pode relatar Δ% por dataset (Yang et al., 2024) e IPR/RSD por cliente × tarefa (Bao et al., 2023), calculados a partir dos `*_test_results.csv` já existentes.
2. **A razão de cancelamento e o cosseno não medem NT. Eles testam um mecanismo candidato.** Cancelamento inter-dataset baixo, perto da referência ortogonal, não é evidência de conflito no sentido do PCGrad (cos < 0; Yu et al., 2020b) nem de incongruência no sentido do CFL, em que o cancelamento vem de atualizações antiparalelas (Sattler et al., 2019). A leitura prudente é: **NT observada; conflito direcional como causa principal pouco sustentado pelos dados atuais.** Isso não elimina o conflito como contribuição secundária. O cosseno global pode esconder conflito por bloco ou em fases específicas do treino. Além disso, as garantias do CFL valem só perto de pontos estacionários.
3. **Diluição do passo como mecanismo de otimização.** Com atualizações quase ortogonais, a média já encolhe o passo. Para normas iguais e pesos uniformes, ‖média‖ ≈ ‖Δ‖/√n, e é isso que a referência ortogonal captura. Vale verificar se o fator ~10× medido é compatível com essa referência, dados o número de clientes e os pesos reais. Se o fator for maior, há outro componente: pesos assimétricos ou normas desiguais (o Φ de Yu et al., 2020b). Na literatura, isso se aproxima de:
   - a leitura de Kurin et al. (2022), em que mudança de magnitude age como sub-otimização ou regularização implícita do tronco, diagnosticada pela norma da atualização;
   - o "fator algoritmo" de Wang et al. (2019) e Zhang et al. (2023), que não separam otimização de divergência;
   - a restrição da contribuição de cada participante em Yu et al. (2020a), com clipping DP e mediana.

   Nenhum trabalho avaliado define "diluição do passo" como forma de NT. A tese deve apresentá-la como **hipótese de mecanismo própria do projeto**.
4. **Previsões coerentes com o diagnóstico, a testar (não confirmadas):**
   - Se o conflito não é o custo dominante, métodos que só ajustam a direção (PCGrad, CAGrad, o HCA do FedHCA2) não devem fechar o gap. Isso é consistente com (Yang et al., 2024) e (Kurin et al., 2022). A restrição c·‖média‖ do HCA não restaura magnitude (Lu et al., 2024).
   - Ajustar o tamanho efetivo do passo do servidor antes de afirmar NT segue a recomendação de "tune before claiming" (Xin et al., 2022).
   - Clientes grandes (ISIC, SIIM, TCGA) devem ter NTG maior e o BUSI, pequeno, é onde a transferência positiva é mais plausível (Ben-David et al., 2010; Wang et al., 2019; Tahir et al., 2024; Bao et al., 2023). Basta reportar o NTG em função do tamanho do cliente.
5. **Diagnósticos complementares sugeridos pela literatura:**
   - Afinidade federada no estilo TAG. Comparar Z_{j→j} (delta próprio) com Z_{agg→j} (delta agregado) separa a perda por diluição do dano direcional (Z_{i→j} < 0), coisa que a razão de cancelamento não consegue. É uma adaptação multi-step, não a Eq. 1 exata (Fifty et al., 2021).
   - Gap de separação entre cossenos intra e inter-dataset, para decidir se agrupar por dataset se justifica (Sattler et al., 2019).
   - Braço de federação só intra-dataset ou por coalizão (Bao et al., 2023; Sattler et al., 2019).
   - Braço FedPer + fine-tuning pós-hoc. Se esse braço ainda perder para o local-only, o gap não se explica por falta de personalização (Yu et al., 2020a).
   - H-divergência entre features do tronco como covariável descritiva (Ben-David et al., 2010), com a ressalva de que distância não prevê transferência (Tahir et al., 2024).
6. **Coerência com o SOTA.** No FedHCA2, o FedPer fica levemente abaixo do local (Δm −0,16%; Lu et al., 2024). No cenário cross-domain com clientes de tarefa única, o FedAvg perde muito para o local (Yang et al., 2024). O resultado do projeto não é anômalo. Esses benchmarks usam imagens naturais e não informam diretamente sobre domínios médicos.

---

## 8. Lacunas abertas

1. **Nenhuma definição de NT para FL multitarefa com compartilhamento parcial** (tronco compartilhado com stems e heads personalizados). As definições formais são de fonte e alvo únicos (Wang et al., 2019; Zhang et al., 2023), e o FMTL usa um Δ% implícito (Yang et al., 2024; Lu et al., 2024).
2. **Nenhuma decomposição do NTG em mecanismos** (conflito direcional × magnitude/diluição × divergência de distribuição × regularização). Zhang et al. (2023) listam a identificação dos fatores exatos da NT como direção futura.
3. **Nenhuma medida empírica de conflito entre clientes no SOTA de FMTL.** FedHCA2 invoca o conflito só na teoria (Lu et al., 2024) e o FMTL-Bench não o mede (Yang et al., 2024). A razão de cancelamento com referência ortogonal parece não ter equivalente nas fontes avaliadas.
4. **Falta explicar por que a cirurgia de gradiente falha em FL** (Yang et al., 2024).
5. **Domínios médicos cross-modality e pares segmentação + classificação** não aparecem em nenhum dos trabalhos avaliados.
6. **Rigor estatístico.** Faltam múltiplas seeds e variância de hiperparâmetros (Xin et al., 2022) na maior parte dos trabalhos de FMTL.
7. **Cobertura desta busca.** Há poucos métodos de 2023–2026 e nada sobre model merging. Isso é uma lacuna da própria síntese, não necessariamente da literatura. Recomenda-se uma busca dirigida antes de afirmar qual é o estado da arte atual.

---

---

Fontes: [`fontes.md`](fontes.md). Fichas de leitura por artigo: [`fichas.md`](fichas.md).
