# Plano de desenvolvimento — próximas etapas

Documento vivo que registra e acompanha as próximas atividades do projeto.
Criado em 2026-10-04.

## Fluxo de trabalho

1. **Planejamento macro** (este documento): objetivo, entregáveis e dependências de cada etapa.
2. Uma etapa por vez. Antes de implementar, preencher a seção **Detalhamento** da etapa com o máximo
   de precisão: arquivos, funções, testes e comandos.
3. Cada implementação tem a sua branch (`feat/<etapa>`, `exp/<ablação>`, `docs/<tema>`). O status deste
   arquivo é atualizado nessa branch, e o merge na `main` leva junto o status.
4. Ao concluir uma etapa, registrar o resultado no **Registro** dela e detalhar a próxima.

### Legenda de status

| Ícone | Significado |
|---|---|
| ⬜ | pendente |
| 🟨 | planejando (detalhamento em andamento) |
| 🟦 | em implementação |
| ⏳ | aguardando treino executado pelo usuário |
| ✅ | concluído |

### Resumo

| # | Etapa | Status | Branch | Atualizado |
|---|---|---|---|---|
| E0 | Métodos adicionais de verificação de conflito de gradientes | ⏳ | `feat/gradient-conflict` | 2026-10-09 |
| E1 | Validação do treinamento | 🟨 | `feat/training-validation` | 2026-10-08 |
| E2 | App local de monitoramento e comparação | ⬜ | `feat/monitor-app` | — |
| E5 | Validação das fontes (consolidação) | ⬜ | `docs/sources` | — |
| E3 | Ablação: loss fed multitask multi-dataset | ⬜ | `exp/loss-ablation` | — |
| E4 | Ablação: pré-processamento de imagens | ⬜ | `exp/preprocessing-ablation` | — |

Ordem de execução: **E0 → E1 → E2 → E5 → E3 → E4**.
- E0 foi incluída em 2026-10-08 a pedido do usuário. A métrica atual de conflito dá ≈ 0 para todos
  os pares de clientes, inclusive pares que deveriam estar alinhados, e é preciso saber se isso é
  ausência real de conflito ou cegueira da métrica antes de usá-la na E1 (D9) e na E3.
- Validar o treinamento antes de construir em cima dele.
- O app acelera a leitura e a comparação das ablações.
- As fontes verificadas embasam a escolha dos braços das ablações.

---

## Regras transversais

Valem para todas as etapas (fonte: `AGENTS.md`, seção "Non-negotiables").

- A partição federada congelada (`federated_mapping.csv`) **não é regerada** sem intenção explícita.
  Todos os braços de uma comparação leem a mesma partição.
- `Curated_BUSI_preprocessing.py` continua com `INTER_NEAREST`.
- Caminhos de dataset vêm sempre de `src/dataset/paths.py`. Os arquivos `mapping.csv` e
  `federated_mapping.csv` nunca são editados à mão.
- Uma ablação de pré-processamento que altere imagens gravadas usa uma **nova variante**
  (`data/<ds>/<variant>/`) e nunca sobrescreve `processed_128`.
- Comparar apenas braços com o mesmo modo de budget (`local_training.mode`).
- p-valores são exploratórios (`exploratory_only_non_independent_client_fold_pairs`). Nunca escrever
  "estatisticamente significativo".
- `numpy<2`. Nada de `sigmoid` antes do critério DICE.

### Protocolo de handoff de treino

O treino completo é executado pelo usuário. Sempre que uma etapa precisar de um, ela registra:

1. **O que alterar**: manifest/arms em `studies/` e overrides de `src/config.yaml`, com o diff exato.
2. **Comando exato**, por exemplo `python -m src.experiments.study_runner --seed-profile operational --arms ...`.
3. **Tempo/recursos estimados**, quando houver referência em runs anteriores.
4. **Artefatos esperados de volta**: diretório do run ou estudo, e quais arquivos serão analisados.
5. **Critério de sucesso** do treino: o que indica que ele rodou corretamente.

---

## E0 — Métodos adicionais de verificação de conflito de gradientes

**Objetivo.** Buscar na literatura e implementar **mais dois métodos** de medir conflito de gradiente
entre clientes/tarefas, complementares ao cosseno entre deltas que já existe. Antes disso, validar se
a métrica atual é capaz de detectar conflito.

**Situação atual.**
- `FedPerStrategy._gradient_conflict` (`src/federated/server.py:315`): cosseno par a par entre os
  **deltas de trunk** (trunk devolvido − trunk enviado) de cada cliente, por bloco e total. O
  resultado é gravado em `aggregation_history.json`.
- `cancellation_from_gram` (`src/federated/negative_transfer.py`): razão de cancelamento
  `1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖`, comparada com a referência ortogonal.
- Fonte que motivou a métrica: FedBone (arXiv 2306.17465).

**Evidência que motivou a etapa** (run de 29/09; 200 rodadas; 16 clientes, isto é, 120 pares):

| bloco | média cos | p5 | p95 | fração < 0 | fração < −0.1 |
|---|---|---|---|---|---|
| encoder2 | −0.001 | −0.033 | +0.031 | 0.53 | 0.000 |
| encoder5 | +0.001 | −0.009 | +0.012 | 0.46 | 0.000 |
| shared_total | +0.001 | −0.009 | +0.013 | 0.48 | 0.000 |

- A distribuição é **idêntica** para pares intra-dataset e inter-dataset, inclusive entre clientes do
  mesmo dataset, da mesma tarefa e com partição quase IID (Dirichlet α=100). Esses pares deveriam
  ter cosseno claramente positivo.
- Conclusão: hoje é impossível distinguir "sem conflito" de "métrica cega". O cancelamento
  `overall ≈ orthogonal_reference` (A6 da E1) é consequência direta disso, e não uma evidência
  independente.

**Hipóteses a testar** (para a métrica atual ficar cega):
- H1. **Adam:** o passo é normalizado por coordenada (≈ lr·sinal(m)/√v) e usa um estado de
  otimizador por cliente. O delta vira quase "sinal + ruído" e perde a direção do gradiente.
- H2. **Acúmulo de 10 steps:** o delta de uma rodada soma 10 passos sobre batches diferentes. É um
  deslocamento, não um gradiente no ponto enviado.
- H3. **Stem personalizado por cliente:** o `encoder1` não é compartilhado e cada cliente tem a sua
  inicialização (seed por `client_id`). Assim, o espaço de entrada do trunk difere entre clientes,
  mesmo dentro de um dataset.
- H4. **Bug de medição:** referência `_sent_arrays` errada, ordem de tensores trocada, precisão,
  parâmetros que não são pesos treináveis.
- H5. **Dimensionalidade:** com ~10⁷ parâmetros, vetores quase independentes ficam quase ortogonais.
  O cosseno global dilui um conflito localizado em poucas camadas ou canais.

### Detalhamento

#### E0.1 — Validação da métrica atual (controles)

1. **Controle sintético** (teste unitário de `_gradient_conflict` e `cancellation_from_gram`):
   - deltas idênticos → cos = 1;
   - deltas opostos → cos = −1;
   - deltas ortogonais → cos = 0;
   - ordem de tensores e `_sent_arrays` corretos (H4).
2. **Controle positivo real** (CPU, curto; eu executo): dois clientes **com os mesmos dados** e o
   mesmo stem, 1 step e SGD (sem Adam). O cosseno esperado é ≈ 1. Depois, religar um fator por vez
   (Adam → H1, 10 steps → H2, stems diferentes → H3) e medir quanto cada um derruba o cosseno.
3. Resultado: tabela "fator → cosseno observado", que diz se a métrica atual é utilizável e em que
   condições.

#### E0.1 — Resultado (executado em 2026-10-08)

**Script:** `scripts/gradient_conflict_controls.py`.
- Reconstrói o estado de um run terminado: trunk global + stem/cabeças/Adam de cada cliente.
- Também roda com `--at-init`, que reconstrói o estado da rodada 1.
- Mede o cosseno com o **próprio** `FedPerStrategy._gradient_conflict`, ligando um fator por vez.
- Cada condição é repetida sobre os batches da rodada seguinte. Isso dá o *teto intra-cliente*:
  o quanto um cliente concorda consigo mesmo.
- Artefatos (não versionados) em `runs/validation/gradient_conflict_controls/20260929_run_{all_clients,at_init}/`.

**Controles sintéticos:** já existiam em `tests/test_model_split_server.py` e `tests/test_negative_transfer.py`
(idênticos → 1, opostos → −1, ortogonais → 0, delta nulo → indefinido, blocos separados).
O trunk tem 10 tensores e 4,66 M de parâmetros, **todos treináveis**: nenhum buffer entra no delta.

**Média do cosseno, pares do mesmo dataset e mesmas tarefas** (`intra_same_tasks`, 20 pares):

| condição | estado final (rodada 200) | inicialização (rodada 1) | teto intra-cliente (final / init) |
|---|---|---|---|
| `real_adam10` (o que o servidor mede) | 0.000 | 0.001 | 0.54 / 0.42 |
| `adam1` (sem acúmulo de steps) | 0.001 | 0.001 | 0.92 / 0.26 |
| `sgd10` (sem Adam) | 0.012 | −0.002 | 0.28 / 0.64 |
| `grad10` (gradiente bruto no trunk enviado) | 0.013 | −0.003 | 0.29 / 0.65 |
| `grad10_refpers` (**stem/cabeças iguais** dentro do dataset) | 0.061 | **0.619** (mediana 0.86) | 0.27 / 0.63 |

Por par, na inicialização, com `grad10` → `grad10_refpers`:
- ISIC seg × seg: 0.00 → **0.99**;
- SIIM × SIIM: 0.00 → **0.86**;
- TCGA: 0.00 → **0.85**;
- BUSI: 0.00 → **0.41**;
- ISIC cls × cls: 0.00 → 0.00. Nesse caso o gradiente de classificação no trunk inicial é dominado
  por ruído; ver o teto.

**Veredito sobre as hipóteses:**
- **H4 (bug de medição): refutada.** A métrica reproduz o valor do servidor, e cada cliente concorda
  consigo mesmo (0.30–0.66).
- **H1 (Adam) e H2 (10 steps): refutadas como causa.** Com o gradiente bruto no ponto enviado, sem
  Adam e sem acúmulo, o cosseno entre clientes continua ≈ 0.
- **H3 (stem/cabeças personalizados): confirmada.** Clientes com os mesmos dados e as mesmas tarefas
  produzem gradientes de trunk quase **paralelos** (0.85–0.99) quando compartilham stem e cabeças,
  e **ortogonais** (≈ 0) com os próprios. Isso vale desde a rodada 1: os stems/cabeças são
  inicializados com uma seed diferente por cliente (`stable_client_seed(..., client_id)` em
  `src/federated/client.py:129`), então cada cliente entrega ao trunk uma codificação aleatória
  diferente da mesma imagem.
- **H5 (diluição por dimensionalidade):** não é a explicação principal, já que o mesmo trunk dá
  0.99 com a personalização igualada. Continua relevante por bloco.

**Implicações:**
1. O cosseno entre deltas **não está cego**. Ele mede um fato do desenho atual: os 16 clientes atualizam
   o trunk em direções mutuamente ortogonais. Portanto, "sem conflito" é uma leitura errada; a leitura
   correta é **"sem interação"**. A média de 16 updates ortogonais apenas encolhe cada um (cancelamento
   ≈ referência ortogonal, A6). Por construção, o FedAvg quase não combina informação entre os clientes.
2. Isso afeta a interpretação científica de federado vs local e de `negative_transfer`. O achado deve
   ser levado à E3 e ao artigo, com a cautela de que é **um run, uma seed, um holdout**.
3. Possível correção de desenho, **a decidir pelo usuário; não implementar sem aprovação**: inicializar
   stem/cabeças com seed por **dataset** em vez de por cliente, e/ou federar o stem dentro de cada
   dataset. Isso muda a configuração científica e, portanto, exigiria um braço novo, sem invalidar
   os runs existentes.
4. Para os métodos novos (E0.2/E0.3): todo cosseno entre clientes deve ser lido contra o **teto
   intra-cliente**, porque ruído de gradiente limita o cosseno observável. No estado final, alguns
   clientes concordam consigo mesmos em apenas ≈ 0.0–0.1. Um cosseno normalizado
   `cos_ij / sqrt(teto_i · teto_j)` é candidato natural.

#### E0.1b — Run diagnóstico: seed da personalização por dataset

**Pergunta.** Iniciar stem e cabeças com a mesma seed para todos os clientes de um dataset (opção 1)
mantém o alinhamento ao longo do treino, ou os stems se afastam?

**Implementação.**
- Nova flag `federated.personalized_init_seed: client | dataset`.
- O padrão é `client`, que reproduz bit a bit a seed histórica (teste de regressão em
  `tests/test_model_split_server.py`).
- Código: `initialization_seed` em `src/federated/client.py`, validação em `src/federated/config.py`;
  a política fica registrada no `metadata.yaml` do cliente.
- O local-only usa a mesma política, então o pareamento entre braços se mantém.

**Desenho.**
- Configuração idêntica à do run de 29/09: mesma partição congelada, 4 datasets, 16 clientes, BF16,
  10 steps/rodada.
- Mudanças: apenas `rounds: 30` e a flag.
- Dois braços: `runs/validation/personalized_init_seed/{client,dataset}`.
- Runner: `runs/validation/personalized_init_seed/run_both.sh`.

**Leitura.**
- Trajetória por rodada do cosseno intra-dataset (pares com as mesmas tarefas), que vem do
  `aggregation_history.json`.
- Excesso de cancelamento sobre a referência ortogonal.
- Distância relativa entre os stems dos clientes de um dataset na rodada 30.
- É diagnóstico: 30 rodadas não servem para comparar desempenho.

**Resultado (2026-10-09; um run por braço, uma seed, um holdout; descritivo).**

Cosseno médio entre clientes do mesmo dataset e com as mesmas tarefas (`shared_total`):

| rodada | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| `client` | 0.000 | 0.013 | 0.032 | 0.044 | 0.018 | 0.010 | 0.011 | 0.009 | 0.010 |
| `dataset` | **0.387** | 0.488 | 0.537 | **0.569** | 0.520 | 0.466 | 0.387 | 0.289 | **0.252** |

Por dataset (rodadas 1–5 → 26–30), na política `dataset`:
- TCGA: 0.58 → 0.53;
- SIIM: 0.54 → 0.37;
- ISIC seg: 0.49 → 0.20;
- BUSI: 0.44 → 0.07.

Na política `client`, todos ficam entre 0.00 e 0.06 nas duas janelas.

Por bloco (rodadas 26–30, `dataset`): encoder2 0.47, encoder3 0.30, encoder4 0.18, encoder5 0.08,
bottleneck 0.04. O alinhamento se concentra nos blocos rasos.

**Os stems não se afastam.** Na rodada 30, entre clientes do mesmo dataset:
- política `dataset`: cosseno dos pesos do `encoder1` de 0.999 a 1.000, distância relativa de 1% a 4%;
- política `client`: cosseno ≈ 0.02, distância relativa ≈ 1.4.

O decaimento do alinhamento, portanto, **não vem de deriva do stem**. A explicação mais provável é a
queda da razão sinal/ruído do gradiente à medida que a loss cai, coerente com o teto intra-cliente
baixo medido na rodada 200. Os blocos profundos também se especializam mais. **Não há necessidade
demonstrada da opção 2** (ressincronizar o stem) ao menos até a rodada 30.

**O que não muda:**
- O cosseno **inter-dataset** é ≈ 0 nas duas políticas (r1–5: 0.004 / 0.007; r26–30: ≈ 0.000).
  Datasets de modalidades diferentes têm stems diferentes por construção.
- ISIC seg × ISIC cls (`intra_diff_tasks`) também fica ≈ 0, mesmo com stem inicial igual. As duas
  tarefas não interagem pelo trunk, nem de forma construtiva nem destrutiva.

**Cancelamento acima da referência ortogonal** (negativo significa agregação construtiva):
- `client`: −0.014 (r1–5), +0.001 (r26–30);
- `dataset`: **−0.146** (r1–5), −0.037 (r26–30).

**Métricas de teste na rodada 30** (apenas indicativas):
- Dice positivo, `client` → `dataset`: BUSI 0.488 → 0.538, TCGA 0.408 → 0.446, ISIC 0.703 → 0.698,
  SIIM 0.015 → 0.014.
- O mais visível é a **dispersão entre clientes** do mesmo dataset, que cai muito:
  - BUSI seg: ±0.107 → ±0.036;
  - TCGA seg: ±0.033 → ±0.003;
  - SIIM seg: ±0.017 → ±0.002.
- A acurácia de classificação fica praticamente igual.

**Conclusão.** A opção 1 transforma o FedAvg intra-dataset de "média de vetores ortogonais" em
agregação construtiva. O alinhamento é forte no início e decai (0.57 → 0.25 em 30 rodadas) sem que
os stems se separem. Entre datasets não muda nada. Um braço completo (200 rodadas) com
`personalized_init_seed: dataset` é necessário para medir o efeito em desempenho. É decisão do
usuário e o treino é executado por ele.

#### E0.2 — Busca, prototipagem e escolha de 2 métodos

**O que a E0.1/E0.1b ensinou e que muda os requisitos:**
1. O cosseno entre deltas é válido, mas só enxerga a geometria de **primeira ordem**. Ele não
   distingue "sem conflito" de "sem interação", e não capta a **diluição** pela média (cada cliente
   do BUSI pesa 12,5%).
2. Nenhum controle até agora produziu cosseno **negativo**. A capacidade de detectar conflito, ou
   seja, a sensibilidade negativa, **nunca foi testada**.
3. O ruído de gradiente limita o valor observável. Todo método precisa ser lido contra o **teto
   intra-cliente**.
4. Entre datasets, a interação é ≈ 0 por construção (stems diferentes). Um método útil para a
   pergunta "há transferência negativa?" precisa medir o **efeito na loss**, e não apenas a direção.

**Critérios de escolha (pesos para a matriz de decisão):**

| # | Critério | Peso |
|---|---|---|
| C1 | Responde à pergunta de transferência negativa, isto é, mede efeito na loss/desempenho ou é um proxy validado disso | alto |
| C2 | Independente do otimizador e do acúmulo de steps (H1/H2) | alto |
| C3 | Passa nos controles positivo **e** negativo (abaixo) | eliminatório |
| C4 | Custo viável online: ≤ 20% de acréscimo no tempo da rodada, nas rodadas amostradas | médio |
| C5 | Fonte primária verificada, com uso prévio em MTL ou FL | médio |
| C6 | Complementar ao cosseno atual e ao outro método escolhido | médio |

**Candidatos a verificar** (fórmulas, referências e venues **ainda não conferidos** nas fontes primárias):

| id | Método | O que mede | Fonte a verificar | Custo previsto no nosso setup |
|---|---|---|---|---|
| M1 | Cosseno + **similaridade de magnitude** (GMS) do gradiente bruto no trunk recebido, sobre batches-sonda fixos | Conflito direcional no ponto enviado, sem Adam/steps. A magnitude capta o domínio de um cliente/tarefa. | Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), NeurIPS 2020 | k forward/backward extras por cliente nas rodadas amostradas (k = 10 ⇒ ~2× a rodada). Vetor de 4,66 M (18,6 MB) por cliente. |
| M2 | **Afinidade lookahead**: Z(i→j) = 1 − L_j(θ + Δᵢ) / L_j(θ) | Efeito funcional do passo de i na loss de j. Inclui ordem superior; o sinal negativo é transferência negativa. | Fifty et al., "Efficiently Identifying Task Groupings in Multi-Task Learning" (TAG), NeurIPS 2021 | n² = 256 forwards em batch-sonda por rodada amostrada (só forward). Exige distribuir os 16 deltas aos clientes. |
| M2' | **Ganho da federação** por cliente: L_j(θ_agregado) vs L_j(θ + Δⱼ) | Variante barata de M2 (n forwards): a média ajudou ou atrapalhou o cliente j em relação ao próprio passo? Capta a diluição diretamente. | Derivada de TAG; procurar precedente em FL ("client drift", "personalization gain") | n forwards por rodada; nenhum dado extra a distribuir. |
| M3 | **Pureza de sinal** por coordenada: P = ½(1 + Σgᵢ / Σ\|gᵢ\|) | Conflito localizado em coordenadas/canais, que o cosseno global dilui. | Chen et al., "Just Pick a Sign" (GradDrop), NeurIPS 2020 | Grátis, se M1 já coleta gradientes. |
| M4 | Similaridade **cosseno por camada com EMA** (alvo de similaridade) | Versão por camada do cosseno, com suavização temporal (reduz ruído). | Wang et al., "Gradient Vaccine" (GradVac), ICLR 2021 | Grátis sobre M1 ou sobre os deltas atuais. |
| M5 | Métricas de conflito usadas em FL multitarefa | Para alinhar com a literatura do artigo. | FedBone (arXiv 2306.17465); FedHCA² (CVPR 2024) | A definir após a leitura. |
| M6 | **CKA** das representações do trunk | Similaridade de representação, não de gradiente. Só faz sentido sobre as mesmas imagens, isto é, dentro de um dataset. | Kornblith et al., ICML 2019 | Forward em um conjunto de imagens comum; não serve entre datasets. |

**Plano de execução:**

1. **E0.2.1 — Verificação das fontes.**
   - Ler a fonte primária (arXiv/proceedings) de cada candidato. Registrar a fórmula exata, a
     definição de conflito/afinidade, a venue/ano, se já foi usado em FL, e as limitações declaradas.
   - Corrigir qualquer item da tabela acima que divergir.
   - Entregável: `docs/GRADIENT_CONFLICT_METHODS.md`, com uma seção por método e a referência em
     BibTeX pronta para o `papers/ciarp2026/refs.bib`. Esse arquivo também alimenta a E5.
2. **E0.2.2 — Protótipo offline.**
   - Estender `scripts/gradient_conflict_controls.py` com os candidatos que sobreviverem à E0.2.1.
     Prioridade: M1, M2, M2' e M3, que são computáveis a partir de estados salvos.
   - Rodar sobre 4 estados já existentes, sem treino novo:
     - (a) 29/09 na inicialização;
     - (b) 29/09 no final;
     - (c) braço `dataset` de 30 rodadas;
     - (d) braço `client` de 30 rodadas.
   - O estado (c) é o mais informativo, porque tem alinhamento intra-dataset real.
3. **E0.2.3 — Controles de validação.** Todo candidato tem de passar nos três:
   - **Positivo:** clientes do mesmo dataset com personalização igual devem ter alta
     similaridade/afinidade. Isso já está demonstrado para o cosseno.
   - **Negativo (novo):** um cliente sintético cujo alvo é invertido (máscara `1 − m` na seg; rótulo
     trocado na cls) treina sobre os mesmos dados. O método **tem** de reportar conflito: cosseno < 0,
     afinidade < 0. É o primeiro teste da sensibilidade a conflito.
   - **Teto:** reportar a concordância intra-cliente (batches diferentes) para normalizar e para
     saber o que é detectável.
4. **E0.2.4 — Medir o custo de cada candidato** em tempo de GPU por rodada e em memória, extrapolado
   para 200 rodadas com amostragem (por exemplo, 1 rodada a cada 10).
5. **E0.2.5 — Matriz de decisão.**
   - Tabela de candidatos × C1–C6, com os números dos controles e do custo.
   - Recomendação provável, a confirmar com os dados: **um geométrico (M1, incluindo M3 de graça)
     e um funcional (M2 ou M2')**. **O usuário escolhe os 2.**

**Treino necessário?** **Não.** A E0.2 roda só sobre estados salvos (GPU local, minutos). O treino
completo fica para a E0.3, depois da escolha.

**Critério de pronto.**
- Fontes verificadas e documentadas.
- Cada candidato prototipado com resultados dos 3 controles e custo medido.
- Matriz de decisão apresentada e escolha registrada.

**Riscos.**
- M2 exige distribuir os deltas a todos os clientes. Na simulação dá para fazer via disco, mas isso
  precisa ser verificado na E0.3.
- O controle negativo com alvo invertido pode ser "fácil demais". Se todos os métodos passarem,
  acrescentar um conflito parcial: inverter apenas 25% dos alvos.

#### E0.2.2–E0.2.5 — Resultado da prototipagem (2026-10-09)

**Script:** `scripts/gradient_conflict_methods.py`.
- Reaproveita os helpers de `gradient_conflict_controls.py`, `train_local`, `evaluate_local` e os
  pesos reais de agregação de `FedPerStrategy`.
- Cria um **adversário** para o primeiro cliente de cada grupo (dataset, tarefas): mesmas imagens e
  mesma personalização, alvos invertidos (`1 − máscara`, `(classe + 1) mod K`).
- Repete tudo sobre os batches de duas rodadas (a/b) para medir a estabilidade.
- Estados analisados:
  - **29/09 final** (seed por cliente, rodada 200);
  - **`dataset` 30r final** (seed por dataset, rodada 30).
- Artefatos em `runs/validation/gradient_conflict_methods/`.

| | Controle negativo (adversário × gêmeo) | Pares reais | Estabilidade a/b (pares reais) | Observação |
|---|---|---|---|---|
| **M1** cos do gradiente | **−0.17** (normalizado −0.35) / **−0.11** (−0.19) | intra 0.013 / **0.25** | 0.16 / **0.92** | Detecta conflito. Fica ruidoso no fim do treino, quando o teto intra-cliente é baixo. |
| **M1** magnitude Φ | 0.31 / 0.56 | ISIC seg×cls: **0.33** em 29/09 | — | Informação nova: revela o domínio de magnitude que o cosseno não vê. |
| **M3** pureza de sinal | −0.02 / −0.06 (excesso sobre a base) | ≈ 0 | — | Sinal fraco e redundante com o M1. **Descartar.** |
| **M2** lookahead (Δ real, val) | **−0.96** / **−0.17**, contra +0.08 / +0.02 do próprio passo | entre datasets, efeitos de ±1e-3 invisíveis ao cosseno | 0.41 / 0.32 por par | É o detector mais forte. Enxerga interação entre datasets. Ruidoso por par: exige média sobre rodadas, como o TAG faz. |
| **M2'** ganho da federação | falha em 2/5 gêmeos em 29/09 (o ganho **sobe** com o adversário) | ISIC: +0.025 | 0.99 (inflado pelos adversários) | Na rodada 200 o próprio passo **piora** a val do ISIC (self Z = −0.026, overfitting). O "ganho" mede só a diluição de passos que fazem overfitting. **Ambíguo; não passa no C3.** |

**Achados laterais (descritivos):**
- M2 no estado `dataset` 30r, em Z ×1e-4, rep a | rep b:
  - TCGA → BUSI: **+14.8 | +10.4** (transferência positiva);
  - TCGA → ISIC: **−9.6 | −8.6** (negativa);
  - SIIM → BUSI: **−5.1 | −4.7** (negativa).
  
  São interações entre datasets que o cosseno (≈ 0) não mostra.
- Em 29/09, a rodada 200, quase toda a matriz Z é negativa, inclusive intra-ISIC e intra-BUSI, e o
  self Z do ISIC é −258e-4. **Isso é overfitting, não interferência:** M2 calculado na validação
  confunde as duas coisas.

**Matriz de decisão:**

| | C1 transf. negativa | C2 indep. otimizador | C3 controles | C4 custo | C5 fonte | C6 complementar | Decisão |
|---|---|---|---|---|---|---|---|
| M1 (cos + Φ) | proxy de 1ª ordem | ✔ | ✔ | ~+10% com 1 rodada a cada 10 | PCGrad ✔ | ✔ magnitude | **recomendado** |
| M2 (TAG, Δ real) | ✔ funcional | parcial | ✔ (o mais forte) | estimado +15–25% com 1 rodada a cada 10 (16×16 forwards × 8 batches) | TAG ✔, adaptado | ✔ | **recomendado** |
| M2' | ✔, mas ambíguo | parcial | ✘ (2/5) | grátis com M2 | adaptação | — | não; reportar Z(agg→j) como subproduto do M2 |
| M3 | proxy local | ✔ | fraco | grátis | GradDrop ✔ | redundante | não |

**Ajustes necessários se M1 + M2 forem escolhidos** (entram no detalhamento da E0.3):
1. **M2 também sobre o batch de treino do alvo**, como o TAG original. O Apêndice B.4 diz que treino
   aproxima validação para i ≠ j. Ter os dois separa interferência (treino) de overfitting (val).
2. **Média sobre rodadas amostradas** (TAG: `Ẑ = (1/T) Σ Zᵗ`), porque a estabilidade por par numa
   única rodada é baixa (0.3–0.4).
3. **M1 sempre reportado com o teto intra-cliente** e o cosseno normalizado.
4. Medir o custo real online antes do treino completo.

#### E0.3 — Implementação online de M1 + M2 (escolha do usuário em 2026-10-09)

**Requisito inviolável: a telemetria não pode alterar o treino.**
- Toda coleta roda dentro de `_preserve_rng_state` (`src/federated/client.py:93`), que já existe para
  validação observacional.
- O modelo é restaurado ao estado anterior antes do treino local.
- Os gradientes/updates de treino ficam bit a bit iguais com a coleta ligada ou desligada (teste em
  CPU, E0.3.5).

**E0.3.1 — Configuração (fora do hash científico).**
- Bloco novo:
  ```yaml
  runtime:
    diagnostics:
      gradient_conflict:
        enabled: false           # padrão: runs existentes inalterados
        every_n_rounds: 10       # rodadas amostradas: 1, 11, 21, …, sempre incluindo a última
        probe_batches: 10        # M1: batches do sampler da própria rodada
        sketch_dim: 65536        # M1: CountSketch com seed fixa (ver E0.3.2)
        lookahead:
          enabled: true
          val_batches: 8
          train_batches: 8
  ```
- `runtime` já é removido da assinatura de resume (`training_federated._resume_config_signature`) e do
  hash de estudo (`study_runner`).
- Validação em `validate_runtime_config` (`src/utils/training_runtime.py`).
- Só vale para `standalone: false`. No local-only não há trunk comum.

**E0.3.2 — M1 no cliente** (`fit`, antes de `train_local`, só nas rodadas amostradas).
- Gradiente bruto do trunk no θ recebido, sobre os `probe_batches` da rodada: reutilizar o
  procedimento do `grad10` de `scripts/gradient_conflict_controls.py`, movido para
  `src/federated/conflict_diagnostics.py`.
- As duas metades dos batches dão gᴬ e gᴮ: g = média, e o **teto intra-cliente** = cos(gᴬ, gᴮ), sem
  custo extra.
- Persistência:
  - O vetor exato tem 18,6 MB por cliente e rodada (~6 GB num run). Em vez disso, gravar um
    **CountSketch** de dimensão `sketch_dim`: hash e sinal por coordenada, com seed fixa e comum a todos
    os clientes, e soma via `index_add_`.
  - O sketch preserva produtos internos em esperança.
  - Arquivo: `fold_k/conflict/round_r/<client>_m1.npz` (gᴬ, gᴮ e as normas **exatas** de g, gᴬ, gᴮ,
    para Φ e para normalizar o cosseno).
  - Erro do sketch: validado contra o cosseno exato no teste E0.3.5, com tolerância a definir a
    partir de 1/√k ≈ 0.004.
- Também por bloco (encoder2…bottleneck), com um sketch por bloco, de dimensão proporcional ao bloco.

**E0.3.3 — M2 no fluxo fit → aggregate → evaluate.**
1. `fit` (rodada amostrada): o cliente grava um snapshot da sua personalização **pré-fit** em
   `fold_k/conflict/round_r/<client>_pre.pt`. O TAG mantém fixos os parâmetros específicos da tarefa.
2. `aggregate_fit` (servidor): grava θ_r (o trunk enviado) e os 16 Δᵢ (os updates reais da rodada) em
   `fold_k/conflict/round_r/deltas.npz` (~300 MB, temporário). O caminho vai para os clientes via
   `configure_evaluate` → `config["conflict_round_dir"]`.
3. `evaluate` (cliente j, rodada amostrada): carrega o snapshot pré-fit e θ_r, e calcula Lⱼ nos
   `val_batches` (val) e nos `train_batches` (batch do sampler da rodada, sem augmentation) para:
   - θ_r (base);
   - θ_r + Δᵢ, para cada cliente i (inclusive j, que dá o "self");
   - θ_agg (subproduto M2').

   Grava `fold_k/conflict/round_r/<client>_m2.csv` com colunas
   `source, target, split ∈ {train, val}, loss, z`.
4. Ao fim da rodada, apagar `deltas.npz` e os snapshots `_pre.pt`. Ficam só os CSVs e os sketches.

**E0.3.4 — Consolidação e relatório** (`src/federated/conflict_diagnostics.py`, chamado no fim do run,
no mesmo ponto e com a mesma política não-fatal do `negative_transfer`).
- `conflict_m1_pairs.csv`: rodada, i, j, bloco, cos, cos normalizado `cos / √(tetoᵢ·tetoⱼ)`, Φ,
  tetos.
- `conflict_m2_lookahead.csv`: rodada, origem, alvo, split, Z.
- `conflict_summary.csv`, com média ± dp **sobre as rodadas amostradas** (`Ẑ = (1/T) Σ Zᵗ`, como no
  TAG), por par de **datasets** e por classe de par, nos splits train e val, separadamente.
- Linha no `execution.log`.
- O HTML fica para o app da E2, que lê os CSVs.
- Leitura: Z(train) < 0 e Z(val) < 0 → interferência; Z(train) ≥ 0 e Z(val) < 0 → overfitting, não
  interferência.

**E0.3.5 — Testes** (`tests/test_conflict_diagnostics.py`).
- CountSketch: o cosseno e o produto interno estimados ficam dentro da tolerância do exato em vetores
  aleatórios correlacionados (cos ∈ {−0.5, 0, 0.5, 0.99}).
- Fórmulas: Φ (iguais → 1), Z (passo nulo → 0; passo que dobra a loss → −1), normalização pelo teto.
- **Não-perturbação:** `fit` de um cliente minúsculo em CPU com a coleta ligada e desligada →
  parâmetros finais e histórico idênticos.
- Rodada não amostrada: nenhum arquivo é escrito.
- Leitura tolerante de runs sem a pasta `conflict/` → `n/a`.
- Smoke: `python -m scripts.smoke_federated --setup federated --holdout --samples 2` com a
  coleta ligada (eu executo).

**E0.3.6 — Custo real.** Rodar 3 rodadas do run de 29/09 na GPU com `every_n_rounds: 1` e medir o
acréscimo por rodada (`runtime_events.csv`). Se o custo extrapolado para 200 rodadas passar de 25%,
reduzir `val_batches`/`train_batches` ou aumentar `every_n_rounds`.

**E0.3 — Resultado da implementação (2026-10-09).**
- Código:
  - módulo novo `src/federated/conflict_diagnostics.py`;
  - ganchos em `client.fit/evaluate`, `FedPerStrategy.aggregate_fit/aggregate_evaluate` e
    `training_federated`;
  - padrões e validação em `src/utils/training_runtime.py`;
  - `local_objective` extraído de `train_local` sem mudança numérica.
- Toda falha da coleta é registrada no log (`observational`) e nunca interrompe o treino. Com menos de
  2 batches-sonda, o M1 é pulado com aviso.
- Testes: `tests/test_conflict_diagnostics.py`, 14 testes. Suíte completa: 158 OK.
- Smoke: `python -m scripts.smoke_federated --setup federated --holdout --samples 4 --conflict-diagnostics`
  → `SMOKE_OK`, M1 e M2 nas 2 rodadas, transitórios removidos.
- **Não-perturbação em escala real** (GPU, configuração de 29/09, 3 rodadas, coleta em todas as
  rodadas vs desligada): as métricas de teste são idênticas em todos os 8 pares dataset/tarefa. A única
  diferença é o SIIM seg, 0.0192 vs 0.0191, na 4ª casa, que é não-determinismo do cuDNN/BF16.
- **Custo medido** (`runs/validation/conflict_cost/`):
  - por rodada amostrada, o M1 acrescenta ~2 s ao `fit` (27 → 29 s) e o M2 ~64 s ao `evaluate`
    (9 → 73 s, com 8+8 batches);
  - com 1 rodada a cada 10, isso dá ~+27% num run de 200 rodadas, acima do limite de 25%;
  - **ajuste:** `val_batches` e `train_batches` de 8 para 4. O custo do M2 escala linearmente com os
    batches, então a estimativa é de ~+14%;
  - a variabilidade de Z vem sobretudo de Δ, e não dos batches de avaliação (E0.2), por isso a
    redução custa pouca precisão.

**E0.3.7 — Handoff de treino (usuário).** As configurações estão prontas em
`runs/validation/gradient_conflict_full/{client,dataset}.yaml`. Diferença em relação ao run de 29/09:
apenas `federated.personalized_init_seed` e o bloco `runtime.diagnostics.gradient_conflict` (ligado,
a cada 10 rodadas, 10 batches-sonda, sketch de 65536, lookahead com 4+4 batches). Proposta original:
- (a) o run de 29/09 com a coleta ligada e `personalized_init_seed: client`;
- (b) o mesmo com `personalized_init_seed: dataset`.

Ambos com 200 rodadas. O (a) também serve de verificação de não-perturbação em escala real: as
métricas finais devem bater com as de 29/09, a menos do não-determinismo do cuDNN.

**Treino necessário?** Sim. As métricas novas são coletadas durante o treino e não podem ser
recalculadas a partir dos artefatos existentes, porque nenhum artefato guarda o trunk pós-fit por
rodada. O handoff virá na E0.3: provavelmente uma re-execução da configuração de 29/09 com a
telemetria nova ligada, que é só diagnóstica e não muda o resultado.

**Critério de pronto.**
- A métrica atual foi validada ou refutada com controles.
- Os 2 métodos novos estão implementados, testados e rodados num treino completo.
- Os conflitos medidos foram interpretados.
- D9 da E1 usa os métodos validados.

**Registro.**
- 2026-10-08 — etapa criada a pedido do usuário, após a constatação de cosseno ≈ 0 em todos os pares.
- 2026-10-08 — E0.1 concluída: a métrica é válida, e a ortogonalidade vem dos stems/cabeças
  inicializados por cliente (H3). Próximo passo: decisão do usuário sobre o desenho e, depois, E0.2.
- 2026-10-09 — E0.3 implementada e testada.
  - Não-perturbação confirmada na GPU.
  - Custo ajustado para ~+14%.
  - **Aguardando os 2 treinos completos do usuário (E0.3.7).**
- 2026-10-09 — E0.2 concluída.
  - Protótipo e controle adversário executados.
  - **O usuário escolheu M1 (cosseno + magnitude do gradiente bruto) e M2 (lookahead TAG com o
    update real, em train e val, com média sobre rodadas).**
  - E0.3 detalhada.
- 2026-10-09 — E0.2.1 concluída: fontes verificadas em `docs/GRADIENT_CONFLICT_METHODS.md`.
  - Correções: título do TAG ("for"); venues de GradVac, FedFomo e FedHCA² não confirmadas no texto
    primário; FedBone é preprint.
  - M2' ganhou precedente (FedFomo, Eq. 3).
  - O FedHCA² usa deltas de rodada, como a nossa métrica atual.
  - TAG com η pequeno reduz-se a um produto interno, por isso usar o delta real.
- 2026-10-09 — E0.1b concluída: com `personalized_init_seed: dataset`, o cosseno intra-dataset
  começa em 0.39, chega a 0.57 e cai para 0.25 na rodada 30. Os stems permanecem idênticos
  (cos ≥ 0.999), e o cosseno inter-dataset continua ≈ 0.

---

## E1 — Validação do treinamento

**Objetivo.** Verificar se há problemas durante o treinamento (numéricos, de convergência, de
aprendizado por cliente ou de agregação). Também deixar guardas que detectem esses problemas cedo nos
próximos treinos.

**Situação atual.**
- Checagens de config/contrato em `src/federated/local_trainer.py`, `client.py`, `server.py` e
  `src/training_federated.py`.
- **Já existe guarda de loss não-finita:** `PrecisionPolicy.ensure_finite`
  (`src/utils/training_runtime.py:65`) levanta `FloatingPointError` no treino e na validação locais.
- NaN-mean na agregação. `server.py` já marca NaN quando o delta de um cliente é zero.
- Artefatos disponíveis para auditoria:
  - `fold_*/client_*/training_history.csv`: formato longo, um registro por step.
  - `fold_*/aggregation_history.json`: cosseno entre deltas e `negative_transfer`.
  - `gpu_telemetry.csv`, `runtime_events.csv`, `training_curves/history.csv` e `execution.log`.
- **Lacunas:**
  - não há norma de gradiente registrada;
  - em BF16 não há `GradScaler`, então um gradiente não-finito com loss finita chega ao
    `optimizer.step()` sem ser detectado;
  - não há detecção de divergência, estagnação ou overfitting;
  - não há diagnóstico automático dos artefatos ao fim do run.

**Entregáveis (macro).**
1. **Diagnóstico offline** dos runs existentes: `runs/20260929_*` e os estudos `multi_dataset_*` e
   `example_multi_dataset_dice_bce`. Um script lê os artefatos e sinaliza:
   - NaN/Inf;
   - loss explodindo ou estagnada;
   - cliente que não aprende (cls no piso do preditor constante, Dice colapsado);
   - gap treino/val anômalo;
   - conflito de gradiente acima da referência ortogonal.
2. **Guardas online** no treino: NaN/Inf na loss, norma de gradiente registrada no histórico e eventos
   de alerta em `runtime_events.csv`. A política de abort é configurável.
3. Testes unitários para o diagnóstico e as guardas.
4. Relatório dos achados, com eventuais correções em sub-itens desta etapa.

**Treino necessário?** Um smoke local (`python -m scripts.smoke_federated --setup both`). Se a
auditoria encontrar problemas, há uma re-execução pelo usuário, seguindo o protocolo de handoff.

**Critério de pronto.** Diagnóstico executado sobre os runs existentes, com os achados documentados.
Guardas cobertas por testes. `python -m unittest discover -v` passando.

**Riscos/dependências.** As guardas não podem alterar o resultado numérico de runs saudáveis, pois
isso afetaria a reprodutibilidade. O custo da norma de gradiente por step precisa ser medido.

**Detalhamento.**

#### E1.0 — Auditoria preliminar (feita em 2026-10-08)

Script exploratório, fora do repo, sobre `runs/20260929_082717_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-TCGA_LGG-SIIM_ACR`:
- 4 datasets e 16 clientes;
- CV=1 (holdout), seed 1993, BF16;
- modo `steps` (10 steps/rodada × 200 rodadas);
- `task_weights` seg 4 : cls 1.

Os resultados são descritivos: um holdout e uma seed.

| # | Achado | Evidência | Severidade |
|---|---|---|---|
| A1 | **SIIM-ACR cls não aprende** | Loss de treino cls ≈ 0.69 ≈ ln 2 (acaso binário) do início ao fim, em todos os 4 clientes (variação de −0.7% a −1.4%). Balanced acc. de val 0.52–0.55. | crítica |
| A2 | **SIIM-ACR seg colapsada em 2 de 4 clientes** | `mt_1` e `mt_3` terminam com val dice_positive ≈ 0.000, com pico de 0.03 nas primeiras rodadas. `mt_0` e `mt_2` ≈ 0.16. Teste: dice_positive 0.064 ± 0.075. | crítica |
| A3 | **Overfitting de cls com avaliação na última rodada** | A val loss cls sobe enquanto a de treino cai: `Curated_BUSI_mt_1` (melhor na rodada 87; 0.90 → 1.23) e `TCGA_LGG_mt_0` (melhor na 92; 0.45 → 0.72). Por desenho do protocolo, a avaliação final usa a última rodada. | alta |
| A4 | **ISIC-2018 cls não generaliza** | Treino cai 31–36%. A val loss de `cls_0` fica estável (1.90 → 1.93). Balanced acc. máx. ≈ 0.41–0.46. A acc de teste, 0.274, está abaixo do preditor constante "NV" (0.669), o que é coerente com o `balanced_fold`. | alta |
| A5 | Nenhum valor não-finito | 0 NaN/Inf nos históricos dos 16 clientes. | ok |
| A6 | ~~Sem conflito de gradiente~~ **Sem interação entre clientes** | Cancelamento `overall` = 0.667 ≈ referência ortogonal 0.668. A E0.1 mostrou que a métrica é válida e que os updates são ortogonais por causa dos stems/cabeças inicializados por cliente. | alta (desenho) |
| A7 | ISIC-2018 seg saudável | val dice 0.84–0.87, ainda subindo no fim. | ok |
| A8 | Duplicação no histórico | `post_local_round/train` repete exatamente os valores de `local_step/train` (agregado). Não é erro, mas o diagnóstico deve contar cada série uma única vez. | info |

Leitura: A1 e A2 sugerem que o SIIM-ACR, com 128 px e `λ_cls = 0.2`, está subtreinado ou desbalanceado
entre tarefas. Pneumotórax é uma lesão pequena, e o downsample pode apagá-la. A3 e A4 apontam para a
interação entre o budget e a escolha da rodada final. **Esses achados são entradas para E3 (loss e
ponderação) e E4 (pré-processamento/resolução); a E1 não os corrige.** O objetivo da E1 é que eles
passem a ser detectados automaticamente em todo run.

#### E1.1 — Diagnóstico offline: `src/experiments/training_diagnostics.py` (novo)

- **Entrada:** um diretório de run (`runs/<ts>_*`) ou de estudo (`runs/studies/<id>/`). Num estudo,
  itera sobre os runs listados em `run_index.csv`.
- **Leitura:**
  - `fold_*/client_*/training_history.csv` e `metadata.yaml`;
  - `fold_*/aggregation_history.json`;
  - `negative_transfer_summary.csv`;
  - `{setup}_test_results.csv`;
  - `config.yaml`, de onde vêm número de classes, critério e modo de budget.
- Cada série é lida uma única vez: treino via `local_step`, validação via `post_aggregation_round`
  (federado) ou a fase equivalente no standalone.
- O leitor é tolerante a campos ausentes. Runs antigos sem `negative_transfer` ou sem histórico por
  step recebem status `n/a`, e não erro.
- **Checagens.** Cada uma gera um *finding*: `check_id, severity, fold, dataset, client_id, task,
  metric, evidence, threshold`.

| id | Checagem | Regra (limiar padrão) |
|---|---|---|
| D1 | valores não-finitos | qualquer NaN/Inf em `value` → crítica |
| D2 | loss no nível do acaso | último terço da loss de treino cls ≥ 0.95·ln K (CE; K = nº de classes) → crítica. Só vale para CE, pois a Focal muda a referência. |
| D3 | estagnação | variação relativa entre o primeiro e o último terço da loss de treino > −5% → alta |
| D4 | divergência | último terço da loss de treino > primeiro terço, ou picos de step > mediana + 10·MAD em > 1% dos steps → crítica |
| D5 | overfitting | melhor rodada de val loss < 60% das rodadas, val loss final > 1.15 × mínima e treino ainda caindo → alta. Reporta a melhor rodada. |
| D6 | seg colapsada | val dice_positive do último terço < 0.05 → crítica |
| D7 | cls no piso | val balanced_accuracy do último terço ≤ 1/K + 0.05 → crítica |
| D8 | heterogeneidade intra-dataset | (máx − mín) da métrica de teste entre clientes do mesmo dataset/tarefa > 0.15 com algum cliente em D6/D7 → alta |
| D9 | conflito de gradiente | `overall − orthogonal_reference` > 0.05 (ou intra por dataset) → alta |
| D10 | exposição por budget | épocas efetivas por cliente (steps × batch / `effective_train_examples`); razão máx/mín > 5 → info |
| D11 | gradiente (quando houver E1.2) | norma de gradiente não-finita → crítica; mediana do último terço > 10 × a do primeiro → alta |

- Os limiares ficam num dicionário de constantes do módulo e podem ser sobrescritos por flags da CLI.
  **Nada entra em `config.yaml`**, para não mexer no hash científico dos estudos.
- **Saída** em `<run>/diagnostics/`:
  - `findings.csv`;
  - `summary.json`, com contagem por severidade e status `ok`/`warning`/`critical`;
  - `summary.md`, legível.
  - Num estudo, também `runs/studies/<id>/diagnostics/` com o consolidado por braço.
  - O app da E2 consome `findings.csv`/`summary.json`.
- **CLI:** `python -m src.experiments.training_diagnostics <run_dir|study_dir> [--out DIR] [--threshold D6=0.05 ...]`.
- **Integração:** chamada não-fatal no fim de `src/training_federated.py`, no mesmo ponto onde já roda
  o `negative_transfer`, com um `try/except` que só loga a falha. O resultado do run não muda.

#### E1.2 — Guardas online de gradiente: `src/federated/local_trainer.py`

- Depois de `loss.backward()` e antes de `optimizer.step()`, calcular a norma L2 total dos gradientes
  com `torch.nn.utils.get_total_norm`. A função é apenas leitura e não altera gradientes, portanto não
  muda os números de runs saudáveis.
  - Se a norma for não-finita → `FloatingPointError` com contexto (cliente, rodada, step), mantendo a
    política atual de `ensure_finite`.
  - Registrar `grad_norm` como `metric_name` adicional: por step quando `record_step_history`, e
    média e máximo por época/rodada. A análise lida com o formato longo, e `training_curves.py`
    filtra por nome de métrica, então a linha nova não quebra os dashboards.
  - Expor `grad_norm_mean` e `grad_norm_max` no dicionário retornado por `train_local`.
- Medir o custo no `scripts.benchmark_training_runtime`. Se passar de 2% do tempo de step, tornar a
  medição desligável via `runtime.diagnostics.grad_norm` (o bloco `runtime` já é excluído do hash em
  `study_runner.py`).
- **Servidor** (`src/federated/server.py`, em `aggregate_fit`): verificar a finitude dos parâmetros do
  trunk recebidos de cada cliente antes de agregar. Se um cliente devolver valor não-finito, falhar
  com mensagem clara, em vez de propagar NaN para o trunk global.

#### E1.3 — Testes: `tests/test_training_diagnostics.py` (novo) e extensão de testes existentes

- Fixtures sintéticas de `training_history.csv`, uma por checagem D1–D10: um caso que dispara e um
  saudável que não dispara.
- Leitura de um run antigo sem `negative_transfer` → `n/a`, sem exceção.
- `train_local` com um gradiente forçado a Inf → `FloatingPointError`.
- `train_local` num modelo minúsculo em CPU com seed fixa: as losses e os pesos finais são
  **idênticos** com e sem a medição de norma (garantia de não-regressão).
- `aggregate_fit` com um cliente devolvendo NaN → erro explícito.

#### E1.4 — Execução e relatório

1. Rodar o diagnóstico sobre o run de 29/09 e sobre os 10 braços de `runs/studies/multi_dataset_balance_v1`.
   Os outros estudos têm só o plano, sem runs.
2. Confirmar que D2, D6, D5 e D7 reproduzem A1–A4.
3. Registrar os achados consolidados abaixo, em "Registro".
4. Rodar `python -m scripts.smoke_federated --setup both --holdout --samples 2` (CPU, curto; eu
   executo) e `python -m unittest discover -v`.

**Treino pelo usuário:** **não é necessário na E1.** As guardas são somente leitura, então não há
motivo para refazer runs. Os achados A1–A4 orientam os braços da E3/E4, e lá haverá o handoff de treino.

**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-08 — branch `feat/training-validation` criada; auditoria preliminar (E1.0) feita;
  detalhamento E1.1–E1.4 escrito.

---

## E2 — App local de monitoramento e comparação

**Objetivo.** Aplicação local para acompanhar os treinamentos e comparar treinamentos selecionados.

**Situação atual.**
- `src/experiments/training_curves.py` gera `history.csv` e `dashboard.html` (estático, um por run).
- Relatórios HTML estáticos:
  - por run: `src/experiments/run_report.py`;
  - estudos: `src/experiments/analyze.py`;
  - `src/federated/negative_transfer.py`.
- Estilo de figuras em `src/utils/plot_style.py`. Estudos indexados por `run_index.csv`.
- Não há nenhuma dependência web no `requirements.txt`. Restrições: `numpy<2`, `protobuf>=4` (flwr).

**Requisitos (macro).**
- Listar runs avulsos (`runs/`) e estudos (`runs/studies/<id>/`) com seus braços.
- Selecionar um treino para visualizar e vários para comparar: curvas de loss/métricas por rodada,
  por dataset e por cliente, além de métricas finais lado a lado.
- Acompanhar um run em andamento com refresh periódico.
- Mostrar os alertas do diagnóstico da E1.
- Rodar localmente com um único comando.

**Decisão em aberto (no detalhamento).** Stack: servidor stdlib + HTML/JS estático (sem dependência
nova) **ou** Streamlit/afim (nova dependência, a validar contra `numpy<2`).

**Treino necessário?** Não. Usa os artefatos existentes e, para validar o modo "ao vivo", um smoke.

**Critério de pronto.** O app abre, lista os runs existentes, compara pelo menos dois runs/braços e
atualiza um run em andamento. Os testes da camada de leitura de artefatos passam.

**Riscos/dependências.** Depende da E1 (alertas). Os formatos de artefato variam entre versões dos
runs, então o leitor precisa ser tolerante a campos ausentes (ex.: `negative_transfer` em históricos
antigos).

**Detalhamento.** _(a preencher)_

**Registro.**
- 2026-10-04 — etapa planejada (macro).

---

## E5 — Validação das fontes de cada decisão de implementação

**Objetivo.** Garantir que cada decisão de implementação tenha fonte primária verificada, ou seja
explicitamente marcada como decisão própria.

**Situação atual.** Fontes espalhadas em `papers/ciarp2026/refs.bib`, `docs/RESEARCH_NOTES.md`,
`referencies/`, `docs/DICE_BCE_SUPERVISION.md` e `data/*/PAPER_NOTES.md`.

**Entregáveis (macro).**
1. **Registro de decisões**, um documento em `docs/` com uma tabela: decisão → onde está no código →
   fonte primária → status (verificado / divergente / sem fonte / decisão própria) → observação.
2. Decisões que a tabela deve cobrir:
   - FedPer e FedAvg;
   - agregação hierárquica e `dataset_weights`;
   - partição Dirichlet;
   - DiceBCE e Focal loss;
   - deep supervision `1/(n+1)`;
   - mascaramento de supervisão parcial (ConDistFL/UFPS);
   - topologia HC-FMTL (FedHCA²) e conflito de gradiente (FedBone);
   - métrica de cancelamento (negative transfer);
   - protocolo estatístico (Wilcoxon exploratório);
   - pré-processamento por dataset.
3. Lista de divergências entre código e fonte, com proposta de correção ou justificativa.
4. Atualização de `refs.bib`, se faltar referência (ex.: MOCHA, Smith et al. 2017).

**Caráter transversal.** E3 e E4 registram as fontes de suas decisões à medida que avançam. A E5
consolida e verifica as decisões já existentes antes das ablações.

**Treino necessário?** Não.

**Critério de pronto.** Toda decisão listada com status, e nenhuma divergência sem encaminhamento.

**Detalhamento.** _(a preencher)_

**Registro.**
- 2026-10-04 — etapa planejada (macro).

---

## E3 — Ablação: loss fed multitask multi-dataset

**Objetivo.** Definir a loss do treinamento federado multitarefa multi-dataset por ablação controlada.

**Situação atual.**
- **Segmentação:** `DiceBCE` (padrão, 0.5/0.5), além de DICE, FocalDICE, GeneralizedDICE, Jaccard, BCE
  e outras (`src/utils/experiment_init.py`, `src/utils/segmentation_loss.py`).
- **Classificação:** CE ou Focal (`src/utils/criterions.py`), com class weighting por dataset.
- **Combinação entre tarefas:** lambdas fixos (`resolve_task_lambdas`, `_combined_loss` em
  `src/utils/supervision.py`).
- **Agregação:** `aggregation.task_weights` e `dataset_weights` (`src/federated/server.py`).
- **Não existe:** ponderação adaptativa (Kendall uncertainty weighting, GradNorm, PCGrad/CAGrad).

**Entregáveis (macro).**
1. Matriz de braços, separando o nível **local** (loss do cliente) do nível de **agregação**:
   - loss de seg;
   - loss de cls;
   - ponderação entre tarefas.
2. Implementação apenas do que faltar (ex.: um método adaptativo), com testes.
3. Novo manifest em `studies/`. Todos os braços sobre a mesma partição e o mesmo modo de budget.
4. Handoff de treino para o usuário.
5. Análise: app (E2), `analyze.py` e diagnóstico (E1). Decisão da loss registrada com as fontes (E5).

**Treino necessário?** Sim, o estudo completo, executado pelo usuário.

**Critério de pronto.** Loss escolhida, com evidência descritiva e fonte registrada.

**Riscos/dependências.** O número de braços cresce rápido, então é preciso priorizar. Métodos
adaptativos interagem com a agregação (FedBone), o que exige cuidado na interpretação.

**Detalhamento.** _(a preencher)_

**Registro.**
- 2026-10-04 — etapa planejada (macro).

---

## E4 — Ablação: pré-processamento de imagens

**Objetivo.** Medir o efeito de escolhas de pré-processamento no desempenho.

**Situação atual.**
- **Interpolação:** BUSI usa `INTER_NEAREST` (congelado); ISIC, TCGA-LGG e SIIM-ACR usam `INTER_AREA`.
  Máscaras sempre em nearest.
- **Augmentation:** flip H/V e rotação em `src/dataset/BUSI_dataset.py`.
- **Legado:** CLAHE, SOBEL, brilho e contraste como canais extras.
- **Normalização:** hook existe, mas está desligado (`federated_dataloader.py`, `normalization=None`).

**Restrição central.** Ablações que alteram a imagem gravada exigem uma **nova variante** de dataset,
sem sobrescrever `processed_128` nem regerar a partição congelada. Por isso a preferência vai para
variações *on-the-fly* no dataloader (normalização, CLAHE, augmentation), que preservam a partição.
Antes de criar uma variante gravada, é preciso verificar se a partição pode ser reutilizada.

**Entregáveis (macro).**
1. Matriz de braços: on-the-fly e, se justificado, variante gravada.
2. Implementação configurável, com testes.
3. Manifest, handoff de treino, análise e fontes registradas.

**Treino necessário?** Sim, o estudo completo, executado pelo usuário.

**Critério de pronto.** Pipeline de pré-processamento escolhido, com evidência descritiva e fonte
registrada.

**Detalhamento.** _(a preencher)_

**Registro.**
- 2026-10-04 — etapa planejada (macro).
