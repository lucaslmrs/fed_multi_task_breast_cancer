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
| E0 | Métodos adicionais de verificação de conflito de gradientes | ✅ | `feat/gradient-conflict` | 2026-10-15 |
| B0 | Baselines não federados por dataset (seg / cls / multitask) | ✅ | `feat/baselines` | 2026-10-16 |
| E1 | Validação do treinamento | ✅ | `feat/training-validation-v2` | 2026-10-16 |
| E2 | App local de monitoramento e comparação | ✅ | `feat/monitor-app` | 2026-10-16 |
| E5 | Validação das fontes (consolidação) | ✅ | `docs/sources` | 2026-10-16 |
| E3 | Ablação: loss fed multitask multi-dataset | ✅ | `exp/loss-ablation` | 2026-10-18 |
| E4 | Ablação: pré-processamento de imagens | ⏳ | `exp/preprocessing-ablation` | 2026-10-18 |
| E6 | Compartilhamento ampliado (decoder federado) | ⏳ | `exp/wider-sharing` | 2026-10-18 |
| E7 | Transfer learning (ResNet-34 ImageNet) | ⏳ | `exp/transfer-learning` | 2026-10-18 |

Ordem de execução: **E0 → E1 → E2 → E5 → E3 → (E4 ∥ E6) → E7**.
- E0 foi incluída em 2026-10-08 a pedido do usuário. A métrica atual de conflito dá ≈ 0 para todos
  os pares de clientes, inclusive pares que deveriam estar alinhados, e é preciso saber se isso é
  ausência real de conflito ou cegueira da métrica antes de usá-la na E1 (D9) e na E3.
- E6 e E7 foram incluídas em 2026-10-18 a pedido do usuário. A E7 depende do mecanismo de
  compartilhamento da E6 (`shared_modules`), por isso vem depois dela.
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

**E0.4 — Resultado dos treinos completos (executados pelo usuário em 2026-10-14, em paralelo).**

Runs:
- `runs/20261014_211153_…` (`personalized_init_seed: client`);
- `runs/20261014_211215_…` (`dataset`).

Ambos com 200 rodadas, 21 rodadas amostradas e 0 falhas da coleta. Scripts de análise em
`runs/validation/gradient_conflict_full/`. Um run por braço, uma seed, um holdout: **descritivo**.

1. **Piso de ruído entre execuções.** O run `client` difere do de 29/09 em até ~0.02 nas métricas
   finais (BUSI cls 0.613 vs 0.590; ISIC cls 0.255 vs 0.274).
   - A divergência começa **no 1º step da rodada 1**, antes de qualquer atualização (ISIC seg: loss
     1.38250065 vs 1.38249981), e cresce suavemente, sem saltos nas rodadas amostradas.
   - Isso é não-determinismo do BF16/cuDNN (`cuda_benchmark: true`), e não perturbação pela coleta.
   - **Consequência: diferenças de ~±0.02 entre dois runs quaisquer estão dentro do ruído.**
2. **M1 (cosseno do gradiente bruto), média sobre as rodadas amostradas:**

   | classe de par | `client` | `dataset` |
   |---|---|---|
   | intra-dataset, mesmas tarefas | 0.003 (normalizado 0.002) | **0.157 (normalizado 0.55)** |
   | intra-dataset, tarefas diferentes (ISIC seg×cls) | 0.003 | 0.008 |
   | inter-dataset | −0.001 | −0.001 |

   - Com seed por dataset, o alinhamento intra-dataset vai de 0.54 (rodada 1) a ~0.05–0.15 na segunda
     metade do treino e ≈ 0 na rodada 200. O teto intra-cliente fica entre 0.3 e 0.5.
   - **Entre datasets e entre seg×cls do ISIC, não há conflito de 1ª ordem em nenhuma das duas
     políticas.**
   - Similaridade de magnitude inter ≈ 0.41: as normas dos gradientes diferem ~4× entre datasets.
3. **M2 (lookahead com o update real):**
   - O Z entre clientes é **negativo** no split de treino em 70–100% das rodadas, inclusive intra-dataset
     na política `dataset`, onde o M1 é positivo.
   - Como o M1 inter ≈ 0, o termo de 1ª ordem é nulo. O Z < 0 vem do **termo de 2ª ordem
     (curvatura)** de aplicar o passo **inteiro** de outro cliente: L(θ+Δ) ≈ L + g·Δ + ½ΔᵀHΔ. O
     "tragic triad" do PCGrad inclui justamente a curvatura.
   - **Sem uma referência nula (passo aleatório com a mesma norma), Z < 0 não pode ser lido como
     transferência negativa.**
   - O **update agregado** que o servidor aplica de fato tem **Z > 0 no treino para os 4 datasets nos
     dois runs**: BUSI +14/+17, TCGA +12/+16, ISIC +4/+8, SIIM +0.5/+0.8 (×1e-4). A média encolhe
     cada passo, e a curvatura cai com o quadrado do peso.
   - Na validação, o próprio passo **piora** BUSI, ISIC e TCGA (overfitting no fim do treino), enquanto
     o agregado continua ≥ 0, exceto TCGA na política `dataset`.
4. **Métricas de teste, `dataset` vs `client`:**
   - acima do piso de ruído: ISIC cls 0.255 → 0.386, TCGA cls 0.885 → 0.923, TCGA seg 0.604 → 0.641;
   - **SIIM seg colapsa: 0.066 → 0.000**;
   - BUSI dentro do ruído.

**Conclusão da E0 (o que os dados sustentam):**
- Os dois métodos concordam que **não há conflito de 1ª ordem entre datasets**. As atualizações entre
  datasets não interagem nem de forma construtiva nem destrutiva.
- O dano aparente nos pares (M2 < 0) é efeito de curvatura de passos inteiros e não acontece no update
  realmente aplicado, que ajuda todos os datasets no treino.
- **Isso não prova ausência de transferência negativa no desempenho.** Para isso falta o par
  federado × local-only com a configuração atual.

**Pendências abertas pela E0.4** (propostas; decisão do usuário):
- (a) Referência nula no M2: avaliar Lⱼ(θ + r) com r aleatório de norma igual à mediana de ‖Δ‖, que
  custa +2 avaliações por cliente. Sem isso, o Z pareado não é interpretável.
- (b) Runs local-only pareados com os dois runs acima, para a pergunta de desempenho.
- (c) Investigar o colapso da seg do SIIM com seed por dataset; liga-se a A1/A2 da E1.
- (d) Repetir com `cuda_benchmark: false` ou com mais seeds para medir o piso de ruído de forma
  controlada.
- (e) Memória: os atores Ray chegaram a 2.3–3.3 GB de RSS + ~0.8 GB de swap cada. Registrar a memória
  por processo na E1.

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

**E0.3.8 — Resultado dos treinos completos (2026-10-14).**

Runs:
- `runs/20261014_211153_*`: `personalized_init_seed: client`;
- `runs/20261014_211215_*`: `personalized_init_seed: dataset`.

Os dois rodaram em paralelo, levaram ~2h14 cada, não registraram nenhum aviso da coleta e geraram os
3 CSVs de conflito. **Uma seed, um holdout: tudo é descritivo.**

1. **Ruído entre runs (GPU).**
   - O run `client` não reproduz 29/09 bit a bit. As trajetórias divergem **desde o 1º step**
     (|Δloss| de 1e-5 a 3e-4), e dois runs **sem** coleta (29/09 vs `conflict_cost/off`) divergem
     com a mesma magnitude. A causa é `cuda_benchmark` + BF16, e não a telemetria; a
     não-perturbação está provada bit a bit em CPU.
   - Diferença de teste 29/09 → `client` depois de 200 rodadas:
     - BUSI: balanced acc 0.551 → 0.603, Dice positivo 0.631 → 0.650;
     - ISIC: balanced acc 0.352 → 0.351;
     - TCGA: balanced acc 0.854 → 0.858, Dice 0.594 → 0.604.
   - **Ruído de referência por run: até ~0.05 no BUSI (dataset pequeno) e ~0.01 nos demais.**
     Nenhuma comparação de run único deve tratar diferenças dessa ordem como efeito.

2. **M1 (cosseno do gradiente bruto).**
   - `client`: intra-dataset ≈ 0 nas 21 rodadas amostradas (−0.009 a +0.014). Confirma a
     ortogonalidade estrutural ao longo de todo o treino.
   - `dataset`: intra-dataset em 0.54 na rodada 1, depois 0.43 (r11), 0.25 (r31), ~0.05–0.17 a
     partir de r91. Por dataset (média do run): TCGA 0.28, SIIM 0.19, ISIC 0.14, BUSI 0.09. No
     final, concentrado em encoder2/3 (0.08–0.10) contra bottleneck 0.01.
   - Entre datasets: ≈ 0 nas duas políticas, em todas as rodadas.
   - O cosseno normalizado pelo teto é instável (> 1 em algumas rodadas). Com 5 batches por metade,
     o teto é ruidoso: usar só como indicação.
   - **Magnitude Φ:** intra-dataset 0.89–0.94; ISIC seg×cls 0.57–0.61; **entre datasets
     0.40–0.43**. As normas de gradiente diferem muito entre datasets, então alguns datasets
     dominam a direção do trunk. Isso é informação que o cosseno não dá.

3. **M2 (lookahead com o update real), em Z × 1e-4, médias sobre 21 rodadas:**

   | classe de par | `client` train / val | `dataset` train / val |
   |---|---|---|
   | self | +7.9 / −11.1 | +11.2 / −32.9 |
   | aggregate | +5.4 / +2.4 | +8.3 / +3.4 |
   | intra, mesmas tarefas | −10.3 / −4.1 | −36.3 / −23.0 |
   | intra, tarefas diferentes (ISIC seg×cls) | −15.0 / −7.0 | −25.2 / −14.1 |
   | entre datasets | −7.6 / −2.8 | −7.5 / −1.5 |

   - Entre datasets, o efeito é **negativo e consistente** no split de treino (80–100% das rodadas
     com média < 0 na maioria dos pares) e **cresce no fim do treino** (r ≤ 61: −3.2; r ≥ 141:
     −12.4).
   - O alvo mais afetado é o **BUSI**: ISIC → BUSI −33 / −13 (95% das rodadas negativas), TCGA →
     BUSI −22 / −6. O SIIM quase não reage (|Z| < 0.5), porque a loss dele quase não se move.
   - **Self no fim:** train ≈ 0, val −52 a −67, ou seja, overfitting nas duas políticas (coerente
     com A3/A4).
   - **Aggregate > 0 nas duas políticas**: a média federada reduz a loss, e mais com a seed por
     dataset.
   - **⚠ Limitação que impede afirmar transferência negativa:** Z aplica o Δᵢ **inteiro** de outro
     cliente. Perto de um mínimo, *qualquer* passo dessa norma que não esteja alinhado aumenta a
     loss pela curvatura (½ ΔᵀHΔ > 0). O Z negativo entre datasets pode ser só curvatura, e não
     conflito. Falta um **controle nulo**: a mesma conta com uma direção aleatória de mesma norma
     (e/ou com Δᵢ escalado pelo peso real de agregação wᵢ, já que o custo de curvatura escala com
     wᵢ² e o efeito de 1ª ordem com wᵢ).

4. **Desempenho: `dataset` vs `client` (diferenças maiores que o ruído do item 1 em negrito):**
   - TCGA: balanced acc **0.858 → 0.907**, Dice positivo **0.604 → 0.641**, AUC 0.931 → 0.956.
   - ISIC cls: balanced acc 0.351 → 0.390, **macro-F1 0.214 → 0.271**, AUC 0.756 → 0.785.
   - BUSI: balanced acc 0.603 → 0.661, que está dentro do ruído do BUSI. A dispersão entre os 2
     clientes cai (Dice mín/máx 0.553/0.747 → 0.588/0.707).
   - **SIIM seg colapsa para 0.000 nos 4 clientes** (contra 0.066 no `client`). O Dice de validação
     cai a 0 a partir de r100 e não volta, enquanto no `client` ele sobe no fim (0.012 → 0.079).
     SIIM cls: macro-F1 0.494 → 0.462.

**Conclusão da E0** (formulação para relatório; uma seed):
- A métrica de cosseno é válida. Com stems inicializados por cliente, ela mostra **ausência de
  interação** entre clientes ao longo de todo o treino.
- A seed por dataset cria alinhamento intra-dataset real, que decai com o treino mas não some, e
  está associada a ganhos acima do ruído em TCGA e ISIC cls. Também está associada ao colapso
  completo da segmentação do SIIM.
- Entre datasets, o cosseno é ≈ 0, mas o lookahead é sistematicamente negativo, sobretudo sobre o
  BUSI. Isso é **compatível com transferência negativa**, mas **ainda não a demonstra**: falta o
  controle nulo de curvatura (item 3).

**E0.3.9 — Controle nulo do M2 (2026-10-15).**

`scripts/gradient_conflict_methods.py --null-control`, offline, sobre o estado final (r200) dos dois
treinos completos. Para cada par (i → j), Z foi calculado com quatro passos:
- o Δᵢ real;
- uma direção **aleatória com a mesma norma por tensor**;
- Δᵢ × wᵢ, com o peso real de agregação;
- a direção aleatória × wᵢ.

Saída: `runs/validation/gradient_conflict_methods/20261015_*/m2_null_control.csv`.

| Z × 1e-4 (train / val) | Δ real | aleatório, mesma norma | Δ × wᵢ | aleatório × wᵢ |
|---|---|---|---|---|
| **entre datasets**, `client` | **−16.1 / −4.4** | +0.6 / +1.9 | +0.4 / +1.4 | +0.7 / +1.9 |
| **entre datasets**, `dataset` | **−22.1 / −6.7** | +0.3 / −0.6 | −0.2 / −0.2 | +0.1 / −0.2 |
| intra, mesmas tarefas, `dataset` | −149.0 / −40.0 | +0.2 / −0.4 | +0.2 / +2.3 | +0.1 / −1.0 |
| self, `client` | +467 / −182 | +1.0 / −0.3 | **+60 / +8** | +1.3 / +1.0 |
| self, `dataset` | +493 / −58 | −0.5 / +0.1 | **+50 / +0.4** | +0.7 / +0.9 |

**Leitura:**
1. **A curvatura genérica não explica o Z negativo.** Uma direção aleatória de mesma norma quase
   não muda a loss (|Z| < 2e-4). O update real de outro dataset é bem pior que o aleatório (−16 a
   −22 no train): ele aponta para direções de **alta curvatura da loss do alvo**. Isso é
   interferência real, embora o cosseno de 1ª ordem seja ≈ 0. É coerente com o "tragic triad" do
   PCGrad: o conflito aqui é de curvatura, e não de direção.
2. **Na escala em que a agregação realmente aplica o update (× wᵢ), a interferência entre datasets
   some:** −0.2 a +0.4e-4, indistinguível do aleatório escalado. O custo de curvatura escala com wᵢ²
   e os pesos são pequenos (6–12% por cliente), por isso ele fica desprezível.
3. **O efeito dominante da federação é a diluição do próprio passo.** O self cai de +467/+493 com o
   Δ inteiro para +50/+60 com Δ × wⱼ, ou seja, cerca de 10× menos progresso por rodada no próprio
   objetivo. Os outros clientes não compensam (intra/inter × wᵢ ≈ 0).
4. Entre clientes do mesmo dataset na política `dataset`, o Δ inteiro de um irmão é muito prejudicial
   (−149), mas também é desprezível na escala real (+0.2).

**Conclusão sobre transferência negativa** (estado final, uma seed, um holdout):
> Não há transferência negativa entre datasets **na escala aplicada pela agregação**. Os updates
> de outros datasets interferem por curvatura quando aplicados inteiros, mas os pesos hierárquicos
> os reduzem a um efeito desprezível. O custo mensurável da federação é a **diluição** do passo de
> cada cliente (~10×), e não o conflito entre datasets.

Ressalvas: só foi medido o estado final, e o M2 online mostrou o efeito entre datasets crescendo
ao longo do treino (com Δ inteiro). O M2 online não registra a versão escalada; incluir Δ × wᵢ na
coleta online é uma melhoria simples para a E3.

**Pendências antes de fechar a E0:**
- ~~(a) Controle nulo do M2~~ → **feito na E0.3.9.**
- ~~(b) Investigar o colapso do SIIM seg~~ → **movido para a E3/E4** (decisão do usuário em
  2026-10-15).
- ~~(c) Replicar com mais seeds~~ → **movido para a E3/E4** (decisão do usuário em 2026-10-15).

**Critério de pronto.**
- A métrica atual foi validada ou refutada com controles.
- Os 2 métodos novos estão implementados, testados e rodados num treino completo.
- Os conflitos medidos foram interpretados.
- D9 da E1 usa os métodos validados.

**Registro.**
- 2026-10-08 — etapa criada a pedido do usuário, após a constatação de cosseno ≈ 0 em todos os pares.
- 2026-10-08 — E0.1 concluída: a métrica é válida, e a ortogonalidade vem dos stems/cabeças
  inicializados por cliente (H3). Próximo passo: decisão do usuário sobre o desenho e, depois, E0.2.
- 2026-10-14 — Treinos completos analisados (E0.4).
  - Não há conflito de 1ª ordem entre datasets.
  - O M2 pareado precisa de uma referência nula.
  - O piso de ruído entre execuções é de ~±0.02.
  - Pendências (a)–(e) aguardam decisão.
- 2026-10-15 — Controle nulo do M2 (E0.3.9). O Z entre datasets é interferência de curvatura com o
  Δ inteiro, e desprezível na escala real da agregação; o custo dominante é a diluição (~10×).
  (b) e (c) foram para a E3/E4. **E0 concluída**, faltando apenas o PR.
- 2026-10-14 — Treinos completos concluídos e analisados (E0.3.8).
  - O ruído GPU entre runs (até ~0.05 no BUSI) foi quantificado.
  - A seed por dataset cria alinhamento intra-dataset, ganha em TCGA/ISIC cls e colapsa o SIIM seg.
  - O M2 entre datasets é negativo, mas falta o controle nulo de curvatura para concluir.
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

## B0 — Baselines não federados por dataset

**Objetivo** (pedido do usuário em 2026-10-15, antes de continuar). Ter uma referência de quanto um
treinamento normal, **não federado**, alcança em cada dataset, separadamente para segmentação,
classificação e multitarefa.

**Decisões do usuário:**
- **Partição congelada:** treino com a união dos clientes de cada dataset no mesmo master do
  federado, avaliado com `unified_eval` nas **mesmas fatias de teste por cliente**, e portanto
  diretamente comparável aos runs federados.
- **MTnnUNet:** o mesmo modelo do federado, com só a loss de seg, só a de cls, ou as duas.
- **1 seed (1993):** diferenças menores que o ruído entre runs (até ~0.05 no BUSI, ~0.01 nos demais;
  E0.3.8) não são efeito.

**Implementação** (`src/training_baseline.py`):
- O pool de treino/val de um (dataset, modo) é a união das fatias `train`/`val` de todos os clientes
  daquele dataset que possuem a(s) tarefa(s). O código recusa uma imagem que apareça em dois clientes.
- No modo multitarefa, a função `pivot_multitask_mapping` deixa uma linha por imagem só com a
  supervisão concedida. No ISIC, os pools de seg e cls são disjuntos, então o baseline multitarefa é
  parcialmente supervisionado (`has_mask`/`has_label`).
- A loss multitarefa usa os mesmos λ do federado (`aggregation.task_weights`, seg 0.8 / cls 0.2). Os
  pesos de classe são `balanced_fold` e o critério de classificação é o de cada dataset, como nos
  clientes.
- O laço de treino é **centralizado e normal**: até `training.epochs` (200) épocas, com seleção do
  melhor modelo e early stopping pela loss de validação do pool (`max_patience` 50) e o scheduler
  configurado (plateau).
  - **Diferente do budget fixo de 200 rodadas do federado, por desenho:** o baseline mede o que um
    treino comum alcança.
- Saída em `runs/baseline/<segmentation|classification|multitask>/<dataset>_<timestamp>/`:
  - `config.yaml`, `execution.log`;
  - `history.csv` (por época);
  - `fold_0/best.pt`;
  - `baseline_test_results.csv` e `baseline_cls_predictions.csv` (mesmo esquema do federado,
    `setup=baseline`).
- Testes: `tests/test_training_baseline.py` (composição do pool, ISIC disjunto, teste fora do pool,
  imagem duplicada).
- Smoke na GPU: os 12 baselines com 1 época e 16 amostras passaram (`runs/validation/baseline_smoke/`).

**Tamanhos e tempo medidos:**
- Pools medidos: SIIM 6.743 imagens de treino (~30 s/época) e ISIC cls 5.578 (~20 s/época).
- Os demais pools foram estimados pela proporção: BUSI ~250 (~1 s/época) e TCGA/ISIC seg ~2.700
  (~12 s/época).
- Pior caso, sem early stopping: ~3h (seg) + ~3.5h (cls) + ~4h (multitarefa) em sequência. Com
  early stopping deve ser bem menos.

**Treino pelo usuário** (configuração = a do run de 29/09, que fixa o protocolo):
```bash
python -m src.training_baseline --config runs/20260929_082717_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-TCGA_LGG-SIIM_ACR/config.yaml --mode segmentation --dataset all
python -m src.training_baseline --config runs/20260929_082717_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-TCGA_LGG-SIIM_ACR/config.yaml --mode classification --dataset all
python -m src.training_baseline --config runs/20260929_082717_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-TCGA_LGG-SIIM_ACR/config.yaml --mode multitask --dataset all
```
- Os três comandos podem rodar em paralelo, um por terminal. Usam pouca memória de GPU, e cada um
  roda os 4 datasets em sequência.
- Sucesso: 4 linhas `BASELINE_OK` por comando e um `baseline_test_results.csv` em cada pasta.

**Critério de pronto.**
- 12 runs concluídos.
- Tabela baseline × federado (`client`/`dataset`) por dataset e tarefa registrada aqui, com a faixa
  de ruído.

**Resultado (2026-10-16).**
- Os 12 runs foram concluídos.
- Relatório: `runs/baseline/report.html`, com tabela, gráfico, matrizes de confusão de cls e de
  pixel, e curvas. Gerador: `scripts/baseline_report.py`; recriar com
  `bash runs/baseline/build_report.sh`.
- Uma seed, um holdout. "Ruído" = |29/09 − federado seed/cliente|, que são dois runs com a mesma
  configuração.

| dataset | métrica | base seg | base cls | base multitask | fed seed/cliente | fed seed/dataset | ruído |
|---|---|---|---|---|---|---|---|
| BUSI | balanced acc | — | 0.473 | 0.505 | 0.603 | **0.661** | ±0.052 |
| ISIC | balanced acc | — | 0.481 | **0.549** | 0.351 | 0.390 | ±0.001 |
| SIIM | balanced acc | — | **0.772** | 0.763 | 0.590 | 0.583 | ±0.002 |
| TCGA | balanced acc | — | 0.887 | **0.923** | 0.858 | 0.907 | ±0.004 |
| BUSI | Dice positivo | **0.683** | — | 0.678 | 0.650 | 0.648 | ±0.020 |
| ISIC | Dice positivo | **0.880** | — | 0.876 | 0.842 | 0.846 | ±0.001 |
| SIIM | Dice positivo | 0.005 | — | 0.000 | **0.066** | 0.000 | ±0.002 |
| TCGA | Dice positivo | **0.733** | — | 0.709 | 0.604 | 0.641 | ±0.009 |

**Leitura:**
1. **Classificação: o federado fica bem abaixo do baseline em ISIC e SIIM.**
   - ISIC: 0.35–0.39 vs 0.48–0.55.
   - SIIM: 0.59 vs 0.77.
   - As matrizes mostram o porquê:
     - No ISIC, o federado desloca as predições para **MEL**: 50% dos NV são preditos como MEL no
       `seed/cliente`, enquanto o baseline acerta 63–68% dos NV.
     - No SIIM, o federado marca a maioria dos negativos como pneumotórax (recall de
       `no_pneumothorax` 0.38–0.46, contra 0.70–0.74 no baseline).
   - Em ambos, o federado erra **muito mais** que o baseline treinado com os mesmos dados.
2. **BUSI é a exceção.** O federado supera o baseline (0.60–0.66 vs 0.47–0.51), acima do ruído do
   BUSI (±0.05). As curvas mostram o baseline de cls do BUSI com validação muito ruidosa (dataset
   pequeno, ~250 imagens): o early stopping escolhe um checkpoint instável.
3. **Segmentação: o baseline é melhor ou igual.** ISIC +0.04 e TCGA +0.09/+0.13, acima do ruído;
   BUSI +0.03, no limite do ruído.
4. **SIIM seg não é aprendida por nenhum setup**, nem pelo baseline centralizado (Dice positivo
   0.005; a matriz de pixel mostra 99.6–100% dos pixels de lesão preditos como fundo; a loss de
   validação estaciona em 0.25 depois de ~50 épocas). **O colapso do SIIM não é causado pela
   federação nem pela seed por dataset:** a tarefa, com esta resolução (128 px) e esta loss, não é
   aprendida. Isso reforça a entrada da E4 (resolução) e da E3 (loss) para o SIIM.
5. **Multitarefa vs tarefa única (baseline):**
   - melhora a classificação: ISIC +0.07, TCGA +0.04, BUSI +0.03 (dentro do ruído do BUSI);
   - mantém ou piora levemente a segmentação: TCGA −0.02, demais ≈ iguais.
6. **FP em imagem vazia (BUSI) = 1.0 em todos os setups.** Toda imagem `normal` de teste recebe
   algum pixel predito como lesão. Isso é comum a todos os setups e deve ser investigado na E1
   (limiar ou pós-processamento estão fora do protocolo atual).

**Ressalvas:**
- O orçamento é diferente por desenho: os baselines usam early stopping pela validação e o federado
  usa 200 rodadas fixas, com avaliação na última. Parte da vantagem do baseline na classificação
  pode vir da seleção do checkpoint, já que o federado faz overfitting no fim (E0.3.8, self Z val
  < 0).
- **Teto de épocas:** seg SIIM (melhor época 199), seg TCGA (190), multitask BUSI (167) e multitask
  TCGA (177) pararam nas 200 épocas ainda melhorando. Esses baselines podem estar
  **subestimados**.

**Registro.**
- 2026-10-16 — 12 baselines concluídos; relatório gerado; resultados acima.
- 2026-10-16 — **correção do item 4 da Leitura** (no detalhamento da E3). A seg do SIIM **é**
  aprendida no início: o Dice positivo de validação chega a 0.237 na época 14 e depois cai para
  ≈ 0, enquanto a val loss continua caindo, e o checkpoint escolhido é o da época 199. O problema é
  um desalinhamento entre a loss e a métrica, não só a resolução. O texto original foi mantido acima
  como registro.
- 2026-10-15 — etapa criada a pedido do usuário. Implementação, testes e smoke concluídos;
  aguardando os treinos.

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

#### Revisão do detalhamento após E0 e B0 (2026-10-16, aprovada pelo usuário)

O detalhamento E1.1–E1.4 de 2026-10-08 é anterior à E0 e à B0, e duas coisas mudaram:
- a D9 original lia "sem conflito" onde o certo é **"sem interação"** (E0.1), e a E0 deixou dois
  métodos validados (M1/M2) e um controle nulo (E0.3.9) que mudam a leitura do Z;
- a B0 revelou quatro problemas que nenhuma checagem cobria: federado abaixo do baseline, colapso de
  predição numa classe, FP em imagem vazia e teto de épocas → **D12–D15**.

**Vintages de artefatos** (determinam o que cada checagem consegue ler):

| Vintage | Runs | O que existe | Consequência |
|---|---|---|---|
| Federado novo | `20260929_082717_*`, `20261014_211153_*`, `20261014_211215_*` | `training_history.csv` por cliente (longo), `metadata.yaml`, `aggregation_history.json` com `val_*`/`train_*` por cliente, `negative_transfer_summary.csv`; os dois de 14/10 também têm `conflict_*.csv` | todas as checagens |
| Estudo antigo | 10 braços de `runs/studies/multi_dataset_balance_v1` (BUSI+ISIC, CV=4, 50 rodadas) | **sem `training_history.csv`**; histórico de agregação sem métricas de val por cliente; só `multitask_primary` tem `gradient_conflict`; `execution.log` tem a métrica de val por dataset/tarefa por rodada; pastas `fold_*.incomplete_*` | checagens de série por cliente → `n/a`; fallback por dataset/tarefa via `execution.log`; D12 → `n/a` (outra partição) |
| Baseline | 12 em `runs/baseline/<modo>/<ds>_<ts>/` | `history.csv` largo por época, `baseline_test_results.csv` (com `best_epoch`), `baseline_cls_predictions.csv` | D1, D3–D7, D13–D15; D2 só no modo `classification` |

#### E1.1 — Diagnóstico offline: `src/experiments/training_diagnostics.py` (novo)

- **Entrada:** um ou mais caminhos. Diretório com `*_test_results.csv` → um run (`kind` =
  `federated`/`standalone`/`baseline`). Diretório com `run_index.csv` → estudo (itera `run_path`,
  anotando `arm_id`/`seed`). Outro diretório → busca runs até 3 níveis (serve para `runs/baseline`).
  Pastas `fold_*.incomplete_*` são ignoradas.
- **Leitura**, tolerante a artefato ausente (→ `n/a` com motivo, nunca exceção):
  - histórico por cliente lido **uma vez**: treino = `local_step`/`local_epoch` (agregado por rodada
    para as regras de terço; a série por step fica para picos e `grad_norm`); val =
    `post_aggregation_round` (federado e standalone). `post_local_round` é ignorado (A8);
  - `history.csv` do baseline convertido para o mesmo formato;
  - fallback `execution.log` (`per-dataset/task metric {...}`) com `source=execution_log`;
  - `metadata.yaml`, `aggregation_history.json`, `negative_transfer_summary.csv`,
    `conflict_summary.csv`, `*_test_results.csv`, `*_cls_predictions.csv`, `config.yaml`.
- **Finding:** `run, kind, arm_id, seed, check_id, severity (critical|high|info), fold, dataset,
  client_id, task, metric, value, threshold, evidence, source`. Só checagens disparadas viram linha; o
  estado de cada checagem (`ok`/`triggered`/`n/a` + motivo) vai para o `summary.json`.

| id | Regra (limiar padrão) | Sev. | Reencontra |
|---|---|---|---|
| D1 | NaN/Inf em qualquer valor do histórico | critical | A5 |
| D2 | só CE: último terço da loss de treino cls ≥ 0.95·ln K | critical | A1 |
| D3 | variação relativa último/primeiro terço da loss de treino > −5% | high | A1 |
| D4 | último terço > primeiro terço, ou > 1% dos steps acima de mediana + 10·MAD | critical | — |
| D5 | melhor val loss antes de 60% das rodadas, val final > 1.15·mín e treino ainda caindo (`info` no baseline, que seleciona o melhor checkpoint) | high | A3, A4 |
| D6 | val `dice_positive` no último terço < 0.05 | critical | A2 |
| D7 | val balanced acc. no último terço ≤ 1/K + 0.05 | critical | A1 |
| D8 | (máx − mín) da métrica de teste entre clientes do mesmo dataset/tarefa > 0.15 com algum cliente em D6/D7 | high | A2 |
| D9 | conflito/interação (abaixo) | info/high | A6, E0 |
| D10 | épocas efetivas por cliente; razão máx/mín > 5 | info | — |
| D11 | `grad_norm` não-finita → critical; mediana do último terço > 10× a do primeiro → high | — | (E1.2) |
| D12 | média por dataset/tarefa < melhor baseline − ruído (BUSI 0.05, demais 0.01) | high (fed) / info (standalone) | B0 itens 1 e 3 |
| D13 | classe mais predita c*: `frac_pred(c*) − prev(c*) ≥ 0.20` → high; `frac_pred(c*) ≥ 0.95` → critical | high/critical | ISIC NV→MEL, SIIM → pneumotórax |
| D14 | `empty_fp_image_rate ≥ 0.5` com `n_empty ≥ 5` | high | BUSI = 1.0 |
| D15 | baseline sem early stop e `best_epoch ≥ 0.9·epochs` → high; federado/standalone com mínimo da val loss e melhor métrica nos últimos 10% das rodadas → info | high/info | baselines seg SIIM/TCGA, multitask BUSI/TCGA |

**D9 reformulada** (a fonte usada vai em `source`):
1. `conflict_summary.csv` (M1/M2):
   - **sem interação** (info): |cos M1 intra, mesmas tarefas| < 0.02. A leitura é "os clientes não
     interagem pelo trunk", e não "sem conflito";
   - **conflito de 1ª ordem** (high): cos M1 médio de uma classe de par ou par de datasets < −0.05;
   - **interferência** (info, com ressalva): par de datasets com Z_train < 0 e Z_val < 0, ambos com
     `média + 2·dp/√rodadas < 0` (consistência descritiva). A evidência registra que o Z com o Δ
     inteiro superestima o efeito, e que na escala ×wᵢ ele some (E0.3.9);
   - **overfitting** (high): `self` com Z_train ≥ 0 e Z_val < 0;
   - **agregado prejudicial** (high): `aggregate` com Z_train < 0.
2. Sem M1/M2: `negative_transfer` (`overall − orthogonal_reference` > 0.05 → high; |excesso| ≤ 0.02
   → info "sem interação").
3. Só `gradient_conflict` no histórico: cosseno médio intra-dataset, mesma regra de "sem interação".
4. Nada → `n/a`.

**D12:** `--baseline-root` (padrão `runs/baseline`), baseline mais recente por (modo, dataset). Só
compara quando `paths.partition_file(config)` resolve para o mesmo master nos dois configs; senão
`n/a` ("fatias de teste diferentes"). Métricas: `balanced_acc` e `macro_f1` (cls), `dice_positive`
(seg). Referência = melhor modo de baseline com a tarefa; a evidência lista todos.

- Limiares: dicionário de constantes do módulo, sobrescrevível por `--threshold CHAVE=VALOR` e
  `--noise DATASET=VALOR`. **Nada entra no hash científico.**
- **Saída:** `<run>/diagnostics/{findings.csv, summary.json, summary.md}`; num estudo ou diretório de
  vários runs, também `<dir>/diagnostics/` consolidado. `summary.json`: `status` (`critical`,
  `warning` ou `ok`), contagens por severidade e estado por checagem.
- **CLI:** `python -m src.experiments.training_diagnostics PATH [PATH ...] [--threshold ...]
  [--noise ...] [--baseline-root ...]`. Sai com 0: o diagnóstico não é um gate.
- **Integração não-fatal** no fim de `training_federated.run` (nos dois setups) e de
  `training_baseline.run`, desligável em `runtime.diagnostics.training.enabled` (o bloco `runtime` já
  é excluído do hash e da assinatura de resume).

#### E1.2 — Guardas online somente leitura

- `local_trainer.train_local`, entre `backward()` e `step()`: `torch.nn.utils.get_total_norm` dos
  gradientes; não-finita → `FloatingPointError`. Registro como `grad_norm` por step (histórico
  detalhado), `grad_norm_mean`/`grad_norm_max` por época e por rodada (`post_local_round`), nas
  métricas de `fit` e em `train_grad_norm_{mean,max}` no `history.csv` do baseline.
  Ligado por `runtime.diagnostics.grad_norm.enabled` (padrão `true`); se o custo medido passar de 2%
  do step, o padrão vira `false`.
- `server.aggregate_fit`: `_validate_update_finite` logo após `_validate_update_shapes`; um tensor
  não-finito → `FloatingPointError` com o cliente e o tensor, antes de agregar.

#### E1.3 — Testes

- `tests/test_training_diagnostics.py`: fixtures sintéticas por checagem (dispara / não dispara), run
  antigo → `n/a`, `post_local_round` não contado duas vezes, D12 com partição diferente → `n/a`, D9 sem
  interação vs interferência vs overfitting, estudo com `.incomplete`, override de limiar.
- Gradiente Inf com loss finita → `FloatingPointError`; NaN devolvido por um cliente → erro no servidor.
- **Não-regressão bit a bit em CPU** (padrão `test_probes_do_not_perturb_training`): `train_local` e
  `client.fit` com a norma ligada e desligada → pesos e losses idênticos.
- Padrões/validação dos campos novos de `runtime.diagnostics`.

#### E1.4 — Execução e relatório

1. `python -m unittest discover -v`; smoke CPU `python -m scripts.smoke_federated --setup both
   --holdout --samples 2` e um baseline mínimo em CPU.
2. Custo da norma na GPU (só se nenhum treino estiver rodando).
3. Diagnóstico sobre os 3 runs avulsos, os 10 braços de `multi_dataset_balance_v1` e os 12 baselines.
4. Conferir A1–A4 (E1.0), D12–D15 (B0) e a leitura de conflito da E0; registrar a tabela por run abaixo.

**Treino pelo usuário:** não é necessário (guardas somente leitura; diagnóstico offline).

**Fora do escopo:** memória RSS por ator Ray (pendência E0.4 (e)) — proposta para a E2/E3.

#### E1 — Resultado da implementação e da execução (2026-10-16)

**Código** (branch `feat/training-validation-v2`):
- `src/experiments/training_diagnostics.py` (novo): D1–D15, CLI, saída `diagnostics/`, descoberta de
  run/estudo/pasta de runs, leitores tolerantes às três vintages;
- `src/federated/local_trainer.py`: `gradient_norm` (`get_total_norm`, somente leitura) entre
  `backward()` e `step()`, `FloatingPointError` se não-finita; `grad_norm` por step,
  `grad_norm_{mean,max}` por época e por rodada;
- `src/federated/client.py`: linhas `grad_norm_{mean,max}` no `post_local_round` e
  `train_grad_norm_{mean,max}` nas métricas de `fit` (logo, no `aggregation_history.json`);
- `src/federated/server.py`: `_validate_update_finite` antes de agregar;
- `src/training_baseline.py`: `train_grad_norm_{mean,max}` no `history.csv`;
- gancho não-fatal `run_training_diagnostics` no fim de `training_federated.run` e de
  `training_baseline.run`;
- `runtime.diagnostics.grad_norm.enabled` e `runtime.diagnostics.training.{enabled, baseline_root}`
  (padrões `true` / `runs/baseline`), validados em `training_runtime.py`, fora do hash científico.

**Testes:** `tests/test_training_diagnostics.py` (24) e `tests/test_gradient_guards.py` (9), incluindo
a **não-regressão bit a bit em CPU** (pesos, estado do Adam e cada loss por step idênticos com a norma
ligada e desligada), gradiente Inf com loss finita → erro antes do `step()`, NaN de um cliente → erro
no servidor. Suíte completa: **197 OK**. `sync_agent_assets --check`: OK.

**Smoke CPU:** `python -m scripts.smoke_federated --setup both --holdout --samples 2` → `SMOKE_OK`
(federado e standalone) e `SMOKE_ANALYSIS_OK`; o diagnóstico rodou no fim de cada um e `grad_norm`
aparece no histórico e no `aggregation_history.json`. Baseline mínimo em CPU (multitask BUSI, 2 épocas,
16 amostras, config em fp32) → `BASELINE_OK` com `train_grad_norm_*` no `history.csv`. Artefatos em
`runs/validation/e1_smoke/`.

**Custo da norma (GPU, RTX 5070 Ti, MTnnUNet BF16, batch 32, 128 px, 10 steps/rodada, 15 pares
intercalados):** mediana 509 → 517 ms por rodada, **+1.6%** (abaixo do limite de 2%). O padrão fica
ligado.

**Ajustes de calibração feitos ao rodar nos artefatos reais:**
- **D4 (picos):** a regra por step (mediana + 10·MAD) disparava em todos os clientes de seg, porque a
  loss de um step é bimodal (batches com e sem pixel de lesão) e porque as rodadas iniciais de uma
  loss decrescente ficam "acima da mediana". Agora o pico é lido na média por rodada, contra a
  **mediana móvel** (janela 11), e exige > 3× o nível local **e** > 10·MAD. Nos runs saudáveis a
  razão máxima valor/mediana local é 1.87.
- **D15 (baseline):** com 0.9·épocas o multitask BUSI (167) e o TCGA (177) não eram pegos; o limiar
  ficou em **0.8** com a exigência de ter usado as 200 épocas. Pega exatamente os 4 da B0.
- O `run_index.csv` de `multi_dataset_balance_v1` lista só 2 dos 10 braços; a descoberta também
  varre `runs/seed_*/` do estudo.

**Achados por run** (sev. = critical/high/info; `n/a` = checagens sem dados; uma seed, descritivo):

| run | tipo | status | sev. | n/a | checagens disparadas |
|---|---|---|---|---|---|
| `20260929_082717` | federated | critical | 10/22/10 | 1 | D2 D3 D5 D6 D7 D9 D10 D12 D13 D14 D15 |
| `20261014_211153` (seed/cliente) | federated | critical | 10/22/18 | 1 | D2 D3 D5 D6 D7 D9 D10 D12 D13 D14 D15 |
| `20261014_211215` (seed/dataset) | federated | critical | 11/20/15 | 1 | D2 D3 D5 D6 D7 D9 D10 D12 D13 D14 D15 |
| `balance_v1/ablation_budget` | federated | ok | 0/0/3 | 9 | D15 |
| `balance_v1/ablation_flat` | federated | ok | 0/0/11 | 9 | D15 |
| `balance_v1/ablation_focal` | federated | critical | 1/1/9 | 9 | D10 D13 D15 |
| `balance_v1/ablation_weighting` | federated | ok | 0/0/11 | 9 | D15 |
| `balance_v1/local_epochs_ce` | standalone | ok | 0/0/8 | 9 | D15 |
| `balance_v1/local_steps_ce` | standalone | ok | 0/0/10 | 9 | D10 D15 |
| `balance_v1/local_steps_focal` | standalone | warning | 0/1/11 | 9 | D10 D13 D15 |
| `balance_v1/multitask_local` | standalone | warning | 0/1/4 | 9 | D10 D13 D15 |
| `balance_v1/multitask_primary` | federated | warning | 0/1/5 | 8 | D9 D10 D13 D15 |
| `balance_v1/primary` | federated | warning | 0/2/10 | 9 | D10 D13 D15 |
| `baseline/classification/Curated_BUSI` | baseline | ok | 0/0/0 | 6 | — |
| `baseline/classification/ISIC_2018` | baseline | ok | 0/0/1 | 6 | D5 |
| `baseline/classification/SIIM_ACR` | baseline | ok | 0/0/0 | 6 | — |
| `baseline/classification/TCGA_LGG` | baseline | ok | 0/0/1 | 6 | D5 |
| `baseline/multitask/Curated_BUSI` | baseline | warning | 0/3/0 | 5 | D14 D15 |
| `baseline/multitask/ISIC_2018` | baseline | ok | 0/0/1 | 6 | D5 |
| `baseline/multitask/SIIM_ACR` | baseline | critical | 1/0/0 | 5 | D6 |
| `baseline/multitask/TCGA_LGG` | baseline | warning | 0/1/0 | 5 | D15 |
| `baseline/segmentation/Curated_BUSI` | baseline | warning | 0/2/0 | 7 | D14 |
| `baseline/segmentation/ISIC_2018` | baseline | ok | 0/0/0 | 8 | — |
| `baseline/segmentation/SIIM_ACR` | baseline | critical | 1/1/0 | 7 | D6 D15 |
| `baseline/segmentation/TCGA_LGG` | baseline | warning | 0/1/0 | 7 | D15 |

Detalhe em `<run>/diagnostics/summary.md`; consolidados em
`runs/studies/multi_dataset_balance_v1/diagnostics/` e `runs/baseline/diagnostics/`.

**O que o diagnóstico reencontra:**
- **A1 (SIIM cls não aprende):** D2 nos 4 clientes SIIM nos 3 runs (CE do último terço 0.67–0.71 vs
  ln 2 = 0.693), D3 (−0.7% a −1.9%) e D7 (balanced acc. de val 0.51–0.54).
- **A2 (SIIM seg colapsada):** D6 em `mt_1`/`mt_3` no 29/09 e no `client`; nos **4** clientes no
  `dataset` (o colapso da E0.3.8). Também D6 nos baselines seg e multitask do SIIM: a tarefa não é
  aprendida por nenhum setup (B0 item 4).
- **A3 (overfitting de cls com avaliação na última rodada):** D5 em `Curated_BUSI_mt_1` (melhor na
  rodada 87) e `TCGA_LGG_mt_0` (melhor na 92) nos 3 runs.
- **A4 (ISIC cls não generaliza):** **não** pela D5 (a val loss fica estável, 1.90 → 1.93, sem subida de
  15%); aparece pela **D12** (balanced acc. −0.16 a −0.20 vs baseline) e pela **D13** (MEL predito
  para 38–47% das imagens contra 11% de prevalência).
- **A5:** nenhum D1 (nenhum valor não-finito) e, após a calibração, nenhum D4 em nenhum run.
- **A6 / E0:** D9 "sem interação" no 29/09 (cancelamento − referência = −0.0006), no `client`
  (cos M1 intra, mesmas tarefas = 0.003) e no `multitask_primary` antigo (cos dos deltas 0.004); **não**
  dispara no `dataset` (cos M1 intra 0.157), como esperado. Nos dois runs de 14/10: **overfitting**
  (Z `self` train ≥ 0, val < 0) como high, e **interferência** entre datasets (Z train e val < 0 com
  sinal consistente) como info, sempre com a ressalva do Δ inteiro (E0.3.9). Os pares que passam no
  critério de consistência mudam entre os dois runs (ex.: ISIC → BUSI só no `client`).
- **B0:**
  - D12 (abaixo do baseline além do ruído): ISIC cls/seg, SIIM cls, TCGA cls/seg nos 3 runs; BUSI seg
    só no 29/09 (−0.052, no limite do ruído de 0.05). BUSI cls nunca (o federado é melhor);
  - D13: ISIC → MEL e SIIM → pneumotórax (57–66% preditos vs 22% de prevalência) nos 3 runs;
  - D14: BUSI = 1.0 no teste em todos os setups federados e nos baselines seg/multitask; também na
    validação de alguns clientes TCGA e SIIM (0.53–0.72);
  - D15: os 4 baselines de B0 (seg SIIM 199, seg TCGA 190, multitask BUSI 167, multitask TCGA 177).

**Achados novos:**
- No estudo antigo, a Focal no ISIC colapsa para **NV** (`ablation_focal`: 99% das predições, D13
  critical; `local_steps_focal`: 91%), enquanto a CE desloca para MEL. O BUSI cls também sobre-prediz
  `normal` (35% vs 14%) em `primary` e `ablation_focal`.
- D15 é frequente no federado (clientes de seg ISIC/SIIM/TCGA com o mínimo da val loss nas últimas
  rodadas): o budget de 200 rodadas também é um teto para a seg, ao lado do overfitting da cls (D5).
- D10: exposição desigual por budget `steps` (razão 14× entre BUSI e SIIM no 4-datasets; 29–47× no
  estudo antigo), como previsto em AGENTS.md.

**Divergências em relação ao plano:**
- A4 não é detectado pela D5 (ver acima); fica coberto por D12/D13.
- Os 10 braços do estudo não têm histórico por cliente: D1–D7/D11 ficam `n/a`; D5/D6/D15 só pelo
  fallback do `execution.log` (por dataset/tarefa) e D12 `n/a` (partição BUSI+ISIC CV=4, diferente do
  master dos baselines).
- O teste de não-regressão é no nível de `train_local` (com o mesmo cliente MTnnUNet do teste de
  sondas), não de `client.fit` completo.
- Guardas e integração saíram num único commit, em vez de dois.
- Memória RSS por ator Ray (pendência E0.4 (e)) ficou fora; registrada na E2.

**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-08 — branch `feat/training-validation` criada; auditoria preliminar (E1.0) feita;
  detalhamento E1.1–E1.4 escrito.
- 2026-10-16 — detalhamento revisto após E0/B0 (D9 reformulada, D12–D15 novas, vintages de
  artefatos) e aprovado pelo usuário; branch `feat/training-validation-v2` a partir da `main`
  (PR #8 mesclado).
- 2026-10-16 — **E1 concluída.** Diagnóstico D1–D15 e guardas de gradiente implementados e
  testados (197 OK, não-regressão bit a bit em CPU, custo +1.6% na GPU); smoke CPU OK; diagnóstico
  rodado nos 25 runs, reencontrando A1–A3, A5, A6 e os achados da B0 (A4 via D12/D13). Calibração
  de D4 e D15 nos artefatos reais. Detalhamento da E2 iniciado.

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

**Detalhamento** (iniciado em 2026-10-16, ao fechar a E1; decisões respondidas; aprovado
pelo usuário em 2026-10-16).

#### E2.0 — Fontes de dados que o app consome (nada é recalculado pelo app)

| Artefato | Origem | Uso no app |
|---|---|---|
| `diagnostics/{findings.csv, summary.json, summary.md}` por run; `diagnostics/{runs.csv, findings.csv, summary.json}` consolidado em estudos e em `runs/baseline` | E1 (`training_diagnostics`) | selo de status por run (`ok`/`warning`/`critical`), lista de achados filtrável por check/severidade/dataset/cliente, estado `n/a` com o motivo |
| `fold_*/client_*/training_history.csv` (longo) | treino federado/standalone | curvas por cliente; treino = `local_step`/`local_epoch`, val = `post_aggregation_round` (nunca `post_local_round`, A8); `grad_norm` (E1.2) |
| `history.csv` (largo) | baseline (B0) | curvas por época, convertidas pelo mesmo leitor da E1 (`read_baseline_series`) |
| `fold_*/aggregation_history.json` | servidor | pesos por cliente e por rodada, `train_*`/`val_*` por cliente, cosseno entre deltas, `negative_transfer` |
| `conflict_{summary,m1_pairs,m2_lookahead}.csv` | E0.3 | painel de conflito: M1 por classe de par/bloco ao longo das rodadas, matriz Z de M2 (train/val), sempre com a ressalva do Δ inteiro (E0.3.9) |
| `negative_transfer_{rounds,summary}.csv` | `negative_transfer.py` | cancelamento vs referência ortogonal por rodada |
| `{setup}_test_results.csv`, `{setup}_cls_predictions.csv`, `baseline_*` | avaliação final | métricas finais lado a lado e matrizes de confusão; comparação com o baseline só quando o master é o mesmo (regra da D12) |
| `run_report.html`, `negative_transfer.html`, `runs/baseline/report.html`, `training_curves/dashboard.html` | relatórios estáticos existentes | links diretos (o app não os reimplementa) |
| `runs/studies/<id>/run_index.csv` + `runs/seed_*/<arm>` | `study_runner` | árvore estudo → braço; braços fora do índice também aparecem (como na descoberta da E1) |

**Leitor:** reaproveitar `discover_runs`, `run_kind`, `read_client_series`, `read_baseline_series` e
`read_log_series` de `src/experiments/training_diagnostics.py`, que já toleram as três vintages
(federado novo, estudo antigo sem histórico, baseline). O app não duplica regras de leitura.

#### E2.1 — Decisões (respondidas pelo usuário em 2026-10-16)

1. **Stack:** servidor stdlib (`http.server.ThreadingHTTPServer`) + página estática HTML/JS. Nenhuma
   dependência Python nova; `numpy<2`/`protobuf>=4` intocados. Gráficos com Plotly.js carregado de
   CDN com versão fixa (o app avisa na página se o CDN não carregar).
2. **Ao vivo:** polling da página (padrão 10 s, configurável) relendo os artefatos pelo `mtime`. O
   diagnóstico exibido é o salvo em `diagnostics/`; um botão o recalcula **sob demanda** em memória,
   sem escrever na pasta do run.
3. **Memória RSS por ator Ray:** fora da E2 (o app é só leitura e não muda o treino). O painel de
   recursos mostra `gpu_telemetry.csv` e `runtime_events.csv` quando existem. Pendência mantida.

#### E2.2 — Fatos levantados na auditoria (2026-10-16)

- `discover_runs` só reconhece um run com `{kind}_test_results.csv`, portanto **não vê run em
  andamento**. O catálogo do app estende a descoberta sem alterar a função da E1.
- Durante o treino federado, o que é atualizado por rodada:
  - `fold_*/client_*/training_history.csv` (reescrito via `.csv.tmp` + replace);
  - a linha por rodada do `execution.log` (lida por `read_log_series`);
  - `gpu_telemetry.csv`.
- `fold_*/aggregation_history.json` só é gravado **ao fim do fold** (`training_federated.py:462`),
  então pesos/cosseno/cancelamento aparecem apenas para folds concluídos.
- Baseline: `history.csv` é reescrito a cada época (`training_baseline.py:214`).
- Estudos: os braços vêm de `run_index.csv` + `runs/seed_*/<arm>`; o `execution_plan.csv` lista
  também braços ainda não iniciados. Pastas `*.incomplete_*` são tentativas preservadas.
- E1 ainda não está na `main`: a branch `feat/monitor-app` parte de `feat/training-validation-v2`.

#### E2.3 — Arquitetura (`src/monitor/`)

| Arquivo | Conteúdo |
|---|---|
| `catalog.py` | Descoberta e status. Fontes: runs avulsos em `runs/`, `runs/validation/`, `runs/baseline/<modo>/<ds>_<ts>` e estudos `runs/studies/<id>`. Cada run tem um status: `complete` (há `*_test_results.csv`), `running` (sem resultados e `execution.log`/históricos modificados há < `--stale-minutes`, padrão 15), `stale` (sem resultados e parado) ou `pending` (braço do `execution_plan.csv` sem pasta). Cada entrada traz: `kind`, setup, datasets, `local_training.mode`, caminho do master (`_master_path`), folds, selo do diagnóstico salvo e relatórios estáticos existentes. Reusa `run_kind`/`discover_runs`. |
| `readers.py` | Payloads JSON, todos tolerantes a ausência (`{"available": false, "reason": ...}`): `curves` (por cliente via `read_client_series`; baseline via `read_baseline_series`; fallback `read_log_series`; treino = `local_step`/`local_epoch` por rodada, val = `post_aggregation_round`, A8), `final_metrics` (`*_test_results.csv`), `aggregation` (pesos por cliente/rodada e `negative_transfer` de `aggregation_history.json`/`negative_transfer_rounds.csv`), `conflict` (`conflict_*.csv`, com a ressalva E0.3.9), `diagnostics` (salvo) e `diagnose_now` (`diagnose()` em memória), `resources` (`gpu_telemetry.csv`, `runtime_events.csv`). Cache por `(caminho, mtime, tamanho)`. |
| `compare.py` | Comparação de N runs: métricas finais lado a lado por `dataset/task` (média ± dp entre clientes × folds, com delta contra o run de referência) e curvas médias por dataset/tarefa. **Avisos de comparabilidade:** `local_training.mode` diferente (regra do AGENTS.md), master diferente (regra da D12), topologia diferente (comparação só agregada, não pareada por cliente). Nenhum p-valor nem a palavra "significativo". |
| `server.py` + `__main__.py` | `python -m src.monitor [--runs runs] [--host 127.0.0.1] [--port 8765] [--refresh 10]`. Rotas `GET /api/runs`, `/api/run`, `/api/curves`, `/api/final`, `/api/aggregation`, `/api/conflict`, `/api/resources`, `/api/diagnostics`, `/api/compare?paths=...`; `POST /api/diagnose`; `GET /files/<rel>` serve os HTMLs estáticos existentes. Escuta só em `127.0.0.1` por padrão; todo caminho é resolvido e precisa ficar dentro de `--runs` (anti path traversal); somente leitura. |
| `static/{index.html, app.js, style.css}` | Três vistas: **Runs** (árvore estudo → seed → braço + avulsos + baselines, com selo de status/diagnóstico e filtro); **Run** (curvas por dataset → cliente com seletor de métrica e fold, painéis de agregação/cancelamento, conflito, recursos, achados filtráveis por check/severidade/dataset/cliente e links aos relatórios estáticos); **Comparar** (≥ 2 runs marcados na lista). Polling liga sozinho em runs `running`. Tema claro/escuro e paleta de `plot_style.py`. |

#### E2.4 — Testes (`tests/test_monitor.py`, fixtures sintéticas em `tempfile`)

1. Catálogo: um run completo, um em andamento (sem resultados, log recente), um parado, um baseline,
   e um estudo com braço no índice, braço fora do índice e braço só no `execution_plan.csv`. Os
   status esperados são `complete`/`running`/`stale`/`complete`/`pending`.
2. Curvas: a validação vem de `post_aggregation_round` e ignora `post_local_round`. Um histórico
   antigo, sem `training_history.csv`, cai no fallback do `execution.log`.
3. Tolerância: `aggregation_history.json` sem `negative_transfer` e run sem `conflict_*` devolvem
   `available: false` com o motivo, sem exceção.
4. Comparação: dois runs com `local_training.mode` diferentes geram o aviso de budget; masters
   diferentes geram o aviso de master.
5. Servidor: sobe numa porta efêmera; `GET /api/runs` devolve JSON. `GET /files/../config.yaml` e
   caminhos fora de `--runs` dão 403/404. `POST /api/diagnose` não cria nem altera arquivos no run.
6. Cache: reler um arquivo sem mudança de `mtime` não chama o leitor de novo.

#### E2.5 — Validação e entrega

1. `python -m unittest discover -v` (os 197 da E1 mais os novos) e
   `python -m scripts.sync_agent_assets --check`.
2. Manual nos artefatos reais: a lista mostra os 3 runs avulsos, o estudo `multi_dataset_balance_v1`
   (10 braços), os baselines e os smokes da E1. Comparar `primary` × `local_only`.
3. Ao vivo: rodar `python -m scripts.smoke_federated --setup both` com o app aberto. O run aparece
   como `running`, as curvas crescem a cada rodada, e o status vira `complete` ao fim.
4. Documentação: comando em `AGENTS.md` ("Commands") e um parágrafo em `docs/`; o sync é verificado.
5. Commits: (a) `catalog`/`readers`/`compare` + testes; (b) servidor + front-end + teste do servidor;
   (c) docs + registro no `plano.md`.


**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-16 — detalhamento iniciado ao fechar a E1 (fontes de dados, leitor reaproveitado da E1,
  decisões em aberto).
- 2026-10-16 — decisões E2.1 respondidas (stdlib + HTML/JS, polling + diagnóstico sob demanda,
  RSS fora). Auditoria E2.2, arquitetura E2.3, testes E2.4 e validação E2.5 escritas. Branch
  `feat/monitor-app` criada a partir de `feat/training-validation-v2`.
- 2026-10-16 — **E2 concluída.** `src/monitor/` (`catalog`, `readers`, `compare`, `server`,
  `static/`) e `tests/test_monitor.py` (14 testes; suíte 211 OK; sync OK).
  - Nos artefatos reais, o catálogo lista 79 entradas em 2.4 s: 3 runs avulsos, 6 estudos (com os
    10 braços do `multi_dataset_balance_v1` e 25 braços planejados como `pending`), 12 baselines e
    os smokes.
  - Comparar `primary` × `local_steps_ce` × `ablation_budget` mostra o aviso de budget
    (`epochs` vs `steps`).
  - Ao vivo, com `smoke_federated --setup both` e o app aberto, os dois braços passaram de
    `running` a `complete` sozinhos, com as curvas crescendo por rodada (0 → 37 → 55 → 73 → 110
    pontos). O diagnóstico em memória não grava nada no run. Smokes movidos para
    `runs/validation/e2_smoke/`.

  **Divergências em relação ao plano:**
  - O payload de curvas de um run de 200 rodadas tem cerca de 1.3 MB; o servidor comprime com gzip.
  - A telemetria de GPU é resumida em ≈ 600 médias por bin, em vez de uma amostragem com passo
    fixo, porque a série de 1 Hz é ilegível.
  - As mensagens dos leitores (`reason`) ficaram em inglês; a página e os avisos de comparação
    estão em português.
- 2026-10-16 — ajustes pedidos pelo usuário após o primeiro uso (suíte 216 OK; 19 testes do
  monitor):
  - O botão de câmera baixa o gráfico sempre no tema claro, em 2× o tamanho na tela.
  - Card de destaques por dataset: Dice positivo e balanced accuracy de teste, com sparkline da
    validação. Em run sem resultados, mostra o último valor de validação.
  - Identificação do treino: selos `FED`/`LOCAL`/`BASE` e `MTL`/`ST`/`MTL+ST`/`SEG`/`CLS`, nome
    legível, tabela de composição por dataset (tarefas, topologia, clientes por papel) e blocos
    separados de Segmentação e Classificação para métricas e curvas. Também uma legenda
    ("? Legenda"). O selo vem das pastas de cliente do primeiro fold, ou de
    `federated.n_clients` antes do primeiro fold.
  - Paleta do projeto como padrão de figuras, relatórios e monitor: skill `color-palette` (preto,
    vinho, bordô, rosé e creme), validada para daltonismo e contraste nos dois temas. Ela substituiu
    a primeira versão inspirada na Amazon.
  - Esclarecido: os baselines têm 1 fold (holdout 70/30). O "n" nos cards e tabelas é o número de
    fatias de teste dos clientes, não de folds.

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

**Detalhamento.**

#### E5.0 — Fontes já existentes (auditoria de 2026-10-16)

| Onde | O que tem | Estado |
|---|---|---|
| `papers/ciarp2026/refs.bib` | 18 entradas (FedAvg, FedProx, FedPer, FedRep, nnU-Net, U-Net, IN, V-Net, Focal, BUSI, Wilcoxon, Flower…) | Bib do artigo; faltam MOCHA, FedHCA², FedBone, ConDistFL, UFPS, Hsu 2019 e os papers dos datasets ISIC/TCGA/SIIM |
| `docs/GRADIENT_CONFLICT_METHODS.md` | M1–M6 com status, verificados em 2026-10-09 (arXiv/ar5iv) e BibTeX | **Já cumpre o padrão da E5**; só é referenciado |
| `docs/RESEARCH_NOTES.md` | Narrativa do estudo BUSI-only | **Desatualizado**: diz `encoder1..5` compartilhado, `normal` excluído da seg e peso `num_examples × task_weight` (hoje: stem pessoal, máscaras vazias válidas, peso hierárquico por massa) |
| `data/*/PAPER_NOTES.md` | Fatos medidos e papers de cada dataset | Fonte dos datasets; conferir citação primária |
| `docs/DICE_BCE_SUPERVISION.md` | Contrato da loss e da supervisão parcial | **Sem citação** de fonte |
| `referencies/To-do.md` | Busca pendente: como a literatura reporta cancelamento/transferência negativa | Aberto (7 itens) |
| Memória do projeto (topologia HC-FMTL) | MOCHA, FedHCA², FedBone, ConDistFL/UFPS, verificados em 2026-08-19 | Fora do repo; trazer para o registro |

O código quase não cita fontes (só docstrings dos modelos legados e menções a FedPer/FedAvg/FedBone/PCGrad).

#### E5.1 — Inventário de decisões (linhas do registro)

Cada linha: **decisão → local no código (`arquivo::símbolo`) → fonte primária → status → observação**.

| # | Decisão | Local | Fonte candidata | Suspeita a verificar |
|---|---|---|---|---|
| D01 | FedAvg ponderado por amostras (modo `flat`) | `src/federated/server.py::_aggregate_flat` | McMahan 2017 | — |
| D02 | FedPer: trunk compartilhado + cabeças pessoais | `src/federated/model_split.py::shared_keys` | Arivazhagan 2019; Collins 2021 (FedRep) | FedPer personaliza só as camadas finais; aqui decoder inteiro **e stem** são pessoais → adaptação |
| D03 | Stem `encoder1` pessoal por modalidade | idem, `share_stem` | — | Provável decisão própria (aproximar de FedBN/Li 2021?) |
| D04 | Agregação hierárquica + `dataset_weights` | `server.py::_aggregate_hierarchical` | — | Provável decisão própria; buscar precedente |
| D05 | Peso de cliente por massa de supervisão por tarefa | `server.py::_task_factor` | — | Decisão própria |
| D06 | Partição Dirichlet(α) por classe | `src/dataset/federated_partition.py::dirichlet_partition` | Hsu et al. 2019 (arXiv:1909.06335) | Conferir se a Dirichlet é por classe sobre clientes, como em Hsu |
| D07 | Split agrupado por `lesion_id` / paciente | `federated_partition.py::_grouped_dirichlet_partition`, `fold_strategy: stratified_group` | Tschandl 2018 (HAM10000); Buda 2019 | — |
| D08 | DiceBCE 0.5/0.5 com logits | `src/utils/segmentation_loss.py::DiceBCELoss` | Milletari 2016; Isensee 2021 (nnU-Net usa Dice+CE) | Pesos e smoothing |
| D09 | Focal loss (multiclasse) | `src/utils/criterions.py::FocalLoss` | Lin 2017 | γ, α, versão multiclasse por one-hot |
| D10 | Deep supervision `1/(n+1)` | `src/utils/supervision.py::_seg_loss` | Isensee 2021 | **nnU-Net usa `1/2^i` normalizado e descarta a menor escala**: provável divergência; conferir também a ordem de `reversed(outputs)` |
| D11 | Mascaramento de supervisão parcial (`has_mask`/`has_label`; termo ausente ≠ zero) | `supervision.py::_combined_loss` | ConDistFL (arXiv:2308.04070); UFPS (Patterns 2024) | Simplificação: cabeças disjuntas, sem destilação |
| D12 | λ fixos entre tarefas | `supervision.py::resolve_task_lambdas` | Caruana 1997; Kendall 2018 como alternativa | Decisão própria (ablação na E3) |
| D13 | Topologia HC-FMTL (`multi_task`) | `federated_partition.py::_build_multitask_dataset` | FedHCA² (CVPR 2024); Smith 2017 (MOCHA) para desambiguar | Trazer da memória |
| D14 | Conflito: cosseno entre deltas de clientes | `src/federated/conflict_diagnostics.py` | PCGrad; FedBone | Já em `GRADIENT_CONFLICT_METHODS.md` |
| D15 | Razão de cancelamento (intra/inter, referência ortogonal) | `src/federated/negative_transfer.py::cancellation_from_gram` | Yin 2018 (gradient diversity) | Fechar o `To-do.md` aqui |
| D16 | Class weighting `balanced_fold` | `src/dataset/federated_dataloader.py::resolve_class_weights` | King & Zeng 2001 / sklearn `balanced` | — |
| D17 | Oversampling determinístico; proibido em `multi_task` | `src/dataset/BUSI_dataloader.py::deterministic_oversampling` | Aumente-Maestro 2025 (herdado) | — |
| D18 | Wilcoxon pareado exploratório; holdout descritivo | `src/experiments/analyze.py::_paired_inference` | Wilcoxon 1945; Demšar 2006 / Dietterich 1998 (não independência) | — |
| D19 | Métricas: Dice positivo, FP rate em vazios, bal-acc, macro-F1, AUC OvR | `src/federated/unified_eval.py` | Maier-Hein 2024 (Metrics Reloaded) | — |
| D20 | InstanceNorm (sem BN em FL) | `src/models/multitask/MTnnUNet.py` | Ulyanov 2016; Hsieh 2020 (BN em non-IID) | — |
| D21–D24 | Pré-processamento por dataset (BUSI `INTER_NEAREST` congelado; ISIC/TCGA/SIIM `INTER_AREA`; máscaras nearest; classe derivada da máscara em TCGA/SIIM) | `src/dataset/*_preprocessing.py` | Papers dos datasets (`PAPER_NOTES.md`) | BUSI nearest é **decisão herdada**, documentada como tal, nunca "corrigida" |
| D25 | Orçamento local `steps` vs `epochs` | `src/federated/local_trainer.py` | McMahan 2017 (E, B); Wang 2020 (FedNova, heterogeneidade de passos) | — |

A lista final pode crescer durante a verificação; cada linha nova entra com o mesmo formato.

#### E5.2 — Método de verificação

- **Status:** `verificado` (fórmula ou afirmação lida no texto primário), `divergente` (código difere
  da fonte), `adaptação` (difere de propósito, e o motivo está escrito), `decisão própria` (sem fonte;
  justificativa escrita) e `sem fonte` (pendência).
- Leitura do texto primário (arXiv/ar5iv, página do publisher ou PDF em `referencies/`), como na E0.
  Venue e ano são conferidos à parte; venue não confirmada fica anotada.
- Cada linha registra a data de verificação e o trecho da fonte (seção ou equação), sem cópia longa.

#### E5.3 — Entregáveis

1. `docs/DECISION_SOURCES.md`, em português como `GRADIENT_CONFLICT_METHODS.md`: tabela D01–D25,
   seção de divergências e BibTeX das fontes novas.
2. **Divergências:** cada uma com proposta de encaminhamento. A E5 **não altera código de treino nem
   de avaliação**, porque isso mudaria resultados já produzidos. Correções de código viram braço ou
   tarefa da E3/E4, ou PR próprio. Correções só de documentação (ex.: `RESEARCH_NOTES.md`
   desatualizado) entram na própria E5.
3. Atualização da bibliografia (destino em E5.5).
4. `referencies/To-do.md`: itens marcados, com a resposta registrada em D15.

#### E5.4 — Testes

- Nenhum teste novo: a E5 é só documentação (decisão do usuário em 2026-10-16). Os locais
  `arquivo::símbolo` são conferidos à mão ao fechar a etapa.
- Suíte completa (`python -m unittest discover -v`) e `python -m scripts.sync_agent_assets --check`,
  para confirmar que nada fora de `docs/` mudou de comportamento.

#### E5.5 — Decisões em aberto (para o usuário)

1. Destino do BibTeX novo → **`docs/references.bib`, do projeto** (resposta de 2026-10-16). O
   `refs.bib` do CIARP não é tocado.
2. Escopo → **D01–D25 completo** (resposta de 2026-10-16).
3. Teste automático do registro → **não**; E5 fica só em documentação (resposta de 2026-10-16).

#### E5.6 — Commits

(a) registro D01–D13 e fontes de FL e loss; (b) D14–D25 e To-do de cancelamento;
(c) correções de documentação, bibliografia e registro no `plano.md`.

**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-16 — detalhamento escrito (auditoria E5.0, inventário de 25 decisões, método, entregáveis
  e testes). Branch `docs/sources` criada a partir de `feat/monitor-app`.
- 2026-10-16 — decisões E5.5 respondidas: `docs/references.bib` do projeto, escopo D01–D25 e
  nenhum teste novo. Status 🟦.
- 2026-10-16 — **E5 concluída.** `docs/DECISION_SOURCES.md` (26 linhas: D01–D25 mais D06b) e
  `docs/references.bib` (33 entradas, com venue e DOI conferidos no Crossref e nos proceedings).
  Status final: 10 verificadas, 5 adaptações, 6 decisões próprias, 2 divergentes, 2 sem fonte e 1
  adaptação congelada (BUSI `INTER_NEAREST`). Todos os `arquivo::símbolo` do registro foram
  conferidos.
  - **Divergências com encaminhamento** (nenhum código de treino ou avaliação alterado):
    - Div-1, deep supervision: `1/(n+1)` nas 4 saídas, sem normalizar, contra `1/2^i` normalizado
      e sem as 2 menores no nnU-Net. Vira braço `ds_nnunet` da E3.
    - Div-2, Focal com pesos de classe: `p_t` sai da CE ponderada. Corrigir num PR antes de
      qualquer braço Focal da E3. Os resultados federados e o B0 usam CE e não são afetados.
    - Div-3, citação da Dirichlet: o código é Yurochkin 2019 / NIID-Bench (por classe sobre os
      clientes), **não** Hsu 2019 (por cliente sobre as classes).
    - Div-4 (origem do oversampling) e Div-5 (Metrics Reloaded): leituras pendentes.
  - **Achados para a E3:** pelo FedNova, o modo `epochs` (τ_i ∝ n_i) cai no problema de
    inconsistência de objetivo; o modo `steps` não. O FedBone foi publicado (JCST 2024), ao
    contrário do que a nota de 2026-08-19 registrava. O FedBone não reporta métrica de conflito
    por rodada, então a curva por rodada é contribuição nossa.
  - **Documentação:** nota histórica no topo de `RESEARCH_NOTES.md` (o texto dos runs antigos foi
    mantido); critério de classificação por dataset explicitado no `AGENTS.md`; 4 itens do
    `referencies/To-do.md` fechados. GradNorm/CAGrad/Nash-MTL ficam para a E3.

  **Divergência em relação ao plano:** o item 4 dos entregáveis (atualizar `refs.bib`) foi
  substituído por `docs/references.bib`, por decisão do usuário; o `refs.bib` do CIARP não foi
  tocado.

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

**Entradas vindas da E0** (2026-10-15):
- **Colapso do SIIM seg com `personalized_init_seed: dataset`.** O Dice positivo de teste cai para
  0.000 nos 4 clientes; a validação vai a 0 a partir de r100 e não volta. Com seed por cliente,
  fica em 0.066. Hipóteses a testar nos braços:
  - λ_seg vs λ_cls (hoje 0.8/0.2);
  - DiceBCE em uma lesão pequena (classe positiva rara no pixel);
  - interação com o alinhamento intra-dataset (4 clientes idênticos de início).

  Incluir o SIIM como dataset-alvo explícito da ablação da loss.
- **Replicação com seeds.** O ruído GPU entre runs é de até ~0.05 (BUSI) e ~0.01 (demais) num único
  run. **Todo braço da E3 precisa de ≥ 3 seeds** (o perfil `final` do `study_runner`: 1993/1994/1995)
  antes de qualquer conclusão de efeito. Incluir também o braço `personalized_init_seed: dataset` vs
  `client` nessa replicação.
- Métricas de conflito (M1/M2 da E0) ligadas em todos os braços, para ler o efeito da loss no
  conflito e não só no desempenho.

**Riscos/dependências.** O número de braços cresce rápido, então é preciso priorizar. Métodos
adaptativos interagem com a agregação (FedBone), o que exige cuidado na interpretação.

**Detalhamento.**

#### E3.0 — Fatos que moldam a matriz (levantados em 2026-10-16)

- **Custo:** um run federado de 4 datasets com 200 rodadas leva de 1h20 (29/09) a 2h15 (14/10, com
  M1/M2 e dois runs simultâneos). Um baseline seg dos 4 datasets leva ~3 h no pior caso.
- **A seg do SIIM é aprendida e depois desaprendida** (releitura do `history.csv` da B0). O Dice
  positivo de validação chega a **0.237 na época 14** e vai a 0.003–0.007 no fim, enquanto a val
  loss continua caindo (checkpoint escolhido: época 199). Nos outros 3 datasets, pico, checkpoint e
  última época ficam próximos (BUSI 0.660/0.637/0.627; ISIC 0.895/0.894/0.892; TCGA
  0.777/0.747/0.736). É um **desalinhamento entre loss e métrica**: com ~78% de imagens vazias,
  minimizar o DiceBCE (Dice por amostra, smoothing 1) leva a prever "nada". Logo, é assunto da E3,
  não só da E4. Correção registrada na B0.
- **A falha da cls é específica da federação:** o baseline centralizado usa a mesma loss (CE +
  `balanced_fold`) e aprende (SIIM 0.77, ISIC 0.48–0.55), enquanto no federado a CE do SIIM fica em
  ln 2 (A1) e as predições derivam para a classe minoritária (D13).
- **Acoplamento:** `aggregation.task_weights` (seg 4 : cls 1) define o λ local dos clientes
  `multi_task` (`src/federated/client.py::_task_lambdas`) **e** o peso de agregação dentro do ISIC,
  que é `single_task` (seg 80% e cls 20% da massa do dataset, em `server.py::_task_factor`).
- **Conflito intra-cliente ≈ 0** na E0 (cos M1 entre tarefas), por isso PCGrad/CAGrad locais ficam
  de fora.

#### E3.1 — Decisões (grill com o usuário em 2026-10-16)

1. **Duas fases.** Fase A: triagem da loss de seg no baseline centralizado (1 seed, barata). Fase B:
   só a vencedora vai para a matriz federada com 3 seeds.
2. **Seg na Fase A:** Batch Dice, DiceBCE 0.8/0.2 e FocalDICE, contra o controle DiceBCE 0.5/0.5
   (os baselines seg da B0, reaproveitados). A deep supervision nnU-Net (Div-1) **fica fora da E3**
   e continua aberta em `docs/DECISION_SOURCES.md`.
3. **Cls:** sem Fase A, CE fixa. A Focal fica fora (colapsou para NV no estudo antigo e tem a
   Div-2); o PR da Div-2 só é feito se ela voltar.
4. **λ local desacoplado do peso de agregação** por uma chave nova (E3.4 b).
5. **Braços da Fase B:** controle (com par local-only), λ local 1:1, agregação por tarefa 1:1,
   Kendall e `personalized_init_seed: dataset`. Só o controle tem par local-only.
6. **Critério da Fase A:** Dice positivo de **validação**, nunca de teste (E3.2).
7. **Gate do SIIM:** Dice positivo de validação ≥ 0.10 no checkpoint escolhido **e** na última
   época.
8. **Conflito:** M1 + M2 ligados em todos os runs da Fase B (~+25% de tempo).
9. **Orçamento:** modo `steps` (10 por rodada, 200 rodadas) em todos os braços. Pelo FedNova, o
   modo `epochs` (τ_i ∝ n_i) introduziria inconsistência de objetivo (D25).

#### E3.2 — Fase A: triagem da loss de segmentação (baseline centralizado)

| Variante | Diferença em relação ao controle | Código novo |
|---|---|---|
| `control` | — (B0: `runs/baseline/segmentation/*_20261015_*`) | — |
| `batch_dice` | `loss.dice_batch: true` | sim (E3.4 a) |
| `dice_heavy` | `loss.dice_weight: 0.8`, `loss.bce_weight: 0.2` | não |
| `focal_dice` | `loss.function: FocalDICE` | não |

- **Configuração:** a do run de 29/09 (a mesma da B0), mais só a diferença da tabela. Os 3 configs
  ficam versionados em `studies/e3_loss/phase_a/<variante>.yaml`, com um cabeçalho que diz de qual
  config derivam e qual chave muda.
- **Métricas lidas** (`history.csv` de cada run): `val_dice_positive_seg` no pico, no checkpoint
  escolhido (`best`) e na última época; `val_empty_fp_image_rate_seg` (prever mais lesão não pode
  custar FP em todas as imagens vazias).
- **Regra de escolha** (só validação):
  1. **Gate do SIIM:** passa se o Dice positivo de validação for ≥ 0.10 no checkpoint escolhido e na
     última época.
  2. Entre as que passam, vence a de maior Dice positivo de validação no SIIM (checkpoint escolhido),
     desde que não piore BUSI, TCGA ou ISIC além do ruído (BUSI ±0.05; demais ±0.01).
  3. Se nenhuma passar, a seg do SIIM vai para a E4 (resolução) e a escolha usa só os outros 3
     datasets, pela mesma regra de não-piora.
  4. Empate ou nenhuma melhora: fica o DiceBCE 0.5/0.5.
- **Handoff de treino (Fase A):**
  1. *O que alterar:* nada no `src/config.yaml`; os configs da tabela.
  2. *Comando* (um por variante):
     `python -m src.training_baseline --config studies/e3_loss/phase_a/<variante>.yaml --mode segmentation --dataset all --out-root runs/e3_loss/phase_a/<variante>`
  3. *Tempo:* ~3 h por variante no pior caso, ~9 h no total.
  4. *Artefatos:* `runs/e3_loss/phase_a/<variante>/segmentation/<dataset>_*/` (`history.csv`,
     `baseline_test_results.csv`).
  5. *Sucesso:* 12 runs com `history.csv` completo; nenhum D1 (valor não finito) no diagnóstico.

#### E3.3 — Fase B: matriz federada

- **Manifest:** `studies/e3_loss_ablation.yaml`, `study_id: e3_loss_ablation`. O protocolo é o do
  `multi_dataset_siim_dice_bce` (4 datasets; BUSI/TCGA/SIIM `multi_task`; ISIC `single_task`;
  hierárquica uniforme; `dataset_weights` 1:1; `steps` 10; 200 rodadas; CV=1; BF16), fixado em
  `config_overrides`, mais a loss de seg vencedora da Fase A.
- **Partição:** um master novo por seed, gerado pelo `study_runner` a partir do `study_id` (é a
  regeneração intencional da E3; todos os braços leem o mesmo master por seed).

| Braço | `setup` | Diferença em relação ao controle | Pergunta |
|---|---|---|---|
| `primary` | federated | — (λ local 4:1, agregação 4:1, seed pessoal `client`) | controle |
| `local_only` | standalone | — | transferência do controle |
| `lambda_1to1` | federated | `loss.task_lambdas: {seg: 0.5, cls: 0.5}` | a cls do SIIM está sub-ponderada? |
| `agg_task_1to1` | federated | `aggregation.task_weights: {seg: 1, cls: 1}`, λ local fixo em 4:1 | a fatia dos clientes cls do ISIC no trunk importa? |
| `kendall` | federated | `loss.task_weighting: uncertainty` | ponderação adaptativa ajuda? |
| `init_seed_dataset` | federated | `federated.personalized_init_seed: dataset` | o colapso da E0 se replica com 3 seeds? |

- **Seeds:** perfil `final` (1993/1994/1995) para todos: 18 runs.
- **Conflito:** `runtime.diagnostics.gradient_conflict` com M1 e M2 ligados em todos (operacional,
  fora do hash).
- **Comparações** no manifest: `primary_vs_local` e cada braço contra o `primary`
  (`effect_lambda_1to1`, `effect_agg_task_1to1`, `effect_kendall`, `effect_init_seed_dataset`).
- **Handoff de treino (Fase B):**
  1. *O que alterar:* nada no `src/config.yaml`; o manifest acima, com a loss da Fase A preenchida.
  2. *Comando:* `python -m src.experiments.study_runner --manifest studies/e3_loss_ablation.yaml --seed-profile final`
     (antes: `--dry-run` e `--smoke`).
  3. *Tempo:* 18 runs × 1h20–2h15 × 1.25 (M1+M2) ≈ **32–50 h** de GPU.
  4. *Artefatos:* `runs/studies/e3_loss_ablation/` (resultados por braço e seed, `analysis/`,
     `conflict_*.csv`, `negative_transfer_*.csv`, `diagnostics/`).
  5. *Sucesso:* 18 runs `complete` no monitor; `analysis/` gerado; nenhum D1.

#### E3.4 — Código (antes da Fase A e da B)

Toda chave nova é opcional. Ausente, o comportamento e o hash de estudo continuam idênticos aos dos
runs existentes, e isso é coberto por teste.

- **(a) Batch Dice.** `DiceBCELoss(..., batch=False)` repassa `batch` ao `DiceLoss` do MONAI;
  `loss.dice_batch` (padrão `false`) chega por `init_criterion_segmentation`. Cobre os caminhos
  federado, baseline e clássico.
- **(b) Desacoplar λ.** `loss.task_lambdas` (opcional); quando ausente, `_task_lambdas` continua
  derivando de `aggregation.task_weights`. Aplica-se a `client.py` e `training_baseline.py`; o
  servidor continua lendo só `aggregation.task_weights`. O valor efetivo de λ já vai para a
  telemetria do cliente (`task_lambdas`).
- **(c) Kendall.** `loss.task_weighting: fixed | uncertainty` (padrão `fixed`). Com `uncertainty`,
  num cliente com ≥ 2 tarefas a loss vira `Σ_t (λ_t · e^{−s_t} · L_t + s_t / 2)`, com `s_t = log σ_t²`
  aprendido e inicializado em 0. Isso segue Kendall et al. 2018, Eq. 7, na forma usual em log, e a
  forma exata é conferida no texto antes de implementar e registrada na E5. Os `s_t` ficam num
  módulo do modelo com prefixo próprio, então pertencem ao estado **pessoal** (nunca agregados) e
  entram no otimizador. Clientes `single_task` ignoram a chave. Tarefa ausente no batch continua
  omitida, e o termo `s_t/2` também.
- **(d) Manifest e configs** da Fase A e da Fase B.

#### E3.5 — Testes

- `DiceBCELoss` com `batch=True` iguala o Dice do MONAI calculado à mão sobre o batch; com o padrão,
  a saída é bit a bit igual à de hoje.
- `loss.task_lambdas` muda o λ local e **não** muda o peso de agregação; sem a chave, os λ continuam
  iguais aos de hoje.
- Kendall: os `s_t` estão no estado pessoal e fora de `shared_keys`; o gradiente chega aos `s_t`;
  tarefa sem amostra supervisionada não contribui; `fixed` reproduz a loss atual.
- Hash: um manifest sem as chaves novas gera o mesmo hash de antes.
- Suíte completa, `sync_agent_assets --check`, smoke CPU de 2 rodadas com o braço `kendall`, e o
  `study_runner --dry-run` do manifest da Fase B.

#### E3.6 — Leitura e decisão

- Por braço: média ± desvio entre as 3 seeds do Dice positivo e da balanced accuracy por dataset,
  e o delta contra o `primary` pareado por seed.
- Um efeito só conta se o delta médio superar o desvio entre seeds e o ruído GPU (BUSI ±0.05;
  demais ±0.01). Descritivo; p-valores, se houver, são exploratórios.
- Para cada braço, ler também o conflito (M1/M2), o cancelamento e o diagnóstico D1–D15 (A1, A2,
  D13), para explicar o mecanismo e não só o número.
- A decisão final (loss de seg, λ e ponderação) é registrada aqui e em `docs/DECISION_SOURCES.md`.
- **Aplicada em 2026-10-18:** tabelas, regra por célula, mecanismo e decisão no **Registro**
  (Fase B). Vence `init_seed_dataset`; a loss fica DiceBCE 0.5/0.5 com Batch Dice, λ 4:1 e
  agregação 4:1.

#### E3.7 — Pendências conhecidas

- Div-1 (deep supervision) fora da E3 por decisão do usuário; segue aberta.
- Div-2 (Focal com pesos) só é corrigida se a Focal voltar.
- O teto de épocas e rodadas (D15) não é tratado aqui: todos os braços usam o mesmo orçamento.

#### E3.8 — Commits

(a) Batch Dice e `loss.task_lambdas`, com testes; (b) Kendall, com testes; (c) configs da Fase A e
manifest da Fase B; (d) registro no `plano.md` depois de cada fase.

**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-16 — detalhamento escrito a partir do grill (E3.0–E3.8). A releitura da B0 mostra que a
  seg do SIIM é aprendida e depois desaprendida (pico de val 0.237 na época 14). Branch
  `exp/loss-ablation` criada a partir da `main`. Status 🟨, aguardando a aprovação do detalhamento.
- 2026-10-16 — detalhamento aprovado; status 🟦. **E3.4 (a) e (b) implementadas:**
  - `DiceBCELoss(batch=...)` e `loss.dice_batch`. Um único ponto de entrada,
    `init_criterion_segmentation_from_config`, agora serve os caminhos federado, baseline e
    clássico (antes eram 4 chamadas repetidas). `dice_batch` fora do DiceBCE é recusado, não
    ignorado.
  - `loss.task_lambdas` via `client_task_lambdas` (cliente e baseline). Sem a chave, o λ continua
    vindo de `aggregation.task_weights`. Com a chave, ela precisa nomear toda tarefa do cliente.
  - As chaves **não** entram no `src/config.yaml`: o hash de estudo cobre o config inteiro, e um
    teste protege isso.
  - `tests/test_e3_loss_options.py`: 13 testes; suíte com 229 OK; sync OK.
  - Smoke CPU `multi_task` com as duas chaves ligadas passou. O cliente BUSI registrou λ 0.5/0.5,
    e os pesos de agregação do ISIC ficaram em 4:1 (seg 4.0, cls 1.0). Smoke movido para
    `runs/validation/e3_smoke/`.
  - Próximo: E3.4 (c), Kendall.
- 2026-10-16 — **E3.4 (c) implementada (Kendall).**
  - `loss.task_weighting: fixed | uncertainty`. A forma da Kendall et al. 2018 (§3.2, Eq. 10) foi
    conferida no texto: usa-se a forma de classificação, `Σ λ_t e^{−s_t} L_t + s_t/2` com
    `s = log σ²`, para as duas tarefas. Registrada como D12b (adaptação) em
    `docs/DECISION_SOURCES.md`.
  - Aplicada em `local_objective`, o ponto único usado pelo treino e pelas sondas de conflito, que
    assim diferenciam o objetivo real. `s_t` fica num `ParameterDict` com prefixo próprio:
    personalizado, nunca agregado, criado antes do otimizador, só em clientes com ≥ 2 tarefas.
    Tarefa ausente no batch é omitida, inclusive o `s_t/2`. A loss de validação continua com λ
    fixo, para comparar entre braços.
  - `s_t` vai para `aggregation_history.json` por rodada como `train_task_log_var_<tarefa>`.
  - **Bug pego pelo smoke e corrigido:** a avaliação final reconstrói o modelo e carrega o
    `state.pt` em modo estrito, o que falhava com as chaves `task_log_vars.*`. Agora
    `attach_saved_task_uncertainty` recria os slots a partir das chaves do próprio estado. Há
    teste de regressão.
  - `tests/test_e3_kendall.py` com 13 testes; suíte com 242 OK. Smoke CPU `multi_task` com
    `uncertainty` passou nos dois setups (o standalone antes do rename da métrica). `s_t` foi
    salvo no estado pessoal dos clientes BUSI/TCGA/SIIM e está ausente nos clientes `single_task`
    do ISIC.
  - Próximo: E3.4 (d), configs da Fase A e manifest da Fase B.
- 2026-10-16 — **E3.4 (d) concluída; a E3 aguarda os treinos (⏳).**
  - Fase A: `studies/e3_loss/phase_a/{batch_dice,dice_heavy,focal_dice}.yaml`, gerados a partir do
    config da B0 (run de 29/09). Cada um muda só a seção `loss`, e o cabeçalho traz a origem e o
    comando. Smoke de 1 época com 16 amostras nos 4 datasets: 12 runs OK, e a chave de cada
    variante chegou ao `config.yaml` do run (`runs/validation/e3_phase_a_smoke/`).
  - Fase B: `studies/e3_loss_ablation.yaml` (`study_id: e3_loss_ablation`, perfil padrão `final`).
    Fixa no controle tudo o que os braços testam: task_weights 4:1, seed pessoal `client`, M1+M2
    em `runtime`. O bloco `loss` tem um marcador para receber a vencedora da Fase A.
    - `--dry-run`: 18 execuções. Cada braço difere do `primary` só na chave pretendida; o
      `agg_task_1to1` fixa o λ local em 0.8/0.2, igual ao derivado do controle.
    - `--smoke --seed-profile operational`: os 6 braços terminaram, e a análise traz as 5
      comparações.
    - O smoke gerou o master real de `seed_1993` em `data/federated_studies/e3_loss_ablation/`.
      A assinatura da partição não usa nenhum campo que o smoke altera, então é o mesmo master
      do run real.
  - **Próximo passo do usuário:** treinar a Fase A (handoff em E3.2, ~9 h) e trazer os
    `history.csv`. Depois: aplicar a regra de escolha, preencher a `loss` do manifest e rodar a
    Fase B (handoff em E3.3, ~32–50 h).
- 2026-10-17 — **Fase A concluída: vence o Batch Dice.**
  - As 3 variantes rodaram em paralelo na GPU (~3 h no total; 12 runs, 0 erros) em
    `runs/e3_loss/phase_a/`. Seleção pela validação; o teste não foi lido.

  Dice positivo de validação (checkpoint escolhido / última época; pico entre parênteses):

  | Dataset | Controle (B0) | `batch_dice` | `dice_heavy` | `focal_dice` |
  |---|---|---|---|---|
  | BUSI | 0.637 / 0.627 (0.660) | 0.635 / 0.618 (0.635) | 0.644 / 0.631 (0.649) | 0.647 / 0.634 (0.647) |
  | ISIC | 0.894 / 0.892 (0.895) | 0.893 / 0.892 (0.893) | 0.895 / 0.892 (0.895) | 0.894 / 0.893 (0.896) |
  | **SIIM** | 0.003 / 0.007 (0.237, ép. 14) | **0.262 / 0.247** (0.310, ép. 150) | 0.000 / 0.000 (0.272, ép. 18) | 0.032 / 0.043 (0.286, ép. 20) |
  | TCGA | 0.747 / 0.736 (0.777) | **0.788** / 0.758 (0.797) | 0.737 / 0.688 (0.764) | 0.754 / 0.745 (0.784) |

  - **Gate do SIIM:** só o `batch_dice` passa. As outras duas aprendem cedo (pico nas épocas
    18–20) e desaprendem, como o controle. Isso confirma que o colapso vinha do Dice por amostra
    em ~78% de imagens vazias, não da resolução.
  - **Não-piora:** BUSI −0.002 (ruído ±0.05) e ISIC −0.001 (±0.01). TCGA **+0.041**.
  - **Custo registrado:** a taxa de FP em imagem vazia sobe no SIIM (0.000 → 0.112) e no TCGA
    (0.018 → 0.079). O 0.000 do controle vinha de não prever nada.
  - **Ressalvas:** 1 seed, baseline centralizado. O efeito federado é medido na Fase B. A seg do
    SIIM deixa de ser candidata à E4 por "não aprendível a 128 px", mas a resolução continua sendo
    um braço válido da E4.
  - **Manifest da Fase B:** `loss.dice_batch: true`.
- 2026-10-18 — **Fase B concluída (18 runs; seeds 1993–1995): vence `init_seed_dataset`.**
  - `runs/studies/e3_loss_ablation/`: 6 braços × 3 seeds, todos `complete` no `run_index.csv`, com
    `analysis/` gerado e nenhum D1. ~30 h de GPU em sequência (1h29–1h46 por run), abaixo da
    estimativa de 32–50 h.
  - Fontes: `analysis/{summary_by_seed,method_comparisons,per_client_method_deltas}.csv` e, por run,
    `conflict_summary.csv`, `negative_transfer_summary.csv` e `diagnostics/`. Holdout único (CV=1):
    o `analyze.py` marca tudo `descriptive_only_single_holdout` e não calcula Wilcoxon.

  Métrica primária por célula: Dice positivo na seg, balanced accuracy na cls. Média ± desvio entre
  as 3 seeds; em cada seed, média dos clientes da célula (BUSI e TCGA: 2; ISIC seg, ISIC cls e
  SIIM: 4).

  | Célula | `primary` | `lambda_1to1` | `agg_task_1to1` | `kendall` | `init_seed_dataset` | `local_only` |
  |---|---|---|---|---|---|---|
  | BUSI seg | 0.611 ± 0.033 | 0.614 ± 0.020 | 0.609 ± 0.025 | 0.615 ± 0.033 | 0.628 ± 0.024 | 0.660 ± 0.016 |
  | BUSI cls | 0.614 ± 0.061 | 0.551 ± 0.019 | 0.581 ± 0.028 | 0.627 ± 0.012 | 0.637 ± 0.002 | 0.692 ± 0.035 |
  | ISIC seg | 0.845 ± 0.009 | 0.843 ± 0.011 | 0.843 ± 0.009 | 0.845 ± 0.009 | 0.849 ± 0.008 | 0.853 ± 0.011 |
  | ISIC cls | 0.361 ± 0.017 | 0.342 ± 0.006 | 0.356 ± 0.011 | 0.342 ± 0.003 | 0.376 ± 0.016 | 0.362 ± 0.025 |
  | SIIM seg | 0.195 ± 0.007 | 0.150 ± 0.010 | 0.192 ± 0.013 | 0.186 ± 0.009 | 0.209 ± 0.011 | 0.219 ± 0.019 |
  | SIIM cls | 0.578 ± 0.031 | 0.640 ± 0.010 | 0.576 ± 0.026 | 0.593 ± 0.024 | 0.606 ± 0.031 | 0.647 ± 0.035 |
  | TCGA seg | 0.683 ± 0.051 | 0.680 ± 0.050 | 0.685 ± 0.052 | 0.685 ± 0.052 | 0.685 ± 0.047 | 0.695 ± 0.060 |
  | TCGA cls | 0.889 ± 0.026 | 0.880 ± 0.030 | 0.895 ± 0.024 | 0.890 ± 0.023 | 0.909 ± 0.014 | 0.909 ± 0.011 |

  Δ contra o `primary`, pareado por seed (média ± desvio do Δ entre seeds). **Negrito**: passa a
  regra da E3.6.

  | Célula | `lambda_1to1` | `agg_task_1to1` | `kendall` | `init_seed_dataset` |
  |---|---|---|---|---|
  | BUSI seg | +0.003 ± 0.020 | −0.002 ± 0.017 | +0.004 ± 0.009 | +0.017 ± 0.009 |
  | BUSI cls | **−0.064 ± 0.050** | −0.033 ± 0.041 | +0.013 ± 0.049 | +0.023 ± 0.059 |
  | ISIC seg | −0.001 ± 0.002 | −0.001 ± 0.001 | 0.000 ± 0.000 | +0.004 ± 0.002 |
  | ISIC cls | **−0.019 ± 0.012** | −0.005 ± 0.010 | −0.019 ± 0.020 | **+0.015 ± 0.009** |
  | SIIM seg | **−0.045 ± 0.008** | −0.003 ± 0.006 | −0.010 ± 0.005 | +0.014 ± 0.017 |
  | SIIM cls | **+0.062 ± 0.024** | −0.003 ± 0.006 | +0.015 ± 0.015 | **+0.028 ± 0.012** |
  | TCGA seg | −0.004 ± 0.008 | +0.002 ± 0.002 | +0.002 ± 0.003 | +0.001 ± 0.026 |
  | TCGA cls | −0.010 ± 0.011 | +0.006 ± 0.003 | +0.001 ± 0.004 | **+0.020 ± 0.012** |
  | Média das 8 células | −0.010 | −0.005 | +0.001 | +0.015 |

  Δ contra o `local_only`, pareado por seed. **Negrito**: o braço federado fica abaixo do local-only
  pela regra da E3.6.

  | Célula | `primary` | `lambda_1to1` | `agg_task_1to1` | `kendall` | `init_seed_dataset` |
  |---|---|---|---|---|---|
  | BUSI seg | −0.049 ± 0.043 | −0.046 ± 0.024 | **−0.051 ± 0.030** | −0.045 ± 0.040 | −0.032 ± 0.036 |
  | BUSI cls | **−0.078 ± 0.050** | **−0.142 ± 0.016** | **−0.111 ± 0.044** | **−0.065 ± 0.029** | **−0.055 ± 0.035** |
  | ISIC seg | −0.009 ± 0.005 | −0.010 ± 0.005 | −0.010 ± 0.005 | −0.009 ± 0.005 | −0.004 ± 0.008 |
  | ISIC cls | −0.001 ± 0.039 | −0.020 ± 0.032 | −0.006 ± 0.030 | −0.020 ± 0.022 | +0.014 ± 0.041 |
  | SIIM seg | **−0.024 ± 0.013** | **−0.069 ± 0.020** | **−0.027 ± 0.012** | **−0.033 ± 0.017** | −0.010 ± 0.028 |
  | SIIM cls | **−0.069 ± 0.016** | −0.007 ± 0.026 | **−0.072 ± 0.014** | **−0.055 ± 0.012** | **−0.041 ± 0.005** |
  | TCGA seg | **−0.011 ± 0.009** | **−0.015 ± 0.011** | −0.009 ± 0.008 | −0.009 ± 0.008 | −0.010 ± 0.029 |
  | TCGA cls | −0.020 ± 0.021 | **−0.029 ± 0.021** | −0.014 ± 0.019 | **−0.019 ± 0.017** | +0.000 ± 0.009 |
  | Média das 8 células | −0.033 | −0.042 | −0.037 | −0.032 | −0.017 |

  - **Regra da E3.6**, célula a célula: o efeito conta se |Δ médio| superar o desvio do Δ pareado
    entre seeds **e** o ruído GPU (BUSI 0.05; demais 0.01). A regra usa os valores sem
    arredondamento: um −0.010 sem negrito ou fica abaixo de 0.01 antes de arredondar (−0.0095 a
    −0.0098) ou tem desvio maior que o Δ.
    - `init_seed_dataset` × `primary`: Δ positivo nas 8 células; passa em 3 (ISIC cls, SIIM cls,
      TCGA cls). Nenhuma célula piora.
    - `lambda_1to1` × `primary`: passa em 4, com sinais opostos. Sobe a cls do SIIM (+0.062) e
      derruba a seg do SIIM (−0.045) e a cls do BUSI (−0.064) e do ISIC (−0.019). A cls do SIIM
      estava sub-ponderada, mas o ganho sai da seg.
    - `agg_task_1to1` e `kendall` × `primary`: nenhuma célula passa.
    - Contra o `local_only`: o `primary` fica abaixo em 4 células (BUSI cls, SIIM seg, SIIM cls,
      TCGA seg); o `init_seed_dataset`, em 2 (BUSI cls −0.055, SIIM cls −0.041). Nenhum braço
      federado supera o local-only em célula alguma.
    - **Sensibilidade.** Lendo "desvio entre seeds" como o maior desvio dos dois braços comparados,
      e não o do Δ pareado, o `init_seed_dataset` passa só no SIIM seg (+0.014) e perde as 3 células
      de cls. O `lambda_1to1` mantém as 4, e o `kendall` passa a piorar o ISIC cls (−0.019). A
      ordem dos braços não muda.
  - **Média das 8 células (definição).** Média simples dos Δ das 8 células, com peso igual. Em cada
    célula, a média dos clientes por seed e depois a das 3 seeds. Lacuna para o local-only:
    `primary` **−0.0325**, `init_seed_dataset` **−0.0173**.
    - Os −0.036 → −0.018 da mensagem do `e66f575` são a mesma média de 8 células, mas com
      **acurácia** na cls (−0.0355 / −0.0176). Não são média por cliente.
    - A média por cliente (24 pares cliente-tarefa × 3 seeds) dá −0.030 / −0.015 com balanced
      accuracy e −0.038 / −0.017 com acurácia.
    - Vale o número com a métrica primária: **−0.033 → −0.017**.
  - **Métricas secundárias (cls).** Contra o `primary`, o `init_seed_dataset` sobe nas 4 células o
    macro-F1 (+0.014 a +0.046) e a AUC (+0.013 a +0.024). O `lambda_1to1` só sobe no SIIM (macro-F1
    +0.105, AUC +0.045) e cai nas outras 3.

  **Mecanismo** (só o que os arquivos sustentam; média ± desvio entre seeds):
  - **M1** (cosseno dos gradientes do trunk no θ recebido, média das 21 rodadas amostradas), pares
    intra-dataset com as mesmas tarefas: `init_seed_dataset` **0.098 ± 0.006**, demais braços
    0.002–0.003. Normalizado pelo teto intra-cliente: 0.33 ± 0.03 contra −0.001 a 0.011.
    - Por dataset (`init_seed_dataset`): BUSI 0.076, ISIC 0.046, SIIM 0.085, TCGA 0.151. Nos demais
      braços, |cos| ≤ 0.014.
    - Inter-dataset e intra com tarefas diferentes: |cos| ≤ 0.005 em todos os braços.
  - **Cancelamento** (`overall` − referência ortogonal, média das 200 rodadas): `init_seed_dataset`
    −0.017 (primeiro terço −0.047; último −0.001); demais braços −0.0005 a −0.0007 (primeiro terço
    −0.002 a −0.003). O alinhamento é forte no começo e chega perto do ortogonal no último terço,
    como na E0.1b.
  - Os braços de loss (λ, agregação, Kendall) não mexem na geometria: M1 e cancelamento ficam iguais
    aos do `primary`.
  - **Colapso da seg do SIIM** (D6: Dice positivo de validação < 0.05 no último terço). Seeds com
    um cliente colapsado: `primary` 2/3, `lambda_1to1` 3/3, `agg_task_1to1` 2/3, `kendall` 2/3,
    `init_seed_dataset` **0/3**, `local_only` 0/3.
    - É sempre um cliente por run. No `lambda_1to1`, ele nunca aprende (pico de validação de 0.029
      a 0.053).
    - Pior cliente no teste: `primary` 0.113, `lambda_1to1` 0.005, `agg_task_1to1` 0.113,
      `kendall` 0.080, `init_seed_dataset` 0.153, `local_only` 0.157.
    - Com Batch Dice, o colapso da E0 (4 clientes com seed `dataset`) **não se replica**. A resposta
      à pergunta do braço na E3.3 é não. A E0 tinha a mesma política de seed (`dataset`) e a mesma
      seed de treino (1993), mas usava Dice por amostra, outro master de partição e BUSI sem pesos
      de classe. Entre os dois, a loss mudou junto com a partição, então a diferença não se atribui
      só ao Batch Dice. Dentro da E3, a política de seed pessoal é a única diferença entre `primary`
      e `init_seed_dataset`.
  - **Cls do SIIM** (12 pares cliente × seed): D2 (CE de treino no acaso) em 10–12 em todos os
    braços, exceto o `lambda_1to1` (6). D7 (balanced acc. de validação no piso) em 8–10 nos
    federados com λ 4:1, 2 no `local_only` e 0 no `lambda_1to1`.
  - **D13 (ISIC → MEL)** nas 3 seeds de todos os braços federados e em nenhuma do `local_only`. No
    `init_seed_dataset`, a fração predita de MEL é menor (34/34/51% contra 42/42/61% no `primary`;
    prevalência 10–11%).
  - **D15 (teto de rodadas)** disparou nos 18 runs, inclusive nos 3 do `local_only`. Em cada run,
    de 8 a 16 dos 24 pares cliente-tarefa têm o mínimo da val loss e a melhor métrica nos últimos
    10% das rodadas. O orçamento de 200 rodadas limita todos os braços, federados ou não, e por isso
    as próximas etapas (E6, E7) usam 400 rodadas.
    - **Ressalva:** não existe controle de 400 rodadas (`primary` e `local_only`). Comparar um run de
      400 rodadas com a E3 mistura orçamento e intervenção.

  **Decisão.**
  - **Loss de seg:** DiceBCE 0.5/0.5 com Batch Dice (`loss.dice_batch: true`). Fica nos manifests e
    nos configs de run, **não** no `src/config.yaml`: o `StudyHashGuardTests`
    (`tests/test_e3_loss_options.py`) proíbe a chave ali, porque o hash de estudo cobre o config
    inteiro.
  - **Cls:** CE com `balanced_fold`, sem mudança.
  - **λ local:** 4:1, derivado de `aggregation.task_weights` (`loss.task_lambdas` fica ausente).
  - **Agregação por tarefa:** 4:1.
  - **Seed pessoal:** `federated.personalized_init_seed: dataset`, no `src/config.yaml` desde o
    `e66f575`. O fallback do código continua `client`, então todo config sem a chave, inclusive os
    dos runs históricos, resolve como foi produzido.
  - **Rejeitados:**
    - Kendall: nenhuma célula melhora, e M1 e cancelamento ficam iguais aos do controle.
    - λ 1:1: troca seg por cls no SIIM, piora a cls do BUSI e do ISIC e tem a pior média das 8
      células.
    - Agregação 1:1: nenhuma célula muda.
  - Div-1 (deep supervision) e Div-2 (Focal com pesos) seguem abertas.
  - Registro das fontes em `docs/DECISION_SOURCES.md`: notas da E3 em D05, D08, D12 e D12b; D26
    nova.
  - Tudo descritivo: 3 seeds, holdout único, sem p-valores.

---

## E4 — Ablação: pré-processamento de imagens

**Objetivo.** Medir o efeito de escolhas de pré-processamento no desempenho. Na forma decidida em
2026-10-18: entender o que cada dataset precisa e usar **uma resolução por dataset** (TCGA-LGG e
SIIM-ACR a 256 px; BUSI e ISIC a 128 px). Os stems pessoais e o MTnnUNet totalmente convolucional
permitem misturar resoluções no mesmo run.

**Situação atual.**
- **Interpolação:** BUSI usa `INTER_NEAREST` (congelado); ISIC, TCGA-LGG e SIIM-ACR usam `INTER_AREA`.
  Máscaras sempre em nearest.
- **Augmentation:** flip H/V e rotação em `src/dataset/BUSI_dataset.py`.
- **Legado:** CLAHE, SOBEL, brilho e contraste como canais extras.
- **Normalização:** hook existe, mas está desligado (`federated_dataloader.py`, `normalization=None`).

**Entradas vindas da E0** (2026-10-15):
- **SIIM seg (pneumotórax a 128 px).** O colapso da segmentação na política `dataset` e o Dice
  ≈ 0.06 na `client` sugerem que a lesão pode estar sendo apagada pela redução para 128 px. Isso
  exige uma variante de maior resolução, ou um recorte, como braço da E4, com uma nova variante e
  sem tocar em `processed_128`.
- **Replicação com ≥ 3 seeds**, pelo mesmo motivo de ruído entre runs registrado na E3.
  *Substituída em 2026-10-18 por decisão do usuário:* 1 seed (1993), 1 run (E4.1). Multi-seed fica
  pendente (E4.8).

**Restrição central.** Ablações que alteram a imagem gravada exigem uma **nova variante** de dataset,
sem sobrescrever `processed_128` nem regerar a partição congelada. Por isso a preferência vai para
variações *on-the-fly* no dataloader (normalização, CLAHE, augmentation), que preservam a partição.
Antes de criar uma variante gravada, é preciso verificar se a partição pode ser reutilizada.

**Entregáveis (macro).**
1. Matriz de braços: on-the-fly e, se justificado, variante gravada.
2. Implementação configurável, com testes.
3. Manifest, handoff de treino, análise e fontes registradas.

**Treino necessário?** Sim: 1 run federado solto, seed 1993, 400 rodadas, executado pelo usuário
(handoff em E4.6).

**Critério de pronto.** Run de 400 rodadas lido contra a E3, com a decisão sobre a resolução
registrada e as ressalvas explícitas: orçamento (não existe controle de 400 rodadas) e raster do GT
(o Dice a 256 px é medido contra outra máscara; E4.7).

**Detalhamento.**

#### E4.0 — Fatos (levantados em 2026-10-18)

- **Lesão por dataset a 128 px** (`python -m scripts.preprocessing_needs`; tabela completa em
  `docs/PREPROCESSING_NEEDS.md`):

  | Dataset | Bruto | Vazias | Lesão mediana | % do quadro | Lado mín. p5 | < 50 px |
  |---|---|---|---|---|---|---|
  | BUSI | 512×512 | 14,2% | 1.086 px | 6,6% | 11 | 1 |
  | ISIC (seg) | variável (ex. 767×576, 3072×2304) | 0% | 2.697 px | 16,5% | 17 | 1 |
  | TCGA-LGG | 256×256 | 65,1% | 381 px | 2,3% | 8 | 78 |
  | SIIM-ACR | 1024×1024 | 77,9% | 134 px | 0,82% | 5 | 474 |

- **O MTnnUNet só exige H e W divisíveis por 32.** Um forward a 256 px dá as 4 saídas de seg em
  256×256 e o logit de cls; a 240 px falha. O trunk compartilhado é convolucional, então os pesos
  não dependem do tamanho da entrada, e os stems são pessoais.
- **O master copia cada `mapping.csv`.** Nada em `src` lê `data.image_size` no treino; o loader lê
  `img_path`/`mask_path` do master. Mudar a variante de um dataset exige, portanto, um master
  próprio. O `_group_strata` do TCGA usa só `tumor_pixels > 0`, e a ordem dos mappings não depende
  da resolução: o mesmo gerador, sobre os mappings a 256, deve reproduzir a partição da E3 (E4.3).
- **TCGA-LGG a 256 é o próprio dado.** O bruto já é 256×256: em 300 fatias sorteadas, imagem e
  máscara de `processed_256` são idênticas ao TIFF.
- **Normalização de intensidade fica de fora.** Toda conv do MTnnUNet é sem bias e seguida de
  `InstanceNorm2d` sem afim, então uma mudança afim por imagem na entrada é quase anulada.
- **Hardware:** RTX 5070 Ti com 16.303 MiB, ~2,4–2,8 GB ocupados em repouso. O vencedor da E3
  levou 6.068 s para 200 rodadas (rodada sem M1/M2: mediana 25 s; com M1/M2: ~68 s), com pico de
  8.310 MiB na telemetria.

#### E4.1 — Decisões (grill com o usuário em 2026-10-18)

1. **Muda só a resolução.** TCGA-LGG e SIIM-ACR a 256 px; BUSI e ISIC a 128 px, com a
   `processed_128` intocada. Normalização, recorte e augmentation ficam fora.
2. **Um run solto:** `python -m src.training_federated --config …`, seed 1993, 400 rodadas,
   `steps_per_round: 10`, sem `study_runner`, sem gêmeo local-only e sem controle de 400 rodadas.
   A comparação com a E3 fica com o usuário.
3. **Config base:** o do vencedor da E3 (`init_seed_dataset`, seed 1993). O compartilhamento
   continua só no trunk; `shared_modules` é da E6.
4. **Master novo autorizado**, desde que seja o master da E3 seed 1993 lido a 256 px, com
   verificação de equivalência.
5. **"≥ 3 seeds" substituída** por decisão do usuário: 1 seed, 1 run.
6. **512 px descartado:** não cabe com batch 32 em 16 GB.

#### E4.2 — Código

- **C1** `preprocessing_utils.preprocessing_data_config(dataset, argv=None, config_file=…)`.
  - Sem flags: devolve o bloco `data` do `config.yaml` sem mudança (fluxo histórico, com
    `assert_target_dataset`).
  - `--variant processed_<N> --image-size N`, sempre juntos: escolhe o dataset do próprio script,
    tira as classes de `datasets.<DS>.classes`, exige que o nome soletre o tamanho e chama
    `assert_new_variant`, que aborta se a pasta tem `mapping.csv` ou qualquer arquivo. Sem flag
    `--dataset`.
  - Só `TCGA_LGG_preprocessing.py` e `SIIM_ACR_preprocessing.py` ganham `main(argv=None)`; os
    scripts do BUSI e do ISIC ficam intocados.
- **C2** `scripts/partition_equivalence.py`. Exige: mesmas colunas, linhas e ordem; mesmos
  `dataset, fold, client_id, task, split, id`; linhas de BUSI e ISIC idênticas como texto do CSV; em
  TCGA e SIIM, `class`, `id`, `lesion_id` e `official_split` iguais, caminhos iguais depois de trocar
  o prefixo `paths.processed_dir(…processed_128)` pelo de `processed_256`, `dim1 == dim2 == 256`,
  `tumor_pixels > 0` igual linha a linha e arquivos presentes; e o sha256 da referência igual ao do
  seu metadata. Qualquer problema sai com status 1. `--write-metadata STUDY_ID SEED --config …`
  grava `<candidato>_metadata.yaml` no esquema do `_ensure_partition`, mais o caminho do relatório,
  e nunca sobrescreve.
- **C3** `scripts/preprocessing_needs.py`: só leitura, caminhos por `paths.mapping_file`; fração
  vazia, área da lesão (mediana, p5, % do quadro), p5 do lado menor e lesões < `--small-px`.
- **C4** `studies/e4_preprocessing/resolution_256.yaml`, gerado com `safe_load`/`safe_dump` a partir
  do config resolvido da E3. Difere só em: `datasets.{TCGA_LGG,SIIM_ACR}.variant: processed_256`,
  `federated.partition_file` (master da E4), `federated.rounds: 400`, bloco `experiment`
  (`e4_preprocessing` / `resolution_256`) e `runtime.federated.client_resources.num_gpus: 1.0`
  (probe, E4.5; fora do hash).
- Nenhuma chave nova no `src/config.yaml`.

#### E4.3 — Dados e equivalência

- **Variantes** (logs em `runs/validation/e4_preprocessing/{tcga,siim}.log`; nenhum `became empty`):

  | Variante | Linhas | Positivas | Disco | sha256 do `mapping.csv` |
  |---|---|---|---|---|
  | `data/TCGA_LGG/processed_256` | 3.929 | 1.373 | 337 MB | `5a8a93b1…a98024` |
  | `data/SIIM_ACR/processed_256` | 12.044 | 2.667 | 461 MB | `66b647dd…a050e` |

  - O SIIM-ACR repete o tratamento de duplicatas da 128 (3 imagens removidas; o par com máscaras
    diferentes sai inteiro). As `processed_128` continuam com os mesmos sha256 (`cdce37bb…` e
    `68d9a714…`).
- **Controle de determinismo:** o master da E3 regerado a 128 em
  `runs/validation/e4_partition_control/` tem sha256 `389116b5…bd3b`, igual ao original.
- **Master da E4:** `data/federated_studies/e4_preprocessing/seed_1993/federated_mapping.csv`,
  46.555 linhas, sha256 `a3d7d98423b56c4e074f8b460224edc3c1708e64e63fe39a6442868eb74a82d2`. Gerado
  por `python -m src.dataset.federated_partition --config studies/e4_preprocessing/resolution_256.yaml`.
- **Equivalência** (`federated_mapping_equivalence.yaml`, status `ok`): BUSI 900 e ISIC 13.709
  linhas idênticas; TCGA 7.858 e SIIM 24.088 linhas com 0 trocas de classe, id, `lesion_id` ou
  `official_split`, 0 caminhos fora do prefixo esperado, 0 arquivos ausentes e 0 viradas de
  `tumor_pixels > 0`. Positivas: TCGA 1.373 / SIIM 2.667, iguais à referência. O metadata
  (`federated_mapping_metadata.yaml`) traz a assinatura `48ffae03…` e o sha256 do CSV.
- **Loader:** um batch de treino de `TCGA_LGG_mt_0` sai (2, 3, 256, 256) e um de `SIIM_ACR_mt_0`,
  (2, 1, 256, 256); BUSI e ISIC continuam a 128.

#### E4.4 — Testes

- `tests/test_preprocessing_cli.py` (12): flags juntas, nome que soletra o tamanho, classes do
  registro, abort com `mapping.csv` e com pasta não vazia, pasta vazia aceita, fluxo sem flags
  intacto, e um TCGA sintético de ponta a ponta a 256 sem tocar no `config.yaml`.
- `tests/test_partition_equivalence.py` (16): cópia reescalada equivalente; troca de classe, lesão
  sumida, linha do BUSI alterada, ordem/split/cliente trocados, caminho em outra variante, tamanho
  errado, esquema e contagem, dataset desconhecido e arquivos ausentes são problemas; metadata no
  esquema do `study_runner`, gravado uma vez e nunca com problema ou com config errado.
- `tests/test_preprocessing_needs.py` (5) e `tests/test_e4_run_config.py` (3: diff exato contra o
  vencedor da E3 e master igual ao seu metadata; pulam sem `runs/` ou `data/`).

#### E4.5 — Validação

- Suíte completa e `sync_agent_assets --check` (números no **Registro**).
- **Smoke de CPU** (`scripts.smoke_federated --config studies/e4_preprocessing/resolution_256.yaml
  --conflict-diagnostics`): `SMOKE_OK`, 5 clientes (um por dataset/tarefa), M1/M2 nas 2 rodadas.
  Movido para `runs/validation/e4_smoke/`.
- **Probe de memória na GPU** (2 rodadas, só SIIM + TCGA, os 6 clientes a 256 px; GPU sem outros
  processos). Pico = máximo de `memory_used_mb` em `gpu_telemetry.csv`, que inclui os ~2,4 GB de
  repouso:

  | `num_gpus` | Pico (telemetria) | Pico (`nvidia-smi`, 1 s) | Pico reservado por cliente | Rodada 1 | Rodada 2 | OOM |
  |---|---|---|---|---|---|---|
  | 0.5 | **15.588 MiB** | 15.248 MiB | 5.906 MiB | 111 s | 55 s | não |
  | 1.0 | 9.072 MiB | 8.765 MiB | 5.866 MiB | 107 s | 71 s | não |

  - Regra: sem OOM e pico ≤ 14.500 MiB → 0.5. Com 0.5 o pico passou (15.588, na rodada 1, com
    M1/M2; ~14,1–14,4 GB sustentados no resto), então o run usa **1.0**, e o re-probe com 1.0
    passou com folga.
  - As duas rodadas do probe têm M1/M2 (a 1ª e a última sempre têm). Com 1.0, os 6 clientes a 256
    custam só ~1,3× o tempo de 0.5 (71 s contra 55 s na rodada 2): com 0.5, os dois clientes
    simultâneos disputam a mesma GPU.
  - O pico com 0.5 se explica pelos próprios processos do probe: 2 × 5.906 MiB reservados, mais os
    contextos CUDA dos 2 workers e do driver, mais os ~2,4 GB de repouso. Com 1.0 a mesma conta dá
    os 9.072 MiB medidos. Mesmo descontando um contexto alheio (~0,5 GB), 0.5 passaria do limite.
  - **Ressalva de tempo:** um smoke de CPU da E6 (sessão Ray de 22:55:59 a 22:58:25) rodou junto
    com as rodadas 1 e 2 do probe de 0.5. Ele não usa a GPU, mas disputa a CPU, então os tempos com
    0.5 podem estar pessimistas e o ~1,3× entre 1.0 e 0.5 pode estar subestimado. O probe com 1.0
    (23:02:56 em diante) rodou sem sobreposição.
  - Execução: `runs/validation/e4_gpu_probe/` (`config_probe.yaml`, `config_probe_num_gpus_1.yaml`,
    `run/`, `run_num_gpus_1/`). RAM máxima do driver: 2,9 GB.
- **Estimativa de tempo (400 rodadas, `num_gpus` 1.0).** Não medida diretamente: o probe só tem
  rodadas com M1/M2 e só 6 clientes. Supondo que a passagem de 0.5 para 1.0 custe à E3 o mesmo
  ~1,3× medido no probe, as rodadas da E3 ficariam em ~32 s (sem M1/M2) e ~88 s (com). Trocando a
  parcela dos 6 clientes SIIM/TCGA (~45% a 128 px) pela medida a 256: rodada com M1/M2 ≈
  0,55 × 88 + 71 ≈ 120 s; sem M1/M2 ≈ 44–58 s. Com 41 rodadas amostradas e 359 sem amostragem:
  **~6–7 h** (faixa 5,5–8 h), mais ~10 min de teste final e relatórios. A estimativa do plano
  (9–11 h) supunha que 1.0 dobraria o tempo; no probe, foi ~1,3×.

#### E4.6 — Handoff de treino

1. **O que alterar:** nada no `src/config.yaml`. O config é
   `studies/e4_preprocessing/resolution_256.yaml` (versionado).
2. **Pré-voo:**
   - `sha256sum /home/lucas/fed_multi_task_breast_cancer/data/federated_studies/e4_preprocessing/seed_1993/federated_mapping.csv`
     = `a3d7d98423b56c4e074f8b460224edc3c1708e64e63fe39a6442868eb74a82d2`;
   - `nvidia-smi --query-compute-apps=pid --format=csv,noheader` não imprime nada;
   - nenhum outro run de treino ativo. **Este run nunca roda em paralelo com outro** (E6, E7 ou
     qualquer smoke na GPU): com 1.0 ele usa a GPU inteira em sequência.
3. **Comando:**

   ```bash
   cd /home/lucas/fed_e4_preproc && .venv/bin/python -m src.training_federated --config studies/e4_preprocessing/resolution_256.yaml
   ```

4. **Retomada:** `run(cfg, run_path=…, resume=True)` reaproveita só folds completos. Com CV=1, uma
   interrupção recomeça da rodada 1.
5. **Tempo:** ~6–7 h (faixa 5,5–8 h). Conferir no `execution.log`, depois de ~20 rodadas, ~45–60 s
   por rodada sem M1/M2 e ~2 min nas amostradas (1, 11, 21, …, 400).
6. **Disco:** ~3,5–4 GB do run (E3: 2,9 GB a 200 rodadas, com `conflict/` dobrando para ~0,7 GB). Os
   dados a 256 (0,8 GB) já estão gravados.
7. **Artefatos:** `runs/<ts>_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-SIIM_ACR-TCGA_LGG/`
   (`federated_test_results.csv`, `federated_cls_predictions.csv`, `fold_0/`, `gpu_telemetry.csv`,
   `conflict_*.csv`, `negative_transfer_*.csv`, `diagnostics/`, `run_report.html`).
8. **Sucesso:**
   - 24 linhas em `federated_test_results.csv` (BUSI 2×2, ISIC 8, TCGA 2×2, SIIM 4×2);
   - nenhum OOM e o pico de memória registrado em `gpu_telemetry.csv`;
   - 41 diretórios `fold_0/conflict/round_*`, cada um com 16 `*_m1.npz` e 16 `*_m2.csv`
     (`for d in …/fold_0/conflict/round_*; do echo $d $(ls $d/*_m1.npz|wc -l) $(ls $d/*_m2.csv|wc -l); done`);
   - nenhum D1 no diagnóstico.

#### E4.7 — Defeitos registrados, não corrigidos

- **Máscara do BUSI com valor 254.** `benign_id_181_mask.png` tem 1 pixel 254 (além de 1.323 pixels
  255). `BUSI_dataset.py::_load_mask` só mapeia 255 → 1, então esse pixel chega à loss como 254; e
  `tumor_pixels` só conta 255. É a única máscara não binária do BUSI.
- **`transforms.rotation` é ignorado.** `client.py::_default_transforms` aplica sempre
  `RandomRotation(degrees=360)`, qualquer que seja o valor da chave.
- **`assert_variant_resolution` não guarda aspecto nem interpolação**, só o tamanho. Variantes novas
  pela CLI ficam protegidas por `assert_new_variant`.
- **ISIC achatado:** `PRESERVE_ASPECT = False`; Task 1 (aspectos variados) e Task 3 (4:3) são
  esticadas para o quadrado.
- **O Dice a 256 é medido sobre outro raster de GT.** No TCGA e no SIIM, o Dice de teste da E4 usa a
  máscara a 256 px, e o da E3 a de 128 px: alvos diferentes da mesma lesão, sobretudo nas pequenas.
  Comparar os dois mistura modelo e alvo. É ressalva de leitura, não defeito de código.

#### E4.8 — Pendências

- Multi-seed (a E4 tem 1 seed).
- ISIC a 256 px.
- Recorte de pulmão no SIIM-ACR.
- Augmentation anatômica por dataset (a rotação de 360° é implausível em tórax e em RM axial).
- Normalização de intensidade (fora da E4).
- Não existe controle de 400 rodadas a 128 px: o run confunde orçamento e resolução.
- Avaliar o Dice no mesmo raster da E3 (por exemplo, reduzir a predição a 128 px) para isolar o
  efeito do modelo.

**Registro.**
- 2026-10-04 — etapa planejada (macro).
- 2026-10-18 — **detalhamento (E4.0–E4.8) escrito a partir do grill e implementado; a E4 aguarda o
  run (⏳).** Branch `exp/preprocessing-ablation`, worktree `/home/lucas/fed_e4_preproc`.
  - Commits: C1 `2d7618f` (CLI de variante), C2 `62e59a1` (`partition_equivalence`), C3 `16eec5f`
    (`preprocessing_needs`), C4 `bb5b216` (config do run) e `f07cbd9` (teste do config), C5
    `c6e2f54` (docs: `docs/PREPROCESSING_NEEDS.md`, `.rclone-filter` com `data/*/processed_*/**`,
    `AGENTS.md`, skills `dataset-preprocessing`, `drive-sync` e a linha `[ABORT]` nova da
    `error-investigator`, notas da E4 em D22/D23), C6 este registro.
  - Dados: `processed_256` do TCGA-LGG e do SIIM-ACR gerados pela CLI, sem nenhuma lesão sumida.
    Controle de determinismo OK (`389116b5…bd3b`). Master da E4 gerado e equivalente ao da E3:
    46.555 linhas, positivas TCGA 1.373 / SIIM 2.667, sha256 `a3d7d984…82d2` (E4.3).
  - Validação: suíte com 278 testes OK (3 pulados: CUDA indisponível com
    `CUDA_VISIBLE_DEVICES=`); `sync_agent_assets --check` OK; smoke de CPU `SMOKE_OK`
    (`runs/validation/e4_smoke/`); probe de GPU: 0.5 passou de 14.500 MiB (15.588), 1.0 ficou em
    9.072 MiB, então `num_gpus: 1.0` (E4.5).
  - **Achado fora da E4:** `src/monitor/static/index.html` não está versionado (o `*.html` do
    `.gitignore` o ignora), então `tests.test_monitor.ServerTests.test_api_and_page` falha com 404
    num clone ou worktree novo. Nesta worktree, o arquivo foi ligado ao da checkout principal por um
    link simbólico (ignorado pelo git). A correção (versionar o arquivo com uma exceção no
    `.gitignore`) fica pendente.
  - **Próximo passo do usuário:** rodar o handoff da E4.6 (~6–7 h), um run por vez na GPU.

---

## E6 — Compartilhamento ampliado (decoder federado)

**Objetivo.** Medir o efeito de federar também o decoder, e não só o encoder.

**Situação atual.**
- O trunk compartilhado é `encoder2..5` + `bottleneck` (`TRUNK_PREFIXES` em
  `src/federated/model_split.py`): **29.5%** dos parâmetros do MTnnUNet.
- O stem (`encoder1`), o decoder inteiro, as saídas e o classificador são pessoais (D02 e D03 em
  `docs/DECISION_SOURCES.md`).
- O conjunto compartilhado é fixo no código. Não há chave de config para mudá-lo.

**Entregáveis (macro).**
1. Mecanismo configurável de módulos compartilhados (`shared_modules`). Sem a chave, o trunk e o
   hash de estudo continuam os atuais, com teste.
2. Conjunto amplo: federar também `decoder1–5`, `upsample1–5`, `process_encoder_5` e
   `process_decoder_5`, além de `encoder2–5` + `bottleneck`. Passa de 29.5% para **64.1%** dos
   parâmetros (medido para 1 e 3 canais, com 2, 3 e 7 classes).
3. Continuam pessoais: `encoder1`, `output1–4` e `classifier`.
4. Agregação hierárquica inalterada.
5. Handoff de treino, análise (app, diagnóstico D1–D15, M1/M2, cancelamento) e fonte registrada
   (D02 muda de alcance).

**Treino necessário?** Sim: 1 run federado solto, seed 1993, 400 rodadas, executado pelo usuário.

**Critério de pronto.** Run de 400 rodadas lido contra a E3, com a decisão sobre o compartilhamento
registrada e a ressalva de orçamento explícita (não existe controle de 400 rodadas).

**Detalhamento.**

#### E6.0 — Fatos (levantados em 2026-10-18)

- **Tamanhos** (MTnnUNet: 53 chaves, sem buffers). Os tamanhos compartilhados (10/4 663 296 e
  32/10 135 840) são iguais com 1 e 3 canais e com 2, 3 e 7 classes, porque stem, saídas e
  classificador ficam fora do conjunto compartilhado. O total 15 819 799 e as frações abaixo são do
  modelo de referência (BUSI, 1 canal/3 classes); no ISIC (3 canais/7 classes, total 15 821 403) a
  fração ampla é 64.06%.

  | Conjunto | Blocos | Chaves | Valores | Fração |
  |---|---|---|---|---|
  | Trunk (hoje) | 5 | 10 | 4 663 296 | 29.48% |
  | Amplo (E6) | 17 | 32 | 10 135 840 | 64.07% |

- **A regra legada é por prefixo** (`startswith`), não por nome:
  - no Multi_BTSUNet, ela captura também o `bottleneck2` (11/9 chaves com/sem stem, contra 10/8
    por nome exato). Por isso, o caminho legado mantém o `startswith`;
  - o MTUNetPlusPlus não casa nenhuma chave nas duas regras. Federado, ele não compartilhava nada,
    e sem erro.
- **O M1 tinha dois bugs**, e o `observational()` engolia os dois (o smoke passava assim mesmo):
  - as chaves vinham do `state_dict`, então um buffer compartilhado (os FrozenBN da E7) dava
    `KeyError`;
  - com o conjunto amplo, um cliente `single_task` do ISIC tem `grad None` nos blocos que o seu
    objetivo não alcança, e o `.detach()` dava `AttributeError`. Medido: o cliente só-cls deixa
    `decoder1–4` e `upsample1–4` sem gradiente; o só-seg deixa `process_encoder_5` e
    `process_decoder_5`.
- Os logs dos workers do Ray não chegam ao `execution.log` (o run da E3 tem 0 linhas
  `local_training=steps`). A prova de cada cliente são os arquivos que ele grava.
- **Diluição no ISIC.** Dentro do ISIC, os clientes seg têm 80% da massa e os cls 20%
  (`task_weights` 4:1, 4 clientes de cada). O Adam não mexe em parâmetro com `grad None`, então o
  cliente devolve esses blocos iguais ao global (delta zero). A atualização do ISIC sai diluída em
  20% em `decoder1–4`/`upsample1–4` e em 80% em `process_*`.
- **Inicialização.**
  - O trunk inicial da E6 é **idêntico** ao da E3. Conferido: os 10 arrays do trunk saem iguais
    de `_initial_shared_parameters` com os dois configs (mesma seed `__global__`, mesmo modelo de
    referência).
  - O decoder deixa de ser pessoal. Na E3, ele vinha da seed pessoal (por dataset, no vencedor).
    Na E6, ele parte do parâmetro inicial global, igual para os 16 clientes.
  - O estado inicial global depende dos canais de `federated.datasets[0]`: o `encoder1` consome o
    RNG antes dos módulos compartilhados. Com 1 ou 3 canais, `encoder2`, decoder e `process_*`
    saem diferentes; com 2 ou 3 classes, iguais. É comportamento antigo, que agora alcança o
    decoder.
- **Referência de custo** (vencedor da E3, seed 1993, 200 rodadas): ~1h41 de treino (6068 s;
  1h43 de parede no `run_index.csv`) e 2.9 GB em disco. Desses, os clientes ocupam de 119 a 168 MB
  cada, `conflict/` 341 MB (21 rodadas amostradas) e o `aggregation_history.json` 14 MB.

#### E6.1 — Decisões (grill com o usuário em 2026-10-18)

1. **Um run federado solto**, `python -m src.training_federated --config …`, seed 1993. Sem
   `study_runner`, sem gêmeo local-only e sem controle extra. A comparação com os runs anteriores
   fica com o usuário.
2. **Orçamento:** 400 rodadas com `steps_per_round: 10`, o dobro da E3 (o D15 disparou nos 18 runs).
3. **Config base:** `runs/studies/e3_loss_ablation/resolved/seed_1993/init_seed_dataset.yaml`, do
   vencedor da E3. Muda só `rounds`, a intervenção e o bloco `experiment`. O master é o da E3 seed
   1993.
4. **Agregação hierárquica inalterada.** A diluição do ISIC fica como pendência (E6.8).

#### E6.2 — Chave `federated.shared_modules` (C1)

- **Ausente ou null:** `shared_modules(config)` devolve `bool(share_stem)`, o spec legado. O
  `model_split` aplica o `startswith` de sempre, byte a byte.
- **Presente:** devolve uma tupla de nomes de topo, com casamento **exato** do primeiro componente
  da chave (`name.split(".", 1)[0]`): `enc` não captura `enc2`.
- **Validação da lista** em `src/federated/config.py::shared_modules`. Ela roda em toda resolução,
  não só em `validate_federated_config`, então nenhum chamador recebe uma lista inválida. Recusa:
  - lista vazia, valor que não é lista, duplicatas;
  - nome vazio, com espaço nas bordas ou com ponto;
  - `task_log_vars`, que é pessoal (Kendall);
  - `encoder1` incoerente com `share_stem`. Ausente, o `share_stem` vale `true`, então uma lista
    sem `encoder1` exige `share_stem: false` (o config da E3 já tem).

  Uma string como spec dá `TypeError`.
- **`model_split.validate_sharing(model, sharing)`**, chamado em `_initial_shared_parameters` sobre
  o modelo de referência e sobre cada candidato, inclusive no standalone. Exige:
  - conjunto compartilhado não vazio. O MTUNetPlusPlus passa a falhar alto, federado ou local-only;
  - tensores só de ponto flutuante;
  - na lista explícita, todo nome é filho de topo dono de algum tensor (recusa `decoder6` e
    `downsample`).
- **Registro do que foi compartilhado:**
  - `resolved_shared_modules` vai para o `metadata.yaml` de cada cliente (`shared_modules`);
  - uma linha `[sharing] modules=… tensors=32 params=10135840 (64.07% of 15819799)` no
    `execution.log`, uma vez por run (em `run()`). A contagem é sobre tensores do `state_dict`,
    então na E7 inclui os buffers FrozenBN.
- **Call sites:** `client.py`, `training_federated.py`, `conflict_diagnostics.py`,
  `scripts/gradient_conflict_{controls,methods}.py` (inclusive o keyword em controls), a mensagem
  do `server.py` e os testes da E3. Gate: `grep -rn "share_stem=" src scripts tests` não acha
  nenhuma chamada ao `model_split`.
- **Hash:** a chave não entra no `src/config.yaml` (`StudyHashGuardTests` em
  `tests/test_e6_sharing.py`).

#### E6.3 — Correção do M1 (C2)

- `trunk_gradient_halves` usa só as chaves de `named_parameters()` e pula `grad is None`.
- O bloco que o objetivo do cliente não alcança fica **no npz, com zeros**, porque o `m1_pairs`
  indexa todos os blocos do outro cliente. O cosseno desse bloco sai NaN, e o `shared_total` fica
  finito. Com o trunk, a saída é bit a bit igual à de antes.
- `scripts/smoke_federated.py::_verify_m1_coverage`, chamado com `--conflict-diagnostics`. Toda
  `fold_0/conflict/round_*` amostrada precisa ter o npz de todos os clientes do metadata, com o
  mesmo conjunto de blocos, igual aos blocos das chaves de `global_shared.pt`. Imprime
  `M1_COVERAGE_OK rounds=… clients=… blocks=…`.
- Reaplicar o loop antigo em memória faz os testes novos falharem com os erros originais
  (`AttributeError` e `KeyError`).

#### E6.4 — Testes (`tests/test_e6_sharing.py`, 29 testes)

- `ResolverTests` e `ValidationTests`: resolução e todas as recusas da E6.2.
- `LegacyEquivalenceTests`: sem a chave, as chaves são as da regra de prefixo escrita à parte.
  - MTnnUNet: 1 canal/3 classes e 3 canais/7 classes, com e sem stem, com e sem Kendall.
  - Multi_BTSUNet mantém o `bottleneck2`.
  - MTUNetPlusPlus: conjunto vazio, e o `validate_sharing` levanta erro.
- `ExactMatchTests`: `("enc",)` não captura `enc2.*`.
- `WideSetTests`:
  - 32 chaves e 10 135 840 valores, com shapes iguais em gray e rgb;
  - o pessoal é exatamente `{encoder1, output1–4, classifier}`, mais `task_log_vars` com Kendall;
  - ida e volta do estado, e a linha `[sharing]`.
- `InitialSharedTests`: o config de 4 datasets com a lista dá 32 arrays; sem a lista, 10; com o
  MTUNetPlusPlus, erro antes do Flower.
- `M1ProbeTests`:
  - trunk bit a bit igual ao loop antigo;
  - zeros nos blocos não alcançados (só-cls e só-seg);
  - buffer compartilhado sem `KeyError`;
  - clientes mistos passam por `m1_pairs` e `summarize`.
- `SmokeCoverageTests`: o `_verify_m1_coverage` aceita probes completos e recusa probe faltando ou
  com outros blocos.
- `RunConfigTests` (C3): o config versionado tem a lista ampla, `share_stem: false`, 400 rodadas, o
  bloco `experiment` e o master da E3, e passa no `validate_federated_config`. Com `skipUnless` no
  config resolvido da E3 em `runs/`: só `federated.rounds`, `federated.shared_modules` e
  `experiment.{study_id,arm_id,method_id}` diferem dele.
- `StudyHashGuardTests`: a chave fora do `src/config.yaml`.

#### E6.5 — Config (C3): `studies/e6_sharing/wide_sharing.yaml`

- Gerado com `yaml.safe_load` do config base, as mudanças e `yaml.safe_dump(sort_keys=False)`, mais
  um cabeçalho no padrão de `studies/e3_loss/phase_a/*.yaml`. A lista entra logo depois de
  `share_stem`.
- Diferenças, conferidas por flatten-and-diff (161 → 162 folhas) e por `diff` textual:
  - `federated.rounds`: 200 → 400;
  - `federated.shared_modules` (ausente → 17): `encoder2–5`, `bottleneck`, `decoder1–5`,
    `upsample1–5`, `process_encoder_5`, `process_decoder_5` (no arquivo, em ordem decrescente:
    `decoder5` … `decoder1`, `upsample5` … `upsample1`);
  - `experiment`: `study_id: e6_wider_sharing`, `arm_id`/`method_id: wide_sharing`; `seed` 1993 e
    `setup: federated` já eram iguais.
- Fica igual à E3: Adam 1e-4, batch 32, BF16, DiceBCE 0.5/0.5 com Batch Dice, λ e agregação 4:1,
  seed pessoal `dataset`, M1+M2 a cada 10 rodadas (`sketch_dim` 65536), `num_gpus` 0.5 e o master
  `data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv`.

#### E6.6 — Validação (2026-10-18, CPU, `CUDA_VISIBLE_DEVICES=` vazio)

- **Suíte:** 271 testes, 1 erro e 3 skips (CUDA). O erro é antigo e não vem da E6: o
  `.gitignore:116` (`*.html`) ignora `src/monitor/static/index.html`, então a worktree não tem a
  página e o `tests.test_monitor.ServerTests.test_api_and_page` recebe 404.
- `scripts.sync_agent_assets --check`: OK.
- **Smoke com o config versionado:**
  `.venv/bin/python -m scripts.smoke_federated --config studies/e6_sharing/wide_sharing.yaml --conflict-diagnostics`.
  - `M1_COVERAGE_OK rounds=[1, 2] clients=5 blocks=17` e `SMOKE_OK setup=federated`.
  - `global_shared.pt` com 32 chaves; o `metadata.yaml` dos 5 clientes lista os 17 módulos.
  - `ISIC_2018_seg_0` e `ISIC_2018_cls_0` gravaram o npz nas 2 rodadas, cada um com 17 blocos.
    Nenhum aviso `failed; training continues`; nenhum D1 no diagnóstico.
  - O `execution.log` tem a linha `[sharing] … tensors=32 params=10135840 (64.07% of 15819799)`.
  - Movido para `runs/validation/e6_smoke/c3_wide_sharing_20261018_225558_…`, com o stdout e o
    config de entrada. Ao lado está o smoke do C2, feito antes do C3 com um config temporário (o
    da E3 mais a lista), com o mesmo resultado.
- O sha256 do master da E3 seed 1993 continua `389116b5…bd3b`.

#### E6.7 — Handoff de treino

1. **O que alterar:** nada no `src/config.yaml`. O config é o `studies/e6_sharing/wide_sharing.yaml`
   (diff exato na E6.5).
2. **Pré-voo:**
   - `sha256sum /home/lucas/fed_multi_task_breast_cancer/data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv`
     dá `389116b572254ba492d0f11212079d23c2d32dc8855b4028682951d4d207bd3b`;
   - `nvidia-smi --query-compute-apps=pid --format=csv,noheader` sai vazio;
   - um run por vez na GPU. Ordem sugerida: E6 → E7 → E4.
3. **Comando:**
   `cd /home/lucas/fed_e6_sharing && .venv/bin/python -m src.training_federated --config studies/e6_sharing/wide_sharing.yaml`
4. **Tempo e recursos:** ~3h20–3h45 (o braço da E3 levou 1h41 com 200 rodadas). Disco: ~3.5 GB.
   `num_gpus` 0.5. Cada rodada amostrada pelo M2 grava deltas transitórios de ~690 MB. Depois de
   ~20 rodadas, conferir os segundos por rodada no `execution.log`.
5. **Artefatos:** `runs/<ts>_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-SIIM_ACR-TCGA_LGG/`. O
   `runs/` da worktree é link para o da checkout principal, então o diretório fica fisicamente lá.
   A análise usa `federated_test_results.csv`, `conflict_*.csv`, `negative_transfer_*.csv`,
   `diagnostics/` e o app (E2).
6. **Sucesso:**
   - 24 linhas de teste;
   - `global_shared.pt` da rodada 400 com 32 chaves;
   - 41 diretórios `fold_0/conflict/round_*` (1, 11, …, 391 e 400), cada um com 16 `*_m1.npz` e 16
     `*_m2.csv`;
   - nenhum D1.

   ```bash
   R=runs/<ts>_FEDERATED_MTnnUNet_Curated_BUSI-ISIC_2018-SIIM_ACR-TCGA_LGG
   .venv/bin/python -c "import pandas as pd; print(len(pd.read_csv('$R/federated_test_results.csv')))"  # 24
   .venv/bin/python -c "import torch; s=torch.load('$R/fold_0/global_shared.pt', weights_only=False); print(s['round'], len(s['keys']))"  # 400 32
   ls -d $R/fold_0/conflict/round_* | wc -l  # 41
   for d in $R/fold_0/conflict/round_*; do echo $d $(ls $d/*_m1.npz | wc -l) $(ls $d/*_m2.csv | wc -l); done  # 16 16
   .venv/bin/python -c "import pandas as pd; f=pd.read_csv('$R/diagnostics/findings.csv'); print((f.check_id == 'D1').sum())"  # 0
   grep '\[sharing\]' $R/execution.log  # tensors=32 params=10135840 (64.07% of 15819799)
   ```

7. **Retomada:** `run(cfg, run_path=…, resume=True)` reaproveita só folds completos. Com CV=1 há um
   fold só, então uma interrupção recomeça da rodada 1.

#### E6.8 — Pendências

- **Diluição do ISIC** (E6.0): 20% no decoder e 80% em `process_*`, porque os clientes
  `single_task` devolvem blocos que não treinaram. Decisão do usuário: "revisitar no futuro".
- **Inicialização do decoder:** na E6, ele parte do estado global, igual para todos os clientes; na
  E3, vinha da seed pessoal. A comparação com a E3 muda o compartilhamento e a inicialização do
  decoder juntos. O estado global depende dos canais de `federated.datasets[0]` (E6.0).
- **Sketch do M1 por bloco ~2.2× menor:** o `sketch_dim` (65536) é repartido pelo tamanho dos
  blocos, e o total compartilhado passa de 4.66 M para 10.14 M. O cosseno por bloco do trunk fica
  mais ruidoso que na E3.
- **Números de mecanismo não comparáveis com a E3:** as linhas `pair_class`/`dataset_pair` do
  `conflict_summary` (bloco `shared_total`) e o cancelamento `negative_transfer_*`
  (overall/intra/inter, do Gram de todo o conjunto compartilhado) passam a cobrir os 17 blocos, não
  os 5 do trunk. Os registros da E3 (M1 `intra_same_tasks` 0.098 ± 0.006, cancelamento −0.017) são
  sobre o trunk. Além disso, os dois clientes `single_task` do ISIC têm suporte não nulo disjunto em
  `decoder1–4`/`upsample1–4`/`process_*`, o que puxa o cosseno seg×cls do ISIC para 0 (e o D9 para
  "sem interação"). Uma leitura do M1 equivalente à E3 tem de ser recalculada sobre os 5 blocos do
  trunk a partir dos `*_m1.npz` por bloco (normas exatas + sketches) ou de `conflict_m1_pairs.csv`.
  O cancelamento só do trunk passou a ser gravado (decisão do usuário, 2026-10-19): quando há
  mais que o trunk compartilhado, o servidor grava também `negative_transfer_trunk` por rodada
  (Gram só de `encoder2–5` + `bottleneck`, com os pesos reais da agregação), e o relatório escreve
  `negative_transfer_trunk_{rounds,summary}.csv`. É esse o número comparável com o −0.017 da E3.
  O `negative_transfer.html` e o log ainda chamam a grandeza total de "trunk compartilhado".
- O `aggregation_history.json` pode chegar a ~85 MB: 400 rodadas e 17 blocos, contra 14 MB na E3.
- **Sem controle a 400 rodadas** (nem `primary` nem `local_only`): orçamento e intervenção ficam
  confundidos.
- `src/experiments/executive_report.py` descreve o compartilhamento em HTML fixo (só o trunk). Não
  foi alterado.
- O MTUNetPlusPlus federado, ou local-only, agora falha alto (conjunto compartilhado vazio).
- Ficam intocados por serem históricos: `docs/RESEARCH_NOTES.md:45` e o inventário da E5 neste
  arquivo.

#### E6.9 — Commits

- C1 `332b518` feat(federated): `federated.shared_modules` seleciona os módulos do FedAvg.
- C2 `7e46ba8` fix(conflict): o M1 pula parâmetros que o objetivo do cliente não alcança, e buffers.
- C3 `bda85d3` exp(e6): config `studies/e6_sharing/wide_sharing.yaml` e `RunConfigTests`.
- C4 `3305e50` docs: `AGENTS.md` (pipeline e tabela de chaves) e nota da E6 no D02.
- C5 docs(plano): este detalhamento e o registro.

**Registro.**
- 2026-10-18 — etapa planejada (macro).
- 2026-10-18 — **Detalhamento escrito e implementação concluída; a E6 aguarda o treino (⏳).**
  - C1–C4 na `exp/wider-sharing` (worktree `/home/lucas/fed_e6_sharing`, a partir do `e6ced0e`).
    Sem a chave, o compartilhamento é o de sempre; com a lista, 17 módulos, 32 chaves e 64.07%
    dos valores.
  - O M1 com o conjunto amplo quebrava em silêncio nos clientes `single_task` do ISIC (e
    quebraria com os buffers FrozenBN da E7). Está corrigido, e o smoke agora confere a cobertura.
  - Validação: `tests/test_e6_sharing.py` com 29 testes; suíte com 271 testes, só o erro antigo
    do monitor (E6.6); sync OK; dois smokes CPU (C2 e C3) em `runs/validation/e6_smoke/`.
  - Nenhum treino rodado e a GPU não foi usada. `src/config.yaml`, guard-rails,
    `Curated_BUSI_preprocessing.py`, `processed_128` e os masters estão intactos.
  - **Próximo passo do usuário:** rodar o handoff da E6.7 (~3h20–3h45) e trazer o run de volta para
    a leitura contra a E3.

---

## E7 — Transfer learning (ResNet-34 ImageNet)

**Objetivo.** Medir o efeito de um encoder pré-treinado no ImageNet sobre o treino federado.

**Situação atual.**
- Nenhum modelo do zoo (`src/models/`) usa pesos pré-treinados: todos partem de inicialização
  aleatória.
- O MTnnUNet usa InstanceNorm (D20). O conjunto compartilhado é fixo no código até a E6.

**Entregáveis (macro).**
1. Uma arquitetura nova: encoder ResNet-34 do torchvision com pesos ImageNet, com os blocos
   expostos sob os nomes de módulo do MTnnUNet, para que a divisão compartilhado/pessoal continue
   valendo.
2. BatchNorm do encoder como `FrozenBatchNorm`: estatísticas, escala e deslocamento congelados, sem
   divergência entre clientes.
3. Compartilhamento amplo da E6: encoder pré-treinado e decoder compartilhados; o stem (`encoder1`)
   segue pessoal.
4. Handoff de treino, análise e fonte registrada (D20 muda para o encoder pré-treinado).

**Treino necessário?** Sim: 1 run federado solto, seed 1993, 400 rodadas, executado pelo usuário.

**Critério de pronto.** Run de 400 rodadas lido contra a E6, com a decisão sobre o encoder
pré-treinado registrada.

**Riscos/dependências.** Depende do mecanismo `shared_modules` da E6.

**Detalhamento.**

#### E7.0 — Fatos (levantados em 2026-10-18)

- **Modelo.** O `MTResNet34UNet` tem 27.927.639 valores de estado (1 canal, 3 classes), contra
  15.819.799 do MTnnUNet. No compartilhamento amplo da E6 viajam 202 tensores e 23.914.976 valores:
  **85,6%** do modelo, contra 64,1% no MTnnUNet. Ficam pessoais `encoder1`, `output1–4` e
  `classifier`, e as formas compartilhadas são iguais com 1 e 3 canais.
- **Parte pré-treinada.** 180 tensores e 21.301.696 valores: 36 convoluções e os 144 buffers das 36
  camadas `FrozenBatchNorm2d`. Os buffers estão no `state_dict`, mas não em `parameters()`.
- **Loader.** As imagens chegam em float 0–255, sem normalização. As de 3 canais já vêm em RGB
  (`BUSI_dataset.py` converte o BGR do OpenCV). O TCGA entra com os 3 canais de RM
  (pré-contraste, FLAIR, pós-contraste) tratados como RGB.
- **Pesos.** `~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth` (87.319.819 bytes; sha256
  `b627a593…0590f`), o `ResNet34_Weights.IMAGENET1K_V1` do torchvision 0.22. Esse checkpoint não
  traz `num_batches_tracked`.
- **Base.** O E6-C2 (`7e46ba8`) trouxe `federated.shared_modules` e o M1 só sobre parâmetros, pulando
  `grad is None`. Sem ele, os buffers FrozenBN dariam KeyError no M1, e o `observational()` engoliria
  o erro.
- **Master.** `data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv`, sha256
  `389116b5…bd3b` (conferido em 2026-10-18).

#### E7.1 — Decisões (grill de 2026-10-18)

1. Um run federado solto: seed 1993, 400 rodadas, `steps` 10, sem `study_runner`, sem gêmeo
   local-only e sem controle. A comparação com a E6 fica com o usuário.
2. Config base: a do vencedor da E3 (`init_seed_dataset`, seed 1993). Mudam só a arquitetura,
   `shared_modules`, `rounds` e o bloco `experiment`. O master é o mesmo da E3.
3. BN **totalmente congelada** (estatísticas, γ e β). As alternativas viram próximos passos (E7.9).
4. Compartilhamento amplo da E6 (17 módulos), com o stem `encoder1` pessoal. A agregação
   hierárquica não muda.

#### E7.2 — Arquitetura (`src/models/multitask/MTResNet34UNet.py`)

Os nomes de módulo e a geometria são os do MTnnUNet, então split, `validate_sharing` e servidor
funcionam sem mudança.

| Módulo | Conteúdo | Canais | px @128 | Compartilhado | Pré-treinado |
|---|---|---|---|---|---|
| `encoder1` | `LevelBlock(C,32,32)` sobre x cru (0–255) | C→32 | 128 | não | não |
| `encoder2` | `conv1`, `bn1`, `relu` sobre a entrada normalizada | 3→64 | 64 | sim | sim |
| `encoder3` | `maxpool`, `layer1` | 64→64 | 32 | sim | sim |
| `encoder4` / `encoder5` / `bottleneck` | `layer2` / `layer3` / `layer4` | →128 / →256 / →512 | 16 / 8 / 4 | sim | sim |
| `upsample5–1`, `decoder5–1` | ConvT + `LevelBlock`, larguras 512→256→128→64→32→16 | | 8→128 | sim | não |
| `process_encoder_5` / `process_decoder_5` | `ConvInNormLeReLU` 256→256 / 128→256 | | 8 | sim | não |
| `output1–4` | supervisão profunda em resolução cheia, como no MTnnUNet | →1 | 128 | não | não |
| `classifier` | `ConvInNormLeReLU` 768→512, GAP, FC 512→256→n′ (n′ = 1 no binário) | | | não | não |

- O backbone é uma variável local (`resnet34(weights=None, norm_layer=FrozenBatchNorm2d)`), nunca
  atribuída a `self`. Assim, `fc` e `avgpool` não entram no `state_dict` e nenhuma chave se repete.
- A entrada do caminho pré-treinado é calculada no modelo: `x/255`, repetição de 1 para 3 canais e
  normalização pela média e desvio do ImageNet, guardados em buffers `persistent=False` (fora do
  estado).
- `ValueError` se `sequences ∉ {1,3}`, se os canais da entrada não batem ou se H/W não são múltiplos
  de 32.
- O `forward` devolve `([cls], [out4, out3, out2, out1])` em logits crus, sem sigmoid. `width` e
  `deep_supervision` são ignorados.
- Init Kaiming (leaky ReLU, bias zero) nas `Conv2d` de `encoder1`, `decoder1–5` e `output1–4`; o resto
  do que não é pré-treinado fica com a init padrão do PyTorch.

#### E7.3 — Pesos e BN

- `imagenet_checkpoint_path()` = `torch.hub.get_dir()/checkpoints/resnet34-b627a593.pth` (respeita
  `TORCH_HOME`).
- `map_imagenet_state()`: `conv1.`→`encoder2.conv1.`, `bn1.`→`encoder2.bn1.`,
  `layer1.`→`encoder3.layer1.`, `layer2.`→`encoder4.`, `layer3.`→`encoder5.`,
  `layer4.`→`bottleneck.`. Descarta `fc.*` e `num_batches_tracked` e levanta erro em qualquer outra
  chave.
- `load_imagenet_encoder()`:
  - guarda offline: `FileNotFoundError` com o caminho e o comando de download, sem nunca chamar
    `load_state_dict_from_url`;
  - prefixo do sha256 `b627a593` e `torch.load(weights_only=True, map_location="cpu")`;
  - o conjunto de chaves mapeadas é exatamente o dos `pretrained_modules`, e `missing`/`unexpected`
    são conferidos;
  - uma linha `[pretrained] … loaded` no `execution.log`.
- **FrozenBN.** Os buffers float32 viajam no estado compartilhado. Todo cliente devolve os mesmos
  valores, e a média ponderada de arrays idênticos é a identidade bit a bit (`_weighted_arrays`,
  testado). O servidor não muda. Fonte e alternativas no D27 de `docs/DECISION_SOURCES.md`.

#### E7.4 — Integração

- **`init_multitask_model(..., pretrained=None)`**, com import preguiçoso:
  - `MTResNet34UNet` exige `pretrained` explícito. Isso protege baseline, centralized e
    `load_multitask_experiment_artefacts`, que não foram portados.
  - `pretrained=True` em outra arquitetura levanta erro.
  - Nome desconhecido levanta `ValueError`; antes devolvia um `nn.Module()` vazio em silêncio.
- **`training_federated.py`:** `_build_model(..., pretrained=False)`. Só o modelo de referência de
  `_initial_shared_parameters` é construído com `pretrained=True`, e só para arquiteturas em
  `PRETRAINED_ARCHITECTURES`; em seguida vêm `assert_pretrained_shared` e `validate_sharing`.
  Candidatos, `_test_client` e a sonda do log `[sharing]` usam `pretrained=False`.
- **`client.py`** constrói com `pretrained=False`. Os pesos chegam aos clientes só pelos parâmetros
  iniciais globais, nos dois braços: o local-only também os recebe na rodada 1.
- **`model_split.assert_pretrained_shared(model, sharing)`** levanta erro se algum módulo de
  `pretrained_modules` ficar pessoal, porque ele começaria aleatório em todo cliente.
- **M1** usa só parâmetros (E6-C2), então os buffers ficam fora das sondas. No **M2**, os deltas dos
  buffers são zeros.
- **Smoke:** `_verify_frozen_pretrained(run_path, max_conv1_drift=0.05)` roda para arquiteturas
  pré-treinadas.
  - Exige os buffers FrozenBN de `global_shared.pt` (no local-only, os do `state.pt` de cada cliente)
    **iguais** aos do ImageNet.
  - Exige distância L2 relativa de `encoder2.conv1.weight` ao ImageNet < 0,05; uma init aleatória
    fica perto de 1,4.
  - Com `max_conv1_drift=None`, confere só os buffers e imprime a distância: é a checagem pós-treino.

#### E7.5 — Testes (`tests/test_e7_resnet34.py`: 35 testes, 1 pulado sem CUDA)

- **Forward:** C=1 e 3, n=2 e 7, a 128 e a 256. As saídas de seg ficam em resolução cheia, e os
  tensores devolvidos são exatamente as saídas das cabeças (logits crus). Contrato de entrada.
- **`state_dict`:** módulos de topo exatos (sem `backbone`/`fc`), sem `num_batches_tracked`, tudo em
  ponto flutuante, sem mean/std; 144 buffers FrozenBN fora de `parameters()`.
- **Split amplo:** 202 tensores e 23.914.976 valores; formas iguais entre gray e rgb; o pessoal é
  exatamente `{encoder1, output1–4, classifier}`. Compartilhado + pessoal reproduzem o forward bit a
  bit.
- **Carregamento**, com um checkpoint falso numa pasta temporária e o prefixo do sha trocado: mapa
  exato, chave não mapeada recusada, `pretrained=True` carrega só o encoder, sha divergente recusado e
  guarda offline com as funções de download trocadas por falhas (nunca chamadas). O checkpoint real
  é conferido quando está no cache.
- **Demais:** dispatch; freeze (um passo do Adam não muda buffer algum); `assert_pretrained_shared`;
  `_weighted_arrays` como identidade bit a bit; M1 sem KeyError, com os blocos não alcançados zerados
  como na E6; `_initial_shared_parameters` devolvendo os valores do ImageNet;
  `_verify_frozen_pretrained` (passa; falha com um buffer movido; falha só no drift com `conv1`
  aleatória); config do run.
- **BF16 CUDA** (`skipUnless`): forward/backward sob `PrecisionPolicy("bf16","cuda")`, loss e
  gradientes finitos, buffers intactos. Roda uma vez pelo orquestrador, com a GPU livre; os agentes
  rodam tudo com `CUDA_VISIBLE_DEVICES=`.

#### E7.6 — Config (`studies/e7_transfer/resnet34_imagenet.yaml`)

Gerado com `yaml.safe_load` do config base, as mudanças e `safe_dump(sort_keys=False)`, mais o
cabeçalho. O flatten-and-diff contra a base (também coberto por teste) dá só:
- `model.architecture`: `MTnnUNet` → `MTResNet34UNet`;
- `federated.shared_modules`: ausente → os 17 módulos da E6;
- `federated.rounds`: 200 → 400;
- `experiment`: `{study_id: e7_transfer_learning, arm_id: resnet34_imagenet, method_id:
  resnet34_imagenet, seed: 1993, setup: federated}`.

O resto fica igual: Adam 1e-4, batch 32, BF16, M1+M2 a cada 10 rodadas, `num_gpus` 0,5 e o master da
E3 seed 1993.

#### E7.7 — Validação (2026-10-18, CPU, `CUDA_VISIBLE_DEVICES=`)

- Suíte completa: `Ran 304 tests … OK (skipped=4)`. `scripts.sync_agent_assets --check`: OK.
- Smoke `scripts.smoke_federated --config studies/e7_transfer/resnet34_imagenet.yaml
  --conflict-diagnostics` (161 s):
  - `SMOKE_OK`;
  - `M1_COVERAGE_OK rounds=[1, 2] clients=5 blocks=17`, incluindo `ISIC_2018_seg_0` e
    `ISIC_2018_cls_0`;
  - `FROZEN_PRETRAINED_OK fold_0 round 2 buffers=144 conv1_rel_l2=0.000848`;
  - `global_shared.pt` com 202 chaves; no log, `[sharing] … tensors=202 params=23914976 (85.63% of
    27927639)` e `[pretrained] … 180 tensors, 21301696 values`.
  - Movido para `runs/validation/e7_smoke/`, com a saída em `smoke_stdout.log`.
- Pendente: o teste BF16 CUDA, rodado uma vez pelo orquestrador.

#### E7.8 — Handoff de treino

1. *O que alterar:* nada no `src/config.yaml`. O config `studies/e7_transfer/resnet34_imagenet.yaml`
   está versionado.
2. *Pré-voo:*
   - checkpoint presente: `sha256sum ~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth` começa com
     `b627a593`. Se faltar, baixar uma vez (~83 MB; o treino nunca baixa):
     `.venv/bin/python -c "from torchvision.models import resnet34, ResNet34_Weights; resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)"`;
   - `sha256sum /home/lucas/fed_multi_task_breast_cancer/data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv`
     = `389116b5…bd3b`;
   - `nvidia-smi --query-compute-apps=pid --format=csv,noheader` vazio; um run por vez na GPU.
3. *Comando:*

   ```bash
   cd /home/lucas/fed_e7_transfer && .venv/bin/python -m src.training_federated --config studies/e7_transfer/resnet34_imagenet.yaml
   ```

4. *Retomada:* como na E6. `run(cfg, run_path=…, resume=True)` só reaproveita folds completos; com
   CV=1, uma interrupção recomeça da rodada 1.
5. *Tempo e recursos:*
   - ~3h30–4h30 (~0,9–1,1× o custo por rodada da E3, com o modelo reconstruído a cada rodada);
   - disco: ~5 GB, mais ~1,6 GB transitórios (`deltas.npz`) por rodada amostrada pelo M2, apagados no
     fim da rodada;
   - RAM de pico: ~3,5 GB por cliente mais ~3 GB no driver; `num_gpus` 0,5.
6. *Artefatos:* `runs/<ts>_FEDERATED_MTResNet34UNet_Curated_BUSI-ISIC_2018-SIIM_ACR-TCGA_LGG/`.
7. *Sucesso:*
   - os critérios da E6: 24 linhas de teste; 41 diretórios `fold_0/conflict/round_*`, cada um com 16
     `*_m1.npz` e 16 `*_m2.csv`; nenhum D1;
   - `global_shared.pt` da rodada 400 com **202** chaves (o MTnnUNet amplo tem 32);
   - a linha `[pretrained] ImageNet ResNet-34 loaded … 180 tensors` no `execution.log`;
   - buffers FrozenBN da rodada 400 iguais aos do ImageNet:
     `.venv/bin/python -c "from scripts.smoke_federated import _verify_frozen_pretrained as v; v('runs/<run>', max_conv1_drift=None)"`
     imprime `FROZEN_PRETRAINED_OK fold_0 round 400 buffers=144 …` (o `conv1_rel_l2` é só
     informativo).

#### E7.9 — Pendências (próximos passos)

- **(a)** Estatísticas congeladas com γ/β treináveis e compartilhados.
- **(b)** FedBN (`li2021fedbn`): BN pessoal por cliente.
- LR discriminativo: hoje o encoder pré-treinado e o decoder do zero usam o mesmo Adam 1e-4.
- Entrada de 224 px: o ImageNet foi treinado a 224; a 128, o `bottleneck` fica em 4×4.
- Estatísticas do ImageNet aplicadas aos canais de RM do TCGA.
- O `encoder1` agora só alimenta o skip do `decoder1` (no MTnnUNet, alimentava o encoder inteiro): o
  stem pessoal deixa de adaptar a modalidade ao trunk.
- Baseline e centralized ficam protegidos (`pretrained` obrigatório), mas não portados: a E7 não tem
  referência não federada.
- E7 contra E6 muda arquitetura **e** pré-treino ao mesmo tempo. Separar os dois exigiria um braço
  `MTResNet34UNet` sem pré-treino.
- Herdadas da E6: a diluição do ISIC no decoder e em `process_*`, e a falta de controle a 400
  rodadas (orçamento confundido com intervenção).
- Logs do Ray maiores: `init_multitask_model` registra o modelo inteiro a cada construção.

**Registro.**
- 2026-10-18 — etapa planejada (macro).
- 2026-10-18 — **implementação concluída; aguarda o treino (⏳).** Detalhamento E7.0–E7.9 escrito a
  partir do plano aprovado.
  - Commits na `exp/transfer-learning`, a partir do E6-C2 (`7e46ba8`): `99e5dfc` (modelo, dispatch e
    guarda de nome desconhecido), `ec58230` (carregamento do ImageNet, integração,
    `assert_pretrained_shared` e checagem do smoke), `5036d32` (config do run), `64d7d52`
    (`AGENTS.md` e D27), mais este registro. A branch vai ser rebaseada sobre a E6 final.
  - Desvios do desenho: (1) o modelo de referência usa `pretrained=True` só para arquiteturas
    pré-treinadas, porque no MTnnUNet `pretrained=True` levanta erro; (2) `_verify_frozen_pretrained`
    ganhou `max_conv1_drift` (`None` confere só os buffers, para o run de 400 rodadas) e o caminho
    local-only.
  - Suíte, sync check e smoke OK (E7.7). O teste BF16 CUDA fica para o orquestrador.
  - Observação: o `*.html` do `.gitignore` também ignora `src/monitor/static/index.html`. Numa
    worktree nova, o `test_monitor` dá 404 até o arquivo ser linkado da checkout principal; foi
    linkado aqui, sem mudar nada versionado.
