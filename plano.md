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
| E0 | Métodos adicionais de verificação de conflito de gradientes | 🟨 | `feat/gradient-conflict` | 2026-10-08 |
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

#### E0.2 — Busca na literatura (escolher 2 métodos)

**Critérios de escolha:**
- (i) medir conflito sem depender do otimizador e do acúmulo de steps (H1/H2);
- (ii) ser aplicável em FL simulado com 16 clientes e custo viável;
- (iii) ter fonte primária verificável;
- (iv) ser complementar entre si, por exemplo um método geométrico e um funcional.

**Candidatos iniciais** (a verificar nas fontes primárias; nada aqui está confirmado ainda):

| Candidato | Ideia | Fonte a verificar |
|---|---|---|
| Cosseno + similaridade de magnitude do **gradiente bruto** no trunk recebido, sobre um batch-sonda fixo | Elimina Adam e o acúmulo de steps. A magnitude capta o domínio de uma tarefa (seg tem peso 4×). | PCGrad — Yu et al., NeurIPS 2020 ("Gradient Surgery for Multi-Task Learning") |
| **Afinidade lookahead** entre clientes/tarefas: Z(i→j) = 1 − L_j(θ + Δ_i) / L_j(θ) | Medida funcional: o passo de i ajuda ou prejudica a loss de j? Não depende da geometria em alta dimensão. | TAG — Fifty et al., NeurIPS 2021 ("Efficiently Identifying Task Groupings in Multi-Task Learning") |
| Concordância de **sinal** por coordenada | Barato. Sensível a conflito localizado. | GradDrop — Chen et al., NeurIPS 2020 |
| Conflito por camada/canal em FL multitarefa | O que a literatura de FL multitarefa já mede. | FedBone (2306.17465), FedHCA² (CVPR 2024) |
| Similaridade de **representações** do trunk entre clientes | Complemento não-gradiente. | CKA — Kornblith et al., ICML 2019 |

Entregável: uma nota curta (em `docs/`) com o método, a fórmula, a fonte verificada, o custo e a
interpretação de cada candidato. **O usuário escolhe os 2 métodos.**

#### E0.3 — Implementação dos 2 métodos escolhidos

Será detalhada depois da escolha. Pontos já previstos:
- Gradiente bruto num batch-sonda: o cliente calcula o gradiente no trunk recebido **antes** do
  primeiro passo local. Assim, os números do treino não mudam.
- Para não estourar disco ou rede, gravar um *sketch* por projeção aleatória com seed fixa
  (Johnson–Lindenstrauss), que preserva cossenos e normas aproximadamente, e/ou amostrar só algumas
  rodadas.
- Afinidade lookahead custa O(n²) avaliações por rodada, então deve rodar em rodadas amostradas.
- Configuração em `runtime.diagnostics`, que fica fora do hash científico.
- Testes unitários seguindo o padrão de `tests/test_negative_transfer.py`.

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
| A6 | ~~Sem conflito de gradiente~~ **Inconclusivo** | Cancelamento `overall` = 0.667 ≈ referência ortogonal 0.668, mas o cosseno ≈ 0 inclusive entre clientes que deveriam estar alinhados. A métrica pode estar cega; ver E0. | em aberto |
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
