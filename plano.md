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
| E1 | Validação do treinamento | ⬜ | `feat/training-validation` | — |
| E2 | App local de monitoramento e comparação | ⬜ | `feat/monitor-app` | — |
| E5 | Validação das fontes (consolidação) | ⬜ | `docs/sources` | — |
| E3 | Ablação: loss fed multitask multi-dataset | ⬜ | `exp/loss-ablation` | — |
| E4 | Ablação: pré-processamento de imagens | ⬜ | `exp/preprocessing-ablation` | — |

Ordem de execução: **E1 → E2 → E5 → E3 → E4**.
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

## E1 — Validação do treinamento

**Objetivo.** Verificar se há problemas durante o treinamento (numéricos, de convergência, de
aprendizado por cliente ou de agregação). Também deixar guardas que detectem esses problemas cedo nos
próximos treinos.

**Situação atual.**
- Checagens de config/contrato em `src/federated/local_trainer.py`, `client.py`, `server.py` e
  `src/training_federated.py`.
- NaN-mean na agregação. `server.py` já marca NaN quando o delta de um cliente é zero.
- Artefatos disponíveis para auditoria:
  - `fold_*/client_*/training_history.csv`: formato longo, um registro por step.
  - `fold_*/aggregation_history.json`: cosseno entre deltas e `negative_transfer`.
  - `gpu_telemetry.csv`, `runtime_events.csv`, `training_curves/history.csv` e `execution.log`.
- **Lacuna:** não há guarda por step para loss NaN/Inf, nem detecção de divergência, nem monitoramento
  de norma de gradiente ou abort antecipado.

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

**Detalhamento.** _(a preencher quando a etapa iniciar)_

**Registro.**
- 2026-10-04 — etapa planejada (macro).

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
