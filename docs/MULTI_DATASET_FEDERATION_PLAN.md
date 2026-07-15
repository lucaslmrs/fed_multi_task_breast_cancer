# Plano — Ambiente federado multi-dataset (Curated BUSI + ISIC 2018)

> **Status:** aprovado, não implementado (2026-07-14). Nenhum código deste plano foi escrito.
> **Continuidade:** é a expansão que `docs/RESEARCH_NOTES.md` §1.2 já antecipa ao justificar o FedPer
> por *"a future expansion to multiple, heterogeneous datasets per client"*.
> **Pré-requisitos já prontos:** `data/Curated_BUSI/processed_128/` e `data/ISIC_2018/processed_128/`
> existem; ver `data/ISIC_2018/PAPER_NOTES.md` para os fatos medidos do ISIC.

## Contexto

Hoje o pipeline federado roda **uma base por vez**: `federated_partition` lê um único `mapping.csv`
e todos os clientes compartilham `encoder1..5 + bottleneck`. O objetivo é treinar as duas bases
**simultaneamente** na mesma federação — clientes de ultrassom de mama e de dermatoscopia
contribuindo para o mesmo encoder compartilhado.

Isso é a extensão natural do FedPer já implementado: em vez de "vários hospitais, uma modalidade",
vira "várias modalidades, um encoder". A pergunta científica passa a ser: **um encoder compartilhado
entre modalidades diferentes transfere conhecimento útil, ou só interfere?** O baseline local-only já
existente responde isso de graça — se o federado multi-base não bater o local-only, a resposta é não.

### Os números que motivam o desenho

| | Curated BUSI | ISIC 2018 |
|---|---|---|
| Imagens | 450 | 15.414 |
| Modalidade | ultrassom, **grayscale (1ch)** | dermatoscopia, **RGB (3ch)** |
| Classes | 3 (benign/malignant/normal) | 7 (MEL/NV/BCC/AKIEC/BKL/DF/VASC) |
| Pools | seg 386 / cls 450, **mesma imagem serve as duas tarefas** | seg 3.694 / cls 11.720, **disjuntos** |
| Desbalanceamento | 3,5x | 48x (NV = 66%) |

**ISIC é 34,3x maior que BUSI.**

## Decisões tomadas

| Decisão | Escolha |
|---|---|
| Canais (BUSI 1ch vs ISIC 3ch) | **Stem por dataset**: `encoder1` sai do bloco compartilhado; federa `encoder2..5 + bottleneck` |
| Espaço de rótulos | **Por dataset**: cabeça BUSI com 3 saídas, ISIC com 7 (as cabeças já são personalizadas) |
| Agregação sob assimetria 34x | **`dataset_weights` no config**, espelhando o `task_weights` que já existe |

## O que já joga a favor

Boa parte do trabalho pesado já está feita, e vale reusar em vez de reescrever:

- **As cabeças já são personalizadas** (`model_split.py`). Rótulos por dataset saem de graça: um
  `nn.Linear(256, 3)` e um `nn.Linear(256, 7)` nunca se encontram no servidor.
- **`FedPerStrategy` já pondera por `num_examples × task_weight[task]`** (`server.py:31`).
  `dataset_weight` é mais um fator na mesma linha.
- **Um cliente = uma tarefa**, e `local_trainer` só lê `data["mask"]` no seg e `data["label"]` no
  cls. Isso é o que torna a disjunção seg/cls do ISIC um **não-problema**: pools disjuntos são o
  caso natural aqui.
- **`unified_eval.evaluate(model, loader, task, num_classes, device)`** já recebe `num_classes` por
  chamada — serve os dois datasets sem mudança.
- **`paths.py`** já resolve tudo a partir de `data.dataset`; as duas bases já têm `processed_128/`.

## Design

### 1. Config (`src/config.yaml`)

O bloco `data:` hoje descreve **um** dataset. Passa a existir um registro por base, e `data.dataset`
continua existindo para os scripts single-dataset (preprocessing, EDA, treino atual) — nada quebra.

```yaml
datasets:                        # registro por base; consumido só pelo pipeline multi-dataset
  Curated_BUSI:
    variant: processed_128
    channels: 1                  # -> MTnnUNet(sequences=1)
    classes: [benign, malignant, normal]
    seg_exclude_classes: [normal]
    fold_strategy: stratified    # StratifiedKFold sobre `class`; tarefas dividem o mesmo pool
    oversampling: {seg: False, cls: True}
  ISIC_2018:
    variant: processed_128
    channels: 3
    classes: [AKIEC, BCC, BKL, DF, MEL, NV, VASC]
    seg_exclude_classes: []
    fold_strategy: per_task      # pools seg/cls disjuntos, cada um com sua regra (ver §2)
    cls_source_split: train      # só o train oficial tem lesion_id completo
    oversampling: {seg: False, cls: False}   # ver §6: 87x é overfit, usar peso na loss

federated:
  datasets: [Curated_BUSI, ISIC_2018]   # quem entra na federação
  share_stem: False              # False => encoder1 é local (multi-modal). True preserva o experimento atual.
  n_clients: {Curated_BUSI: {seg: 2, cls: 2}, ISIC_2018: {seg: 2, cls: 2}}
  dataset_weights: {Curated_BUSI: 1.0, ISIC_2018: 1.0}
  partition_file: data/federated_multi/federated_mapping.csv
```

`share_stem: True` como default nos runs single-dataset **preserva o experimento congelado atual**.

### 2. Partição multi-dataset (`src/dataset/federated_partition.py`)

A função `build_federated_partition` de hoje já faz o trabalho por base. A mudança é: rodá-la **por
dataset**, taggear, e concatenar num CSV mestre único.

- `client_id` passa de `{task}_{c}` para **`{dataset}_{task}_{c}`** (ex.: `Curated_BUSI_seg_0`).
- Nova coluna **`dataset`** no CSV mestre. `img_path` já é único entre bases (prefixos diferentes).
- Master em `data/federated_multi/federated_mapping.csv` — **não** sobrescreve o
  `data/Curated_BUSI/federated/federated_mapping.csv` congelado.

**Cada base precisa de uma regra de fold diferente** — e é aqui que mora a complexidade real:

| Base | Pool | Regra | Porquê |
|---|---|---|---|
| Curated_BUSI | 450 imagens, tarefas dividem o pool | `StratifiedKFold` sobre `class` | é o que já faz hoje; toda linha tem máscara e rótulo |
| ISIC seg | 3.694 (todos os splits oficiais) | `KFold` simples | **não há `class` para estratificar** (linhas seg do ISIC têm classe vazia) |
| ISIC cls | 10.015 (só `official_split == train`) | **`StratifiedGroupKFold(groups=lesion_id)`** | 44,9% das imagens dividem lesão com outra; split por imagem vaza |

`StratifiedGroupKFold` está disponível (sklearn 1.3.0, verificado).

**Por que só o train oficial do ISIC cls:** `lesion_id` só é publicado para ele (10015/10015). O val
e test oficiais têm 0/193 e 0/1512, então não dá para garantir que não compartilham lesão com o
train. Usar 10.015 com agrupamento correto é melhor que 11.720 com vazamento silencioso.

### 3. Dataset (`src/dataset/BUSI_dataset.py`)

Três mudanças. As duas primeiras são obrigatórias; a terceira é a que decide se isso roda na máquina.

**a) Canais por dataset.** Hoje `cv2.imread(row['img_path'], 0)` força grayscale. Vira parâmetro
`channels` (1 → `IMREAD_GRAYSCALE`, 3 → `IMREAD_COLOR`). Default 1 preserva o BUSI.

**b) Mapa de rótulos derivado do config.** Hoje é hardcoded `benign→0, malignant→1, normal→2`. Vira
`{nome: i for i, nome in enumerate(classes)}`. **Coincidência feliz:** `data.classes` do BUSI já é
`[benign, malignant, normal]`, então derivar da ordem **reproduz exatamente os rótulos atuais** — é
verificável, não é fé. Linhas sem classe (seg do ISIC) recebem `label = -1`; linhas sem máscara (cls
do ISIC) recebem máscara de zeros. Nenhum dos dois é lido, porque cada cliente tem uma tarefa só.

**c) Carregamento preguiçoso.** Hoje o `__init__` decodifica **todas** as imagens para RAM. Medido:

```
ISIC cls/train  10015 linhas --deterministic_oversampling--> 73308 linhas
73308 x 128x128x3 = 3,60 GB por cliente     (BUSI equivalente: 22,7 MB)
```

Mover o `cv2.imread` para o `__getitem__` faz linhas duplicadas virarem só referências ao mesmo
arquivo. Sem isso, o ISIC não sobe junto com Ray — o `CLAUDE.md` já avisa que rodar clientes em
paralelo estoura a RAM, e aqui piora 160x.

### 4. Split do modelo (`src/federated/model_split.py`)

```python
SHARED_PREFIXES = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")   # sem encoder1
```

`encoder1 = LevelBlock(sequences, 32, 32)` é a **única** camada cujo shape depende da modalidade
(`[32, 1, 3, 3]` vs `[32, 3, 3, 3]`); a partir do `encoder2` todo mundo vê 32 canais. Tirar o
`encoder1` do bloco compartilhado é literalmente tudo o que separa "quebra no `np.stack`" de
"funciona".

`is_shared` / `shared_keys` passam a aceitar `share_stem: bool` para preservar o comportamento atual.

> **Duplicação a corrigir:** `training_federated.py:115` reconstrói a lista de prefixos à mão em vez
> de usar `model_split.is_shared` (`model_split.py:22`). Com o split virando configurável, as duas
> cópias divergiriam. Deve passar a chamar `model_split`.

### 5. Cliente (`src/federated/client.py`)

Hoje `self.num_classes = len(config["data"]["classes"])` é global. Passa a resolver pelo dataset do
cliente:

- `roster` vira `[(client_id, dataset, task), ...]`
- `sequences=channels[dataset]`, `n_classes=len(classes[dataset])` no `init_multitask_model`
- `fit()` reporta `dataset` nas métricas (hoje já reporta `task`) — é o que o servidor usa para pesar
- `oversampling` e transforms resolvidos por `(dataset, task)`

O `state.pt` / `best.pt` por cliente já são por diretório — nada muda.

### 6. Servidor (`src/federated/server.py`)

Uma linha em `aggregate_fit`:

```python
weight = fit_res.num_examples * task_weights.get(task, 1.0) * dataset_weights.get(dataset, 1.0)
```

**O ponto crítico.** Com `dataset_weights` todos em 1.0, o peso efetivo do encoder fica:

```
BUSI:  2,8%     ISIC: 97,2%        (ISIC é 34,3x maior)
```

O BUSI não aprenderia nada do federado, e o experimento não responderia nada. Para contribuição
igual, `dataset_weight` deve ser ~inversamente proporcional ao tamanho (`Curated_BUSI: ~34.0`).
Recomendação:

- **começar com contribuição equalizada** e tratar o `num_examples` puro como *ablação* ("FedAvg
  padrão falha sob assimetria de base") — é um resultado publicável por si só;
- o `FedPerStrategy` **já loga os pesos normalizados por rodada** (`server.py:44`), então a
  participação efetiva de cada base é auditável a cada rodada, não presumida.

**Sobre oversampling no ISIC:** `deterministic_oversampling` replicaria DF **87x** e VASC 71x — a
mesma imagem 87 vezes por época é receita de overfit. Prefira `data.classes_weighted` (peso na loss
por frequência inversa), que é o que o paper de referência do ISIC faz e o
`init_criterion_classification` já suporta. Por isso `oversampling.cls: False` para o ISIC no §1.

### 7. Avaliação (`src/training_federated.py`, `unified_eval`)

`unified_eval` não muda (já recebe `num_classes` por chamada). O que muda:

- `_test_client` resolve `num_classes` pelo dataset do cliente
- `federated_test_results.csv` ganha coluna `dataset`
- `analyze.py` agrega por `dataset × task × setup` em vez de `task × setup`
- **AUC/macro-F1 não são comparáveis entre bases** (3 vs 7 classes) — compare federado vs local-only
  *dentro* de cada base, nunca BUSI contra ISIC.

## Arquivos a modificar

| Arquivo | Mudança |
|---|---|
| `src/config.yaml` | bloco `datasets:`; `federated.datasets/share_stem/dataset_weights/n_clients` |
| `src/dataset/federated_partition.py` | loop por dataset, regra de fold por base, colunas `dataset` + `client_id` prefixado |
| `src/dataset/BUSI_dataset.py` | `channels`, mapa de rótulos do config, **lazy load**, tolerar máscara/classe ausente |
| `src/dataset/federated_dataloader.py` | passar `channels`/`classes` do dataset do cliente |
| `src/federated/model_split.py` | `share_stem` → `encoder1` fora do bloco compartilhado |
| `src/federated/client.py` | roster com `dataset`; `sequences`/`n_classes` por base; reportar `dataset` |
| `src/federated/server.py` | `dataset_weights` na ponderação |
| `src/training_federated.py` | roster multi-dataset; usar `model_split.is_shared` (remover duplicação); coluna `dataset` |
| `src/experiments/analyze.py` | agregar por `dataset × task × setup` |
| `CLAUDE.md` | seção do modo multi-dataset |

## Verificação

Em ordem, cada passo barato antes do caro:

1. **Não-regressão single-dataset (primeiro de tudo):** com `share_stem: True` e
   `federated.datasets: [Curated_BUSI]`, o `federated_mapping.csv` do BUSI deve sair **idêntico** ao
   congelado, linha por linha. Mesma técnica que já provou a reorganização multi-dataset e o refactor
   do preprocessing.
2. **Split do modelo:** `python -m src.federated.model_split` (o self-test já existe) com
   `share_stem=False`, e assertar que nenhuma chave `encoder1` está no bloco compartilhado, e que
   `shared + personalized` continua cobrindo todo tensor.
3. **Compatibilidade de shapes entre bases — o teste que justifica o desenho:** instanciar
   `MTnnUNet(sequences=1, n_classes=3)` e `MTnnUNet(sequences=3, n_classes=7)`, e assertar que
   `get_shared_state` dos dois retorna listas de **shapes idênticos**. Se isso passar, o FedAvg não
   quebra. Assertar também que **falha** com `share_stem=True` (prova que a guarda é necessária).
4. **Partição:** contagens por `dataset × task × split × fold`; zero vazamento treino/teste por fold
   em cada base; e **zero `lesion_id` cruzando treino/teste** no ISIC cls (o ponto do
   `StratifiedGroupKFold`).
5. **Memória:** medir RSS de um cliente ISIC cls com lazy load — deve cair de ~3,6 GB para ~centenas
   de MB. Sem isso, o resto não roda.
6. **Smoke run:** `rounds: 2`, `local_epochs: 1`, 1 cliente por (base, tarefa), CPU. Conferir no log
   do servidor que os pesos normalizados batem com o `dataset_weights` configurado.
7. **Run completo** só depois de 1–6.

## Riscos e limitações

- **Interferência negativa é um resultado possível.** Ultrassom e dermatoscopia podem simplesmente
  não ter features de alto nível em comum, e o encoder compartilhado pode piorar as duas. O
  local-only já existente é o juiz. Vale ter isso como hipótese declarada, não como fracasso.
- **`dataset_weights` é um hiperparâmetro novo.** Equalizar contribuição é uma escolha, não uma
  verdade. Merece uma varredura pequena, como o paper do ISIC fez com o 5:1 seg:cls.
- **Não há grupo de lesão para o Task 1 do ISIC.** O agrupamento só cobre o cls. Se o Task 1 tiver
  múltiplas fotos da mesma lesão, o seg do ISIC pode vazar entre folds e não temos como saber.
- **Sem `lesion_id`, o val/test oficial do ISIC fica fora** do pool cls (11.720 → 10.015).
- **Custo computacional.** O ISIC é 34x o BUSI; cada rodada federada fica dominada por ele. Com
  `rounds: 50` e `local_epochs: 2`, planeje horas, não minutos.
- **MTL centralizado continua impossível para o ISIC** (nenhuma linha tem imagem+máscara+rótulo).
  O "teto" da comparação precisa ser redefinido para o modo multi-dataset — fora do escopo aqui.

## Fora do escopo

- Redefinir o baseline centralizado multi-dataset.
- Ablação `share_stem: True` vs `False` (só é possível se os canais forem unificados).
- Espaço de rótulos binário unificado entre bases.
- Task 2 do ISIC (atributos dermatoscópicos).
