# Cheat sheet — estudo federado BUSI + ISIC

Execute tudo a partir da raiz do projeto.

```bash
cd /home/lucas/fed_multi_task_breast_cancer
source .venv/bin/activate
```

## Validação rápida

```bash
# 1. Testes unitários/integrados (sampler, agregação, loaders, losses e análise)
python -m unittest discover -v

# Testar somente o fluxo holdout (70/30, determinismo, loaders e análise)
python -m unittest -v tests.test_holdout

# 2. Resolver os oito braços sem treinar nem regenerar partições
python -m src.experiments.study_runner --dry-run

# 3. Smoke real dos oito braços: 2 rodadas, 1 fold, CPU e amostras limitadas
python -m src.experiments.study_runner --smoke --seed-profile operational

# 4. Smoke federado/local-only específico para CV=1, com partição temporária
python -m scripts.smoke_federated --setup both --holdout --samples 2
```

O smoke com `--holdout` cria e remove uma partição temporária. Ele não sobrescreve a partição
congelada usada pelos estudos de cross-validation.

Artefatos do smoke:

```text
runs/studies/multi_dataset_balance_v1/smoke/
├── execution_plan.csv
├── run_index.csv
├── runs/seed_1993/<arm_id>/
└── analysis/
    ├── summary_per_task_setup.csv
    ├── summary_by_seed.csv
    ├── method_comparisons.csv
    ├── per_client_method_deltas.csv
    ├── pooled_auc.csv
    ├── aggregation_audit.csv
    ├── data_composition.csv
    ├── class_balance_audit.csv
    └── report.html
```

## Execução científica

Primeiro execute apenas a comparação principal com a seed operacional:

```bash
python -m src.experiments.study_runner \
  --seed-profile operational \
  --arms primary local_steps_ce
```

Depois execute ou retome a matriz completa da seed `1993`:

```bash
python -m src.experiments.study_runner --seed-profile operational
```

Quando a execução operacional estiver validada, rode as três seeds finais:

```bash
python -m src.experiments.study_runner --seed-profile final
```

O runner reutiliza braços completos, partições com checksum, resultados de teste por fold e
checkpoints que chegaram à rodada final. Se existir um braço incompleto, retome-o explicitamente;
folds parciais são preservados para diagnóstico e refeitos sem contaminar o estado:

```bash
python -m src.experiments.study_runner \
  --seed-profile operational \
  --retry-incomplete
```

Para retomar apenas um braço específico:

```bash
python -m src.experiments.study_runner \
  --seed-profile operational \
  --arms ablation_budget \
  --retry-incomplete
```

Para recalcular somente tabelas, deltas, AUC e HTML a partir dos runs completos:

```bash
python -m src.experiments.study_runner \
  --seed-profile operational \
  --analyze-only
```

Para gerar ou atualizar o relatório executivo autossuficiente na raiz:

```bash
python -m src.experiments.executive_report \
  --output RELATORIO_EXECUTIVO_FEDERACAO_MULTI_DATASET.html
```

Enquanto a matriz completa ainda não existir, o gerador usa automaticamente `analysis_partial` e
marca o documento como provisório. Quando `analysis/` estiver completo, ele passa a gerar a versão
final.

## Acompanhar execução

```bash
column -s, -t < runs/studies/multi_dataset_balance_v1/run_index.csv | less -S
tail -f runs/studies/multi_dataset_balance_v1/runs/seed_1993/primary/execution.log
```

No smoke, acrescente `/smoke` depois de `multi_dataset_balance_v1`.

## Executar uma configuração isolada

```bash
# Gerar uma partição explicitamente
python -m src.dataset.federated_partition \
  --config src/config.yaml \
  --output data/federated_multi/federated_mapping.csv

# Treinar usando qualquer YAML resolvido
python -m src.training_federated --config CAMINHO_CONFIG.yaml

# Analisar CSVs manualmente
python -m src.experiments.analyze \
  --results RUN_A/federated_test_results.csv RUN_B/standalone_test_results.csv \
  --preds RUN_A/federated_cls_predictions.csv RUN_B/standalone_cls_predictions.csv \
  --out runs/comparison_manual
```

## Holdout 70/30 (`CV = 1`)

O bloco `datasets` não precisa ser alterado. Mantenha `Curated_BUSI` e `ISIC_2018` com suas
estratégias, classes, canais e regras de oversampling atuais.

No `src/config.yaml`, configure:

```yaml
training:
  seed: 1993
  CV: 1
  holdout_test_size: 0.30
```

O resultado é um único `fold=0`: 70% ficam no pool de desenvolvimento e 30% no teste. Nos
loaders clássicos, `data.train_size: 0.8` subdivide o desenvolvimento em aproximadamente 56% de
treino e 14% de validação.

### Treinamento clássico

```bash
# Multitarefa
python -m src.training_multitask

# Alternativas
python -m src.training_segmentation
python -m src.training_classification
```

### Treinamento federado multi-dataset

Use um arquivo novo para o master holdout. **Não sobrescreva** o master congelado de CV:

```yaml
federated:
  datasets: [Curated_BUSI, ISIC_2018]
  partition_file: data/federated_multi/federated_mapping_holdout_70_30.csv
  standalone: false
```

Para um teste curto, use temporariamente:

```yaml
federated:
  rounds: 2
  local_training:
    mode: steps
    steps_per_round: 2
```

Gere e inspecione a partição, depois treine:

```bash
python -m src.dataset.federated_partition --config src/config.yaml

python - <<'PY'
import pandas as pd

path = "data/federated_multi/federated_mapping_holdout_70_30.csv"
df = pd.read_csv(path)
print("folds:", sorted(df["fold"].unique()))
print("datasets:", sorted(df["dataset"].unique()))
print(df.groupby(["dataset", "task", "split"]).size())
PY

# FedPer
python -m src.training_federated --config src/config.yaml
```

A inspeção deve mostrar apenas `folds: [0]` e os datasets `Curated_BUSI` e `ISIC_2018`. O
treinamento rejeita automaticamente um master antigo cujos folds não correspondam a `CV=1`.

Para o baseline pareado, mantenha o mesmo `partition_file`, altere somente
`federated.standalone: true` e execute novamente:

```bash
python -m src.training_federated --config src/config.yaml
```

Para o treinamento completo, restaure o orçamento desejado, por exemplo `rounds: 50` e
`steps_per_round: 10`.

### Analisar Federated versus Local-only

```bash
python -m src.experiments.analyze \
  --results \
    RUN_FEDERATED/federated_test_results.csv \
    RUN_LOCAL/standalone_test_results.csv \
  --preds \
    RUN_FEDERATED/federated_cls_predictions.csv \
    RUN_LOCAL/standalone_cls_predictions.csv \
  --out runs/comparison_holdout
```

Nos CSVs, confirme `evaluation_scheme=holdout`, `n_splits=1` e
`holdout_test_size=0.30`. No holdout, a inferência é marcada como
`descriptive_only_single_holdout`; `wilcoxon_stat` e `wilcoxon_p` ficam ausentes/`NaN`, e a AUC é
rotulada como AUC do teste holdout.

## Braços do manifesto

| Braço | Agregação | Peso de cliente | Orçamento | Classificação ISIC |
|---|---|---|---|---|
| `primary` | hierárquica 50/50 | uniforme | 10 passos | CE + balanced_fold |
| `local_steps_ce` | local-only | — | 10 passos | CE + balanced_fold |
| `ablation_budget` | hierárquica 50/50 | uniforme | 2 épocas | CE + balanced_fold |
| `local_epochs_ce` | local-only | — | 2 épocas | CE + balanced_fold |
| `ablation_weighting` | hierárquica 50/50 | `num_examples` | 2 épocas | CE + balanced_fold |
| `ablation_flat` | flat | `num_examples` | 2 épocas | CE + balanced_fold |
| `ablation_focal` | hierárquica 50/50 | uniforme | 10 passos | focal sem pesos |
| `local_steps_focal` | local-only | — | 10 passos | focal sem pesos |

## Regras de interpretação

- Compare métricas separadamente por `dataset × task`; nunca faça média global ponderada pelo
  número de imagens de BUSI e ISIC.
- O resultado principal é `primary` versus `local_steps_ce`.
- O smoke comprova o fluxo, mas suas métricas não são resultados científicos.
- Os pares cliente × fold não são réplicas independentes; os testes de Wilcoxon são exploratórios.
- Não apresente os resultados como benchmark oficial do ISIC 2018.
- Não use `src.training_centralized` como baseline multi-dataset: os pools ISIC de segmentação e
  classificação não representam um treino centralizado pareado equivalente.
- A matriz operacional completa pode levar muitas horas; execute primeiro os dois braços
  principais e use `run_index.csv` para retomada segura.
