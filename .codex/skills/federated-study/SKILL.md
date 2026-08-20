---
name: federated-study
description: Rodar, retomar ou interpretar o estudo federado multi-dataset (FedPer com Curated BUSI + ISIC 2018) via study_runner, e ler as tabelas de análise que ele produz. Use ao executar braços do estudo, fazer smoke test, retomar execução incompleta, comparar federado contra local-only, ou interpretar dice/AUC/acurácia balanceada e p-valores dos resultados. Gatilhos - "rodar o estudo", "study_runner", "braços", "ablação", "federado vs local", "analisar resultados", "smoke federado", "report.html".
---

<!-- GERADO por scripts/sync_agent_assets.py a partir de .agents/skills/federated-study/SKILL.md — não edite este arquivo. -->
# Estudo federado multi-dataset

Execute tudo a partir da raiz do projeto, com o venv ativo.

## Validação antes de queimar GPU

```bash
python -m unittest discover -v                                  # sampler, agregação, loaders, losses, análise
python -m src.experiments.study_runner --dry-run                # resolve os braços sem treinar
python -m src.experiments.study_runner --smoke --seed-profile operational   # 2 rodadas, só o fold 0, CPU
python -m scripts.smoke_federated --setup both                  # validação ponta a ponta pareada
python -m scripts.smoke_federated --setup both --holdout        # CV=1 com master 70/30 temporário
```

O `--dry-run` resolve as configs de cada braço e escreve `execution_plan.csv` sem tocar em
partição nem em treino. Sempre rode isso depois de mexer no manifesto.

## Execução

```bash
python -m src.experiments.study_runner --seed-profile operational
python -m src.experiments.study_runner --arms primary local_steps_ce      # subconjunto
python -m src.experiments.study_runner --retry-incomplete                 # retoma folds truncados
python -m src.experiments.study_runner --analyze-only                     # só re-roda a análise
```

O manifesto é `studies/multi_dataset_balance_v1.yaml`. `--rebuild-partitions` **regenera a
partição congelada** — não use sem intenção explícita: braços já executados deixam de ser
comparáveis com os novos.

`training.CV=1` seleciona um holdout determinístico. A fração de teste vem de
`training.holdout_test_size` (padrão `0.30`); os 70% restantes formam o pool de
treino+validação. `CV>=2` mantém cross-validation. Alterar `CV` ou a fração do holdout exige uma
partição separada/regenerada com intenção explícita.

## Como o estudo é montado

Oito braços sobre a **mesma partição congelada** e a mesma semente. Eles variam em três eixos:

| Eixo | Valores |
|---|---|
| Orçamento local | `steps` (10 passos fixos/rodada) ou `epochs` (2 épocas/rodada) |
| Agregação | `hierarchical` + `uniform`/`num_examples`, ou `flat` (estilo FedAvg) |
| Perda de classificação | CE com `balanced_fold`, ou Focal sem pesos |

Cada braço federado tem um par local-only com **orçamento idêntico** — é o piso da comparação.
`primary` (federado, steps, hierárquica/uniforme, CE ponderada) pareia com `local_steps_ce`.

## Onde caem os artefatos

```
runs/studies/<study_id>/
├── execution_plan.csv, run_index.csv, study_manifest.yaml
├── resolved/seed_<n>/<arm>.yaml        # config resolvida de cada braço
├── runs/seed_<n>/<arm>/
│   ├── config.yaml, execution.log
│   ├── fold_<k>/                       # global_shared.pt, aggregation_history.json, clientes
│   ├── {setup}_test_results.csv        # uma linha por cliente × split externo
│   └── {setup}_cls_predictions.csv     # uma linha por imagem de teste
└── analysis/
    ├── summary_per_task_setup.csv, summary_by_seed.csv
    ├── method_comparisons.csv, per_client_method_deltas.csv
    ├── pooled_auc.csv, aggregation_audit.csv
    ├── data_composition.csv, class_balance_audit.csv
    └── report.html
```

Um diretório `fold_<k>.incomplete_<timestamp>` marca fold abortado e retomado — não é lixo, é
rastro de `--retry-incomplete`. Um run **sem** `{setup}_test_results.csv` foi interrompido antes
da avaliação: as curvas rodada a rodada ainda estão no `execution.log`, mas não há métrica final.

## Como ler os resultados sem se enganar

- **Acurácia sozinha mente no ISIC.** NV é 67 % das imagens rotuladas; chutar sempre NV dá 0,669.
  Sempre relate acurácia balanceada e macro-F1 ao lado. Chute uniforme em 7 classes = 0,143.
- **AUC alta com acurácia no chão** não é bug: o modelo ordena bem e erra o limiar. É a assinatura
  de perda com pesos de classe sob treino insuficiente.
- **Compare só pares de orçamento igual.** O eixo de orçamento domina os demais; comparar um braço
  `steps` com um `epochs` mede o orçamento, não a federação.
- **Os p-valores de CV não são confirmatórios.** São Wilcoxon pareado sobre pares cliente × fold de uma
  única semente, e o pipeline os marca `exploratory_only_non_independent_client_fold_pairs`. Folds
  compartilham imagens e clientes do mesmo fold foram agregados juntos — não são réplicas
  independentes. Com n = 8, o menor p bilateral possível é 0,0078, e atingi-lo significa apenas
  "os 8 pares foram na mesma direção". Para afirmação confirmatória, rode sementes independentes.
- **Holdout é somente descritivo.** Com `CV=1`, o pipeline não calcula Wilcoxon, grava
  `inference_scope=descriptive_only_single_holdout` e rotula a AUC como teste holdout, não OOF.
- **Nunca misture probabilidades entre datasets.** BUSI tem 3 classes, ISIC tem 7; a análise é
  isolada por `dataset` de propósito.

## Análise avulsa

```bash
python -m src.experiments.analyze \
  --results <*_test_results.csv ...> --preds <*_cls_predictions.csv ...> \
  --out runs/comparison --manifest studies/multi_dataset_balance_v1.yaml
```

Aceita dois ou três conjuntos. Sem `--manifest` ele não sabe quais braços formam par.

## Restrições

- **NÃO** use `--rebuild-partitions` sem confirmação explícita do usuário.
- **NÃO** edite `federated_mapping.csv` à mão — a comparabilidade dos braços depende dele.
- **NÃO** compare braços de orçamentos diferentes como se medissem o efeito da federação.
- **NÃO** descreva um p-valor deste estudo como "estatisticamente significativo".
