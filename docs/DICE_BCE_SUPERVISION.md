# Dice + BCE e supervisão parcial

O protocolo `example_multi_dataset_dice_bce` usa todas as máscaras binárias válidas, inclusive
máscaras vazias. Uma máscara vazia significa ausência do alvo; um arquivo ausente significa
anotação indisponível. Nunca criar um alvo negativo a partir de um placeholder ou da ausência de
um diagnóstico. A classe `normal` não tem tratamento especial no critério.

## Configuração e contrato

```yaml
loss:
  function: DiceBCE
  dice_weight: 0.5
  bce_weight: 0.5
  inversely_weighted: true

evaluation:
  segmentation_primary_metric: dice_positive
  segmentation_protocol: positive_and_empty_v1
```

A perda é `dice_weight * Dice + bce_weight * BCEWithLogitsLoss`. Os pesos são aplicados
literalmente, sem normalização, e precisam ser finitos, não negativos e ter soma positiva.
Dice mantém `sigmoid=True`, `squared_pred=True`, `smooth_nr=1` e `smooth_dr=1`. BCE recebe logits.
A combinação é aplicada a cada saída de deep supervision antes da ponderação inversa existente.
As reduções usam FP32 também sob autocast BF16. As opções anteriores de perda continuam disponíveis.

`has_mask` e `has_label` selecionam as amostras de cada tarefa **antes** de acessar os alvos.
Uma tarefa sem amostras no batch não contribui com perda; o peso da tarefa presente não muda.
Imagens sem qualquer supervisão são rejeitadas. Uma tarefa solicitada precisa receber supervisão
no treino. A máscara vazia válida continua tendo `has_mask=True`.

O clássico multitarefa usa os pesos de tarefa `alpha` e `1-alpha`; o federado mantém seus pesos
normalizados de `aggregation.task_weights`. A nova perda não altera esses pesos. A comparação dos
componentes Dice e BCE com coeficientes iguais não implica gradientes de mesma magnitude.

## Clássico e múltiplas bases

Cada execução clássica trabalha com um dataset. O registro `datasets` fornece canais e, ao trocar
`data.dataset`, as classes compatíveis; subconjuntos válidos de classes continuam possíveis.
Segmentação e classificação isoladas selecionam somente imagens com supervisão da tarefa.
O multitarefa aceita imagens conjuntas, somente segmentação e somente classificação.

Imagens rotuladas, incluindo as que também têm máscara, são divididas uma única vez com
estratificação. O pool sem rótulo de classificação usa uma divisão aleatória. Grupos conhecidos
por `lesion_id` ficam inteiros entre treino, validação e teste; um grupo que atravessa pools de
supervisão independentes é rejeitado. O ISIC usa a origem de classificação configurada em
`cls_source_split: train`, onde os grupos de lesão estão disponíveis. Isso é um split de pesquisa,
não uma reprodução do benchmark oficial do desafio.

No multitarefa clássico, `data.oversampling: false` evita repetir simultaneamente as máscaras das
classes reforçadas. `data.class_weighting: balanced_fold` calcula `N/(K*n_c)` somente a partir
dos rótulos supervisionados de treino do fold; uma classe sem exemplos recebe peso zero.
Os pesos e a ordem das classes ficam em `fold_<n>/supervision_metadata.yaml`.
Tarefas únicas mantêm oversampling configurável quando todas as imagens selecionadas têm rótulo.

O ISIC continua com clientes federados `single_task`: aceitar supervisão parcial no clássico não
transforma conjuntos disjuntos em um cliente federado com ambas as tarefas nas mesmas imagens.
O benchmark `training_centralized` existente continua BUSI-only, usando seu master histórico,
e recebe o critério configurado; não integra os cinco braços do estudo multi-dataset.
Anotação parcial por pixel e segmentação multiclasse não são suportadas por este novo contrato binário.

## Inferência e métricas

O novo padrão desliga `overlap_seg_based_on_class` e `overlap_class_based_on_seg` no clássico.
As opções históricas permanecem disponíveis; não foi criado bloqueio pela classificação no
federado. O limiar de segmentação continua `sigmoid(logits) > 0.5`.

| Campo | Definição |
|---|---|
| `dice_positive`, `iou_positive` | Média por imagem com alvo não vazio |
| `empty_fp_image_rate` | Fração de imagens com alvo vazio que recebem algum pixel positivo |
| `empty_predicted_area_fraction` | Média da fração da imagem prevista positiva entre alvos vazios |
| `n_positive`, `n_empty` | Quantidade de imagens supervisionadas em cada grupo |
| `dice`, `iou` / `DICE`, `Jaccard index` | Métricas globais históricas, incluindo negativos |

Ausência de um grupo produz `NaN` e contagem zero, nunca desempenho perfeito inventado.
No novo protocolo, `dice_positive` é a referência principal de segmentação nas comparações;
leitores de CSVs antigos mantêm `dice` quando não há declaração do novo protocolo. Os resumos
identificam o escopo das métricas. As médias do estudo continuam seguindo a agregação existente
por cliente/fold; não são estimativas automaticamente ponderadas pelo tamanho de cada dataset.
O clássico grava também os componentes e contagens por época em `supervision_metrics.csv`.
A seleção de checkpoints continua usando a perda de validação.

## Migração e execução

O manifesto permanece em `studies/example_multi_dataset.yaml`, com novo `study_id` e os mesmos
cinco braços. As partições base e `single_task` têm diretórios próprios. O config federado avulso
aponta para o novo master base. Os arquivos e resultados anteriores não são sobrescritos e o
protocolo anterior permanece no histórico Git. A perda, seus pesos e a avaliação entram na
assinatura científica; alterar a inclusão de normais também altera a assinatura da partição.

Comandos a partir da raiz do repositório:

```bash
cd /home/lucas/fed_multi_task_breast_cancer
.venv/bin/python -m unittest discover -v
.venv/bin/python -m scripts.sync_agent_assets --check
.venv/bin/python -m src.experiments.study_runner --dry-run

# Smokes de uma época com amostras reais selecionadas depois dos splits.
.venv/bin/python -m scripts.smoke_training_paths --paths multitask segmentation classification
.venv/bin/python -m scripts.smoke_training_paths --dataset ISIC_2018 --paths multitask segmentation classification

# Dois rounds, CPU, partição temporária e braços federado/local-only pareados.
.venv/bin/python -m scripts.smoke_federated --setup both --holdout --topology multi_task --samples 3

# Iniciar o estudo completo quando desejado; os masters compatíveis são reutilizados.
.venv/bin/python -m src.experiments.study_runner --seed-profile operational
```

Os smokes clássicos aceitam `--cpu` para FP32 e descartam seus arquivos temporários depois de
verificar os artefatos. Eles validam execução, não qualidade científica. O estudo operacional
permanece holdout 70/30, seed 1993, 200 rodadas e 10 passos locais por rodada. Seus resultados são
somente descritivos. Não comparar Dice global de protocolos com diferentes proporções de negativos
como se houvesse melhora na localização de lesões.
