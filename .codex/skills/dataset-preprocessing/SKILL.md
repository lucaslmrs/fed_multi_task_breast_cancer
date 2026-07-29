---
name: dataset-preprocessing
description: Executar ou depurar o pré-processamento de um dataset já suportado (Curated BUSI, ISIC 2018), transformando a pasta raw/ do dataset na variante que o treino lê. Use ao preparar dados antes de treinar, validar caminhos e pré-requisitos, entender o mapping.csv gerado, ou diagnosticar um [ABORT] do script de pré-processamento. Gatilhos - "pré-processar", "preprocessing", "preparar os dados", "gerar mapping.csv", "rodar o preprocessing do BUSI/ISIC", "processed_128".
---

<!-- GERADO por scripts/sync_agent_assets.py a partir de .agents/skills/dataset-preprocessing/SKILL.md — não edite este arquivo. -->
# Pré-processamento de um dataset suportado

Para **adicionar um dataset novo** que ainda não tem script, use a skill `dataset-onboarding`.
Esta skill é para rodar e depurar os scripts que já existem.

## O contrato

Cada dataset tem seu **próprio** script (os layouts brutos não têm nada em comum), mas todos
emitem exatamente o mesmo contrato:

```
data/<dataset>/<variant>/
  images/      mapping.csv aponta para cá
  masks/
  mapping.csv
```

O que é genuinamente comum vive em `src/dataset/preprocessing_utils.py` (estatísticas de máscara,
colunas de metadados, helpers de resize, as travas). Os caminhos **nunca** são montados à mão:
saem de `src/dataset/paths.py` a partir de `data.root` / `data.dataset` / `data.variant`.

## Antes de executar

`data.dataset` no `src/config.yaml` **precisa** nomear o dataset que o script trata. Os scripts
abortam de propósito caso contrário — sem essa trava, rodar o script do ISIC com
`data.dataset: Curated_BUSI` sobrescreveria a variante do BUSI.

Checklist, nesta ordem:

1. `data.dataset` no `src/config.yaml` bate com o script que você vai rodar.
2. `data.variant` e `data.image_size` são coerentes entre si (`processed_128` ↔ `image_size: 128`).
   Escrever outra resolução dentro de uma variante existente é bloqueado por `assert_variant_resolution`.
3. `data/<dataset>/raw/` existe e está populado. Só o script de pré-processamento lê essa pasta.
4. Para o BUSI com `CURATED=True`: `data/Curated_BUSI/curation_list.csv` existe, com colunas
   `class` e `id`, separado por `;`.
5. Pacotes disponíveis: `numpy` (**tem que ser `<2`**), `pandas`, `cv2`, `yaml`.

## Executar

```bash
python -m src.dataset.Curated_BUSI_preprocessing   # exige data.dataset: Curated_BUSI
python -m src.dataset.ISIC_2018_preprocessing      # exige data.dataset: ISIC_2018
```

## O que cada script faz

**Curated BUSI** — ultrassom, 1 canal. Filtra por `curation_list.csv` (remoção de duplicatas por
SSIM), sobrando **450 imagens** (222 benign, 164 malignant, 64 normal). Funde múltiplas máscaras do
mesmo caso por união. Toda imagem tem máscara **e** rótulo.

> Ele redimensiona com `INTER_NEAREST`. **Não "conserte" para `INTER_AREA`** — isso muda as
> imagens curadas e invalida a partição federada congelada e todo resultado já derivado dela.

**ISIC 2018** — dermatoscopia, 3 canais. Ingere as **duas** tarefas do desafio num único mapping:
Task 1 (3.694 imagens com máscara) e Task 3/HAM10000 (11.720 com rótulo). Imagens em RGB com
`INTER_AREA`; máscaras com vizinho mais próximo e re-binarizadas. Fatos medidos em
`data/ISIC_2018/PAPER_NOTES.md`.

> Os dois conjuntos do ISIC são **disjuntos** — zero sobreposição de ID, verificada e asseverada
> em `main()`. Nenhuma imagem tem máscara e rótulo ao mesmo tempo.

## Esquema do `mapping.csv`

Colunas base (todo dataset): `img_path, mask_path, class, id, dim1, dim2, tumor_pixels,
y_max, y_min, x_max, x_min, y_size, x_size`.

O ISIC acrescenta três, por causa dos conjuntos disjuntos:

| Coluna | Significado |
|---|---|
| `task` | `seg` (tem `mask_path`, `class` vazio) ou `cls` (tem `class`, `mask_path` vazio) |
| `official_split` | `train`/`val`/`test` — de qual release oficial a linha veio |
| `lesion_id` | agrupa fotos da mesma lesão física. **Divida por ele, não por `id`** — 44,9 % das imagens da Task 3 compartilham lesão. Publicado só para o *treino* da Task 3; vazio no resto |

Colunas derivadas de máscara ficam `NaN` (não `0`) em linhas sem máscara: `0` significa
legitimamente "máscara vazia" para a classe `normal` do BUSI, então `NaN` é o único
"não se aplica" honesto.

**Consequência:** o ISIC serve ao FedPer (cada cliente cuida de uma tarefa, então pools disjuntos
são naturais) mas **não alimenta `training_centralized.py`**, que exige imagem+máscara+rótulo na
mesma linha.

## Depois de executar

- Confira a contagem de linhas e a distribuição de classes contra os números acima.
- Os caminhos dentro do `mapping.csv` são **relativos à raiz do repositório** — mover a pasta do
  dataset invalida o CSV. Regenere em vez de editar à mão (é determinístico dada a mesma config).
- Se este dataset vai para um estudo federado, a partição é gerada **uma vez** e congelada:
  `python -m src.dataset.federated_partition`. Ver a skill `federated-study`.

## Restrições

- **NÃO** rode um script de pré-processamento sem antes conferir `data.dataset`.
- **NÃO** monte caminhos de `data/` à mão — use `paths.processed_dir()`, `paths.mapping_file()`.
- **NÃO** edite `mapping.csv` manualmente.
- **NÃO** modifique os scripts de pré-processamento sem pedido explícito do usuário: eles definem
  artefatos já congelados.
