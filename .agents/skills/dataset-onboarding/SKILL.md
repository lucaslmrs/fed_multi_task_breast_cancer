---
name: dataset-onboarding
description: Integrar um dataset NOVO ao repositório, escrevendo o script de pré-processamento que o converte para o contrato do projeto (pasta da variante com images/, masks/ e mapping.csv). Use ao adicionar uma base de imagens ainda sem suporte, ao escrever um novo script *_preprocessing.py, ou ao decidir como mapear rótulos e máscaras de origem para o esquema do projeto. Gatilhos - "adicionar dataset", "novo dataset", "integrar base", "converter dataset", "suportar outro dataset", "escrever preprocessing para".
---

# Integrar um dataset novo

Para rodar ou depurar um dataset **já suportado**, use `dataset-preprocessing`.

Seu objetivo: produzir uma pasta que o treino consuma **sem nenhuma modificação no código de
treino**. Se você precisou tocar em `federated_dataloader.py` para o dataset funcionar, o script
de conversão ainda não terminou — ou o dataset expõe um caso genuinamente novo, e aí isso precisa
ser discutido com o usuário antes de codar.

## O contrato de saída

```
data/<Nome_Do_Dataset>/
  raw/           # download original extraído — só o script de pré-processamento lê
  archives/      # .zip originais
  <variant>/     # o que o treino lê
    images/
    masks/
    mapping.csv
  federated/     # federated_mapping.csv, se o dataset entrar num estudo federado
```

`<variant>` é nomeada pela resolução (`processed_128` para `data.image_size: 128`). **Nunca**
codifique esses caminhos: derive-os de `src/dataset/paths.py`, que os monta a partir de
`data.root` / `data.dataset` / `data.variant`.

## Esquema do `mapping.csv`

Colunas base obrigatórias:

| Coluna | Conteúdo |
|---|---|
| `img_path`, `mask_path` | caminhos **relativos à raiz do repositório** |
| `class` | rótulo em string; vazio se a linha não tem rótulo |
| `id` | identificador numérico único da imagem |
| `dim1`, `dim2` | altura e largura pré-processadas |
| `tumor_pixels` | pixels de lesão na máscara redimensionada |
| `y_min`, `y_max`, `x_min`, `x_max` | bounding box da lesão |
| `y_size`, `x_size` | altura e largura da bounding box |

Tudo de `tumor_pixels` em diante sai pronto de `preprocessing_utils.add_image_metadata()` —
não reimplemente esses cálculos.

**Se o dataset não fornece as duas supervisões na mesma imagem**, siga o precedente do ISIC 2018 e
acrescente:

| Coluna | Conteúdo |
|---|---|
| `task` | `seg` (tem máscara, `class` vazio) ou `cls` (tem rótulo, `mask_path` vazio) |
| `official_split` | `train`/`val`/`test` do release original, se houver |
| `lesion_id` | agrupa imagens do mesmo objeto físico — **divida por ele, não por `id`** |

Colunas derivadas de máscara ficam `NaN` (não `0`) onde não há máscara. `0` é um valor legítimo
("máscara vazia", caso `normal` do BUSI), então só `NaN` expressa "não se aplica".

Um dataset com `task` **não alimenta `training_centralized.py`** (que exige imagem+máscara+rótulo
na mesma linha), mas serve perfeitamente ao FedPer, onde cada cliente cuida de uma tarefa só.

## Como escrever o script

Comece de `src/dataset/ISIC_2018_preprocessing.py` se o dataset for multi-tarefa ou colorido, e de
`src/dataset/Curated_BUSI_preprocessing.py` se toda imagem tiver máscara e rótulo.

Reuse de `src/dataset/preprocessing_utils.py`:

| Helper | Para quê |
|---|---|
| `assert_target_dataset(cfg, "<Nome>")` | **obrigatório** na primeira linha de `main()`; impede sobrescrever a variante de outro dataset |
| `assert_variant_resolution(mapping_path, size)` | impede misturar resoluções dentro de uma variante |
| `resize_image(img, size, interpolation, preserve_aspect)` | imagens |
| `resize_mask(mask, size, preserve_aspect)` | máscaras — vizinho mais próximo e re-binarização em 0/255 |
| `add_image_metadata(df)` | preenche todas as colunas derivadas de máscara |
| `size_tumor(seg)` | bounding box, se precisar isolado |
| `pad_to_square(img)` | quando `preserve_aspect` importa |

Regras de conversão:

1. **Máscaras múltiplas** do mesmo caso: una-as e garanta saída binária em 0/255.
2. **Interpolação**: imagens com `INTER_AREA` (reduzir) — exceto onde um artefato congelado já
   depende de outra escolha; máscaras **sempre** por `resize_mask`, nunca interpolação suave.
3. **Canais**: grave em escala de cinza para modalidades 1-canal e RGB para 3-canais. Registre a
   contagem em `datasets.<nome>.channels` no `config.yaml` — o FedPer usa isso para decidir o stem.
4. **IDs**: garanta unicidade global e **assevere isso no código**, não deduza de intervalos
   mín./máx. (no ISIC os blocos se intercalam e o intervalo engana).
5. **Normalização** não é problema seu: o dataloader aplica transformações. Aqui é só o
   pré-processamento estático.

## Verificação antes de declarar pronto

1. `python -m src.dataset.<Nome>_preprocessing` roda limpo a partir de um `raw/` real.
2. Conte as linhas e a distribuição de classes; compare com o que o release original documenta.
3. Todo `img_path`/`mask_path` não-vazio existe em disco.
4. Instancie o loader do projeto (`src/dataset/federated_dataloader.py`, ou
   `src/dataset/BUSI_dataloader.py` no caminho legado) apontando para a nova variante e confirme
   que um batch sai com o shape esperado.
5. `python -m unittest discover -v` continua verde.
6. Documente os fatos medidos (contagens, sobreposições, desbalanceamento) num
   `data/<Nome>/PAPER_NOTES.md`, como o ISIC faz — números medidos, não estimados.

## Restrições

- **NÃO** modifique os scripts de pré-processamento existentes ao adicionar um novo.
- **NÃO** monte caminhos de `data/` à mão.
- **NÃO** gere `federated_mapping.csv` como parte do onboarding — a partição é um passo separado,
  gerado uma vez e congelado (ver `federated-study`).
- **NÃO** invente valores para colunas que o dataset de origem não fornece: deixe vazio/`NaN`.
