# Necessidades de pré-processamento por dataset (E4)

Este documento registra o que cada dataset pede do pré-processamento e por que a E4 muda **só a
resolução**, e só em dois datasets: TCGA-LGG e SIIM-ACR passam a 256 px; Curated BUSI e ISIC 2018
ficam em 128 px. A decisão, a execução e os pendentes estão em `plano.md`, seção E4.

Os números de lesão vêm do `mapping.csv` de cada variante e são regerados, só com leitura, por:

```bash
python -m scripts.preprocessing_needs --variants processed_128 processed_256
```

O script localiza cada `mapping.csv` por `src/dataset/paths.py` e nunca abre uma imagem: o mapping
já guarda, por linha, as estatísticas da máscara gravada (`tumor_pixels` e a caixa delimitadora).

## Por que a resolução pode variar por dataset

- **Stems pessoais.** O `encoder1` de cada cliente é pessoal (`share_stem: false`), e o trunk
  compartilhado (`encoder2..5` + `bottleneck`) é convolucional: os pesos não dependem do tamanho
  da entrada.
- **MTnnUNet totalmente convolucional.** Só exige H e W divisíveis por 32 (5 níveis). A 256 px o
  bottleneck fica com 8×8 em vez de 4×4. A cabeça de classificação termina em
  `AdaptiveAvgPool2d`, então também não depende do tamanho.
- **Cada dataset lê a própria variante.** No pipeline multi-dataset, a variante vem de
  `datasets.<nome>.variant`, e o master copia `img_path`/`mask_path` de cada `mapping.csv`. Um run
  pode, portanto, misturar resoluções sem código novo.
- **A normalização de intensidade fica de fora.** No MTnnUNet, toda conv é sem bias e seguida de
  `InstanceNorm2d` sem afim; uma mudança afim por imagem na entrada (`a·x + b`) é quase anulada já
  no primeiro bloco (sobram efeitos de borda do zero-padding e o `eps`). Normalizar seria quase um
  no-op e confundiria a leitura da E4.

## Tabela por dataset

| | Curated BUSI | ISIC 2018 | TCGA-LGG | SIIM-ACR |
|---|---|---|---|---|
| Modalidade | ultrassom de mama | dermatoscopia | RM de cérebro (3 sequências) | radiografia de tórax |
| Bruto | 512×512 (as 450 imagens curadas, medido) | Task 1: tamanhos variáveis (76 tamanhos numa amostra de 400; ex. 767×576, 2048×1536, 3072×2304); Task 3: 600×450 | 256×256 (uniforme) | 1024×1024 (uniforme, DICOM 8 bits) |
| Canais | 1 | 3 (RGB) | 3 (pré-contraste, FLAIR, pós-contraste) | 1 |
| Imagem | `INTER_NEAREST` (congelado, D21) | `INTER_AREA`, sem preservar aspecto (achatado) | `INTER_AREA`; a 256 é a identidade | `INTER_AREA` |
| Máscara | mesmo resize da imagem (nearest) | nearest + re-binarização | nearest + re-binarização; a 256 é a identidade | RLE decodificado a 1024, nearest + re-binarização |
| Linhas com máscara | 450 | 3.694 (só Task 1) | 3.929 | 12.044 |
| Máscaras vazias | 14,2% | 0% | 65,1% | 77,9% |
| **Variante na E4** | **`processed_128`** (intocada) | **`processed_128`** | **`processed_256`** (nativa) | **`processed_256`** |

## Estatísticas de lesão a 128 e a 256 px

Só máscaras positivas. "Área" em pixels do quadro gravado; "% do quadro" é a área sobre `dim1·dim2`;
"lado mín. p5" é o percentil 5 do lado menor da caixa delimitadora; "< 50 px" conta lesões com
menos de 50 pixels.

| Dataset | Variante | Positivas | Área mediana | Área p5 | % do quadro (mediana) | Lado mín. p5 | < 50 px |
|---|---|---|---|---|---|---|---|
| BUSI | `processed_128` | 386 | 1.086 | 136 | 6,63% | 11 | 1 (0,3%) |
| ISIC (seg) | `processed_128` | 3.694 | 2.697 | 297 | 16,46% | 17 | 1 (0,0%) |
| TCGA-LGG | `processed_128` | 1.373 | 381 | 45 | 2,33% | 8 | 78 (5,7%) |
| TCGA-LGG | `processed_256` | 1.373 | 1.529 | 184 | 2,33% | 15 | 13 (0,9%) |
| SIIM-ACR | `processed_128` | 2.667 | 134 | 21 | 0,82% | 5 | 474 (17,8%) |
| SIIM-ACR | `processed_256` | 2.667 | 541 | 85 | 0,83% | 10 | 50 (1,9%) |

- A 256 px, a área de cada lesão é ~4× a de 128 (razão mediana 4,0 nos dois datasets; p5–p95
  3,90–4,08 no TCGA e 3,81–4,21 no SIIM). O "% do quadro" não muda: a resolução não aumenta a
  lesão em relação à imagem, só o número de pixels que a descrevem.
- Lesões com lado menor da caixa < 4 px: TCGA 13 → 1, SIIM 51 → 5. Com < 10 px de área: TCGA
  6 → 1, SIIM 29 → 2.
- **Nenhuma lesão some** no resize, em nenhuma das duas resoluções: as contagens de positivas são
  iguais a 128, a 256 e (no SIIM) a 1024, e os logs de geração não têm nenhum `became empty`.

## Justificativa de cada escolha

- **BUSI → 128 (intocado).** A lesão mediana ocupa 6,6% do quadro e só 1 lesão tem < 50 px a
  128 px. Mudar o BUSI exigiria refazer a variante curada com `INTER_NEAREST` (D21), o que invalida
  todos os masters congelados. Não há ganho que pague isso.
- **ISIC → 128.** A lesão mediana ocupa 16,5% do quadro; é o dataset que menos depende de detalhe
  fino. O achatamento (squash) da Task 1 continua (ver "Problemas conhecidos"). ISIC a 256 fica como
  pendência: custaria ~4× o tempo dos 8 clientes ISIC, os maiores do roster.
- **TCGA-LGG → 256 (resolução nativa).** O bruto já é 256×256, então a variante a 256 é o próprio
  dado sem perda (conferido em 300 fatias: imagem e máscara idênticas ao TIFF). A 128 px, 78 lesões
  têm < 50 px e 65% das fatias são vazias: a lesão pequena é o caso difícil.
- **SIIM-ACR → 256.** É o dataset de lesão mais fina: mediana de 0,82% do quadro e 474 lesões com
  < 50 px a 128 px (17,8% das positivas). 512 px descreveria melhor a lesão, mas não cabe com
  batch 32 em 16 GB (decisão do grill de 2026-10-18); 256 reduz as lesões < 50 px de 474 para 50.
- **Partição.** Um master novo foi autorizado, com uma exigência: ele precisa ser o **mesmo** master
  da E3 (seed 1993) lido em outra resolução. `scripts/partition_equivalence.py` confere linhas,
  ordem, clientes, splits, ids, classes e `tumor_pixels > 0` (ver `plano.md`, E4).

## Problemas conhecidos (registrados, não corrigidos na E4)

- **Máscara do BUSI com valor 254.** `benign_id_181_mask.png` (`processed_128`) tem 1 pixel 254
  além dos 255. O loader (`BUSI_dataset.py::_load_mask`) só mapeia 255 → 1, então esse pixel chega à
  loss com valor 254; e `tumor_pixels` só conta 255. É a única máscara não binária do BUSI.
- **`transforms.rotation` é ignorado.** `client.py::_default_transforms` lê `horizontal_flip` e
  `vertical_flip`, mas aplica sempre `RandomRotation(degrees=360)`, qualquer que seja o valor da
  chave. Rotação de 360° em radiografia de tórax e em RM axial é anatomicamente implausível.
- **`assert_variant_resolution` só olha o tamanho.** Ele impede escrever outra resolução numa
  variante existente, mas não guarda interpolação nem preservação de aspecto: uma variante refeita
  com outro filtro e o mesmo tamanho passaria. Para variantes novas, a CLI agora exige uma pasta
  nova (`assert_new_variant`).
- **ISIC achatado.** `PRESERVE_ASPECT = False` no ISIC: as imagens da Task 1 (aspectos variados) e
  as da Task 3 (600×450, 4:3) são esticadas para o quadrado.
- **Dice a 256 é medido em outro raster de GT.** O Dice de teste do TCGA e do SIIM na E4 é calculado
  contra a máscara a 256 px, não contra a de 128 px da E3. São alvos diferentes da mesma lesão
  (sobretudo nas lesões pequenas), então comparar o Dice da E4 com o da E3 mistura modelo e alvo.
  É uma ressalva de leitura, não um defeito de código.
- **Sem controle de orçamento.** O run da E4 tem 400 rodadas; os da E3, 200. Não existe controle a
  400 rodadas com 128 px.

## Reproduzir

```bash
# variantes (cada uma aborta se a pasta já existir com conteúdo)
python -m src.dataset.TCGA_LGG_preprocessing --variant processed_256 --image-size 256
python -m src.dataset.SIIM_ACR_preprocessing --variant processed_256 --image-size 256
# estatísticas (só leitura)
python -m scripts.preprocessing_needs --variants processed_128 processed_256
```

O master da E4 e a verificação de equivalência estão em `plano.md`, E4.3.
