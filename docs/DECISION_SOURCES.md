# Registro de decisões de implementação e suas fontes (E5)

Cada decisão de implementação do pipeline federado tem uma fonte primária verificada ou está
marcada como decisão própria. Verificado em 2026-10-16 pelos textos primários (ar5iv/arXiv,
páginas dos proceedings e documentação oficial). Venues e DOIs foram conferidos à parte, no Crossref
e nas páginas dos proceedings. As chaves BibTeX estão em [`docs/references.bib`](references.bib).
Os métodos de conflito de gradiente (M1–M6) já foram verificados em
[`GRADIENT_CONFLICT_METHODS.md`](GRADIENT_CONFLICT_METHODS.md) e só são referenciados aqui.

**Status:**
- `verificado`: a afirmação ou fórmula foi lida no texto primário e o código a segue;
- `adaptação`: o código difere da fonte de propósito, e o motivo está escrito;
- `divergente`: o código difere da fonte sem justificativa registrada (encaminhamento abaixo);
- `decisão própria`: não há fonte; a justificativa está escrita;
- `sem fonte`: pendência, com encaminhamento.

A E5 **não altera código de treino nem de avaliação**. Toda correção de código vira braço ou tarefa da
E3/E4, para não invalidar os resultados já produzidos.

---

## Federação e agregação

| # | Decisão | Local | Fonte | Status | Observação |
|---|---|---|---|---|---|
| D01 | FedAvg ponderado por amostras (modo `flat`) | `src/federated/server.py::_aggregate_flat` | `mcmahan2017`, Alg. 1 (§2): `w ← Σ (n_k/m_t) w_k` | verificado | Com `task_weights` = 1 e `client_weighting: num_examples`, o `flat` é exatamente o FedAvg. |
| D02 | FedPer: trunk compartilhado, cabeças pessoais | `src/federated/model_split.py::shared_keys` | `arivazhagan2019`, §3.1–3.2 e §4.3; `collins2021` | adaptação | No FedPer, a parte pessoal é o classificador mais os últimos K_P−1 blocos, e a base é agregada com `n_j/Σn`. Aqui, o decoder inteiro, as saídas e o classificador são pessoais. Isso é o extremo "representação compartilhada + cabeça local" do FedRep, levado a uma U-Net. |
| D03 | Stem `encoder1` pessoal (`share_stem: False`) | `src/federated/model_split.py::shared_prefixes` | `li2021fedbn`, §1 e §4.1 (precedente parcial) | adaptação | O FedBN mantém locais só as camadas de BN, contra o *feature shift*. Aqui, o stem inteiro é local por necessidade: 1 canal (BUSI/SIIM) e 3 canais (ISIC/TCGA) não têm forma comum. A motivação de feature shift é a mesma. |
| D04 | Agregação hierárquica (normaliza dentro do dataset, depois `dataset_weights`) | `src/federated/server.py::_hierarchical_weights` | — | decisão própria | Impede que o dataset com mais clientes ou amostras domine o trunk (1:1 dá participação exata de 50/50). O `flat` é a ablação de referência (D01). O FedHCA² (`lu2024fedhca2`) também aprende pesos de agregação por cliente, mas é outro mecanismo. |
| D05 | Peso do cliente = `client_factor × Σ_t task_weight[t]·mass_t / n` | `src/federated/server.py::_task_factor` | — | decisão própria | Generaliza o peso por tarefa ao cliente `multi_task`. Num cliente `single_task`, reproduz exatamente o peso histórico. **E3 (2026-10-18):** `task_weights` fica em 4:1. O braço `agg_task_1to1` (1:1 na agregação, λ local fixo em 4:1) não mudou nenhuma célula dataset/tarefa pela regra da E3.6 (3 seeds, descritivo). |
| D06 | Partição Dirichlet por classe sobre os clientes | `src/dataset/federated_partition.py::dirichlet_partition` | `yurochkin2019`; `li2022niidbench`, §IV-B ("distribution-based label imbalance": `p_k ~ Dir_N(β)`, fração `p_{k,j}` da classe k ao cliente j) | verificado | **Não citar `hsu2019` para esta partição.** Em Hsu et al. (§3), a Dirichlet é sorteada **por cliente** sobre as classes (`q ~ Dir(αp)`, tamanho de cliente fixo). O código sorteia **por classe** sobre os clientes, que é a forma de Yurochkin/NIID-Bench. |
| D06b | Variante agrupada (grupo atômico, preenchimento guloso dos alvos) | `src/dataset/federated_partition.py::_grouped_dirichlet_partition` | — | adaptação | Mesma Dirichlet de D06, mas aloca um `lesion_id` ou paciente inteiro por vez. As proporções viram metas aproximadas. |
| D07 | Split agrupado por `lesion_id` (ISIC) e por paciente (TCGA-LGG) | `src/dataset/federated_partition.py::_grouped_dirichlet_partition`, `fold_strategy: stratified_group` | `tschandl2018ham10000` (o dataset publica `lesion_id`); `buda2019` (anotação por paciente) | decisão própria | Medido: 44.9% das imagens do Task 3 compartilham lesão (`data/ISIC_2018/PAPER_NOTES.md`). Dividir por imagem vazaria a lesão entre treino e teste. |
| D13 | Topologia `multi_task` (HC-FMTL) | `src/dataset/federated_partition.py::_build_multitask_dataset` | `lu2024fedhca2` (define HC-FMTL: clientes com conjuntos de tarefas diferentes); `smith2017mocha` (FMTL no sentido tarefa ≡ cliente) | verificado | O termo "Federated Multi-Task Learning" é ambíguo. O MOCHA trata cada nó como uma tarefa (personalização), e é isso que um revisor de FL entende. Desambiguar no texto. A contribuição é a instanciação clínica, não "1..N tarefas por cliente". |
| D25 | Orçamento local `epochs` vs `steps` | `src/federated/local_trainer.py` | `mcmahan2017`, §2 (`u_k = E·n_k/B`); `wang2020fednova`, §1 e §3 | verificado | `epochs` é o FedAvg original: o número de passos varia entre clientes. Pelo FedNova, a média ponderada de clientes com números de passos τ_i diferentes converge para um objetivo **inconsistente**. Isso afeta o modo `epochs` (τ_i ∝ n_i), não o `steps`. Registrado como entrada da E3. |
| D26 | Seed pessoal por dataset (`federated.personalized_init_seed: dataset`): todos os clientes de um dataset começam com o mesmo stem e as mesmas cabeças | `src/federated/config.py::personalized_init_seed`; `src/federated/client.py::initialization_seed` | — | decisão própria | **E0.1b** (30 rodadas, 1 seed): o cosseno intra-dataset entre deltas do trunk sobe de ≈ 0 (`client`) para 0.39–0.57 nas rodadas 1–5, e os stems continuam alinhados (cosseno ≥ 0.999 na rodada 30). **E3** (200 rodadas, 3 seeds, descritivo): único braço que melhora o controle sem piorar célula alguma (Δ positivo nas 8 células dataset/tarefa). A lacuna média para o local-only vai de −0.033 a −0.017; o M1 intra-dataset, de 0.003 (controle) a 0.098; e não há colapso D6 da seg do SIIM. O alinhamento decai até perto do ortogonal no último terço do treino. No `src/config.yaml` desde `e66f575`; o fallback do código continua `client`, para os runs históricos resolverem como foram produzidos. O local-only lê a mesma chave, então o pareamento entre braços se mantém. |

## Loss e supervisão

| # | Decisão | Local | Fonte | Status | Observação |
|---|---|---|---|---|---|
| D08 | DiceBCE 0.5/0.5 sobre logits | `src/utils/segmentation_loss.py::DiceBCELoss` | `milletari2016`, §3 (Dice com denominador `Σp² + Σg²`); `isensee2021`, Methods (Dice + CE, "simply averaged" no Ap. B.1) | verificado | `squared_pred=True` segue o V-Net. A média 0.5/0.5 segue o nnU-Net, com BCE no lugar da CE porque a segmentação é binária com um canal. O smoothing 1/1 do MONAI e o Dice por amostra (`batch=False`) são decisões próprias. **E3 (2026-10-18):** o protocolo adota o Batch Dice (`loss.dice_batch: true`), decisão própria apoiada na evidência da E3, sem fonte verificada. Na Fase A, foi a única variante a passar o gate do SIIM (Dice positivo de validação 0.262, contra 0.003 do Dice por amostra, que aprende e desaprende com ~78% de imagens vazias). Na Fase B (3 seeds, Batch Dice em todos os braços), o Dice positivo de teste do SIIM fica entre 0.150 e 0.219 por braço. O colapso D6 que resta atinge um cliente por run e some com o `init_seed_dataset` (D26). A chave fica nos manifests e configs de run, não no `src/config.yaml` (protegido por `StudyHashGuardTests`); sem ela, o código continua com `batch=False`. |
| D09 | Focal loss multiclasse | `src/utils/criterions.py::FocalLoss`, escolhida em `src/utils/experiment_init.py::init_criterion_classification` | `lin2017`, §3.1–3.2 (Eq. 5: `−α_t(1−p_t)^γ log p_t`, com `p_t` independente de α) | **divergente** | Quando há `class_weights`, `pt = exp(−CE)` é calculado sobre a CE **ponderada**, então `pt = p_y^{w_y}` e o fator modulador fica distorcido. Em Lin, o peso α_t multiplica por fora. Sem pesos, a implementação coincide com Lin (α = 1, γ = 2). **Alcance:** só quando `classification_criterion: Focal` e há pesos de classe. O manifest atual usa `CE` em todos os datasets federados, então os resultados federados vigentes não são afetados. Encaminhamento em Div-2. |
| D10 | Deep supervision com pesos `1/(n+1)` | `src/utils/supervision.py::_seg_loss` | `isensee2021`, Methods ("Training schedule") e Ap. B.2 | **divergente** | No nnU-Net, os pesos caem pela metade a cada redução de resolução, são normalizados para somar 1 e as **duas menores resoluções não recebem loss**. Aqui, `reversed([out4..out1])` dá pesos 1, ½, ⅓, ¼ (da resolução cheia à menor), sem normalização e nas 4 saídas. Além disso, as saídas auxiliares são reamostradas para a resolução cheia dentro do modelo, enquanto o nnU-Net reduz o alvo. Herdado do código centralizado (Aumente-Maestro). Encaminhamento em Div-1. |
| D11 | Supervisão parcial por máscara de amostra (`has_mask`/`has_label`; termo ausente é omitido, não somado como zero) | `src/utils/supervision.py::_combined_loss` | `wang2023condistfl` (loss marginal para classes não anotadas + destilação condicional); `jiang2024ufps` | adaptação | Os dois tratam anotação parcial **dentro da mesma cabeça de segmentação**, por isso precisam de loss marginal ou destilação. Aqui, seg e cls ocupam cabeças disjuntas, e basta omitir o termo da tarefa sem rótulo. É uma simplificação estrita, a declarar como tal. |
| D12 | λ fixos entre tarefas (`task_weights` normalizados) | `src/utils/supervision.py::resolve_task_lambdas` | `caruana1997` (MTL com soma de losses); `kendall2018` como alternativa adaptativa | decisão própria | Escolha e ablação são da E3. **E3 (2026-10-18):** fica 4:1, derivado de `aggregation.task_weights` (`loss.task_lambdas` ausente). O braço `lambda_1to1` (0.5/0.5) sobe a cls do SIIM (+0.062 de balanced accuracy) e derruba a seg do SIIM (−0.045 de Dice positivo, colapso D6 de um cliente nas 3 seeds) e a cls do BUSI (−0.064) e do ISIC (−0.019). Tem a pior lacuna média para o local-only (−0.042 contra −0.033 do controle). 3 seeds, descritivo. |
| D12b | Ponderação por incerteza (`loss.task_weighting: uncertainty`, braço `kendall` da E3) | `src/utils/supervision.py::uncertainty_weighted_loss` | `kendall2018`, §3.2, Eq. 10; `s = log σ²` (fim da §3.2) | adaptação | Forma de **classificação** da Eq. 10 (`e^{−s}L + s/2`) para as duas tarefas, porque BCE e CE são verossimilhanças de classificação. O termo Dice não é verossimilhança, então aplicá-lo à seg é extensão nossa. O λ fixo multiplica `e^{−s}`; no ótimo de `s`, o λ se cancela (peso efetivo `1/(2L)`). `s_t` é pessoal (nunca agregado), começa em 0 e só existe em clientes com ≥ 2 tarefas. A loss de validação continua em λ fixo, para comparar entre braços. Verificado em 2026-10-16. **E3 (2026-10-18): rejeitado.** Nenhuma célula dataset/tarefa melhora contra o controle pela regra da E3.6, e M1 e cancelamento ficam iguais aos do controle (3 seeds, descritivo). O código fica, desligado: `fixed` é o padrão. |
| D16 | Pesos de classe `balanced_fold`: `N / (K·n_c)` | `src/dataset/federated_dataloader.py::_balanced_weights` | Heurística `balanced` do scikit-learn (`compute_class_weight`), "inspirada" em `king2001` | verificado | É idêntica à fórmula do scikit-learn. A fonte é a heurística da biblioteca, não uma fórmula de King & Zeng. Citar como tal. A classe ausente num cliente recebe 0 (decisão própria, documentada no código). |
| D17 | Oversampling determinístico por `round(1/p_c)`; recusado em `multi_task` | `src/dataset/BUSI_dataloader.py::deterministic_oversampling` | Herdado do código centralizado (Aumente-Maestro et al., `aumente2025`) | sem fonte | O texto do artigo não foi verificado. Peculiaridade: uma classe com fator 1 (p > 2/3) também recebe 1 cópia extra. O equilíbrio é aproximado, não exato. Não afeta os braços `multi_task`, onde a função é recusada. Encaminhamento em Div-4. |

## Diagnóstico, avaliação e estatística

| # | Decisão | Local | Fonte | Status | Observação |
|---|---|---|---|---|---|
| D14 | Conflito: cosseno entre deltas de clientes (`cos < 0` ⇒ conflito) | `src/federated/conflict_diagnostics.py`; Gram em `src/federated/server.py` | `chen2024fedbone`, Def. 1; `yu2020pcgrad` (M1 em `GRADIENT_CONFLICT_METHODS.md`) | verificado | O FedBone define conflito pelo ângulo entre gradientes de clientes (`cos ω_ij < 0`), mas **não reporta métrica de conflito por rodada**: só tabelas de desempenho e ablação. A curva por rodada é contribuição nossa. |
| D15 | Razão de cancelamento `1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖`, decomposta em intra/inter, com referência ortogonal | `src/federated/negative_transfer.py::cancellation_from_gram` | `yin2018diversity`, Def. 4, §4.1 (parente: `Δ_S = Σ‖∇fᵢ‖² / ‖Σ∇fᵢ‖²`) | adaptação | A diversidade de gradiente usa normas **ao quadrado** e soma sem pesos. A razão daqui usa normas simples, os pesos reais da agregação e fica em [0, 1]. Com pesos uniformes, `‖Σ‖²/Σ‖·‖² = 1/Δ_S` é a forma quadrática equivalente. O nome "transferência negativa" vem do uso informal da literatura de MTL, e a definição geométrica é nossa. |
| D18 | Wilcoxon pareado só exploratório; holdout descritivo | `src/experiments/analyze.py::_paired_inference` | `wilcoxon1945`; `demsar2006` (Wilcoxon para 2 classificadores sobre **datasets independentes**); `dietterich1998` | verificado | Demšar recomenda o Wilcoxon quando as unidades pareadas são independentes. Os pares cliente × fold violam isso (folds de CV sobrepostos e clientes sintéticos do mesmo dataset), o que justifica o rótulo `exploratory_only_non_independent_client_fold_pairs`. Para Dietterich, só a venue foi conferida nesta sessão. |
| D19 | Métricas: Dice positivo, taxa de FP em alvos vazios, balanced accuracy, macro-F1, AUC OvR-macro | `src/federated/unified_eval.py` | `maier-hein2024metrics` (venue conferida); `codella2019isic` (o ISIC 2018 adotou balanced accuracy na classificação) | sem fonte | Recomendações do Metrics Reloaded para referências vazias não lidas nesta sessão. Encaminhamento em Div-5. |
| D20 | InstanceNorm (sem BatchNorm) | `src/models/multitask/MTnnUNet.py` | `ulyanov2016`; `hsieh2020quagmire` (BN é a mais prejudicada em non-IID, pela divergência das estatísticas de minibatch; propõe GroupNorm) | verificado | A IN não guarda estatísticas acumuladas entre clientes. Evita o problema descrito por Hsieh et al. sem precisar do FedBN. |

## Pré-processamento por dataset

| # | Decisão | Local | Fonte | Status | Observação |
|---|---|---|---|---|---|
| D21 | BUSI redimensionado com `INTER_NEAREST` | `src/dataset/Curated_BUSI_preprocessing.py` | `aldhabyani2020` (dataset); documentação do OpenCV | adaptação (congelada) | A doc do `cv2.resize` (4.7.0) recomenda `INTER_AREA` para reduzir. O nearest é herdado da curadoria e fica **congelado**: mudar invalidaria a partição e todos os resultados. Não "corrigir". |
| D22 | ISIC, TCGA-LGG e SIIM-ACR com `INTER_AREA` | `src/dataset/{ISIC_2018,TCGA_LGG,SIIM_ACR}_preprocessing.py` | Documentação do OpenCV `cv2.resize`: "To shrink an image, it will generally look best with INTER_AREA" | verificado | — |
| D23 | Máscaras com nearest e re-binarizadas | idem | — | decisão própria | Preserva os rótulos binários; interpolar produziria valores fracionários. |
| D24 | Classe derivada da máscara (TCGA-LGG: `tumor`; SIIM-ACR: `pneumothorax`) | `src/dataset/{TCGA_LGG,SIIM_ACR}_preprocessing.py` | `buda2019`; competição SIIM-ACR (Kaggle 2019) | decisão própria | A classe é função determinística do alvo de segmentação. Registrado em `data/*/PAPER_NOTES.md`. Ao interpretar o ganho de classificação nesses datasets, lembrar que ele não é independente da segmentação. |

---

## Divergências e encaminhamento

| ID | Decisão | Encaminhamento |
|---|---|---|
| Div-1 | D10, deep supervision ≠ nnU-Net | **E3**: braço `ds_nnunet` (pesos `1/2^i` normalizados, sem as 2 menores saídas) contra o atual. O padrão **não muda**, porque todos os resultados vigentes usam `1/(n+1)`. No texto, descrever como "deep supervision com pesos inversamente proporcionais à profundidade", sem atribuir ao nnU-Net. |
| Div-2 | D09, Focal com pesos distorce `p_t` | **E3**, antes de qualquer braço Focal: corrigir para `p_t` da CE não ponderada e peso por fora (`α_t = w_y`), num PR próprio com teste. Os resultados federados e os baselines B0 usam CE e não são afetados: `training_baseline.py` lê `classification_criterion` do dataset (`CE` nos 4). `loss.classification_criterion: Focal` só alcança o treino clássico (`classic_dataloader.py`). |
| Div-3 | Citação da partição (D06) | Na documentação, citar `yurochkin2019`/`li2022niidbench`, não `hsu2019`. Nenhum código muda. |
| Div-4 | D17, origem do oversampling não verificada | Conferir no artigo ou código de Aumente-Maestro 2025, se for citado. Fora dos braços `multi_task`. |
| Div-5 | D19, métricas sem leitura de fonte | Ler as recomendações do Metrics Reloaded para referências vazias e AUC multiclasse antes de escrever os métodos do artigo. |
| Div-6 | Documentação desatualizada | `docs/RESEARCH_NOTES.md` descreve o sistema dos runs históricos BUSI-only (`encoder1..5` compartilhado, `normal` fora da seg, peso `num_examples × task_weight`). Isso era verdade para aqueles runs, então o texto foi **mantido**, com uma nota datada no topo apontando as diferenças atuais e este registro. O `AGENTS.md` ("Loss and training loop") dizia só "Focal loss for multiclass", que vale para o caminho clássico; foi completado com o critério por dataset dos caminhos federado e baseline. **Feito na E5** (só documentação). |

## Como a literatura reporta cancelamento e transferência negativa (`referencies/To-do.md`)

- **FedBone:** define conflito como `cos < 0` entre gradientes de clientes (Def. 1) e o remove por
  projeção na agregação (GPAggregation). Não reporta métrica de conflito por rodada.
- **PCGrad:** `cos < 0` entre gradientes de tarefas como definição de conflito (M1, já verificado).
- **Gradient diversity (Yin et al. 2018):** `Σ‖gᵢ‖² / ‖Σgᵢ‖²` (Def. 4). É a forma quadrática, sem
  pesos, da nossa razão (D15).
- **Conclusão para o artigo:** reportar a curva por rodada da razão de cancelamento ao lado da
  referência ortogonal, sem limiares qualitativos. Nenhuma fonte verificada oferece limiar, e a
  métrica é nossa (adaptação de Yin et al. ponderada pela agregação real).
- **Ainda em aberto** (não verificados nesta sessão): GradNorm/CAGrad/Nash-MTL e a ligação entre
  perda de desempenho em relação ao local-only e medidas geométricas. Ficam como leitura da E3,
  que é onde métodos adaptativos entram.
