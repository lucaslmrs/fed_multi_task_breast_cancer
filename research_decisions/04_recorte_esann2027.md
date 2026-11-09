# Recorte ESANN 2027 — do problema à metodologia

Documento vivo, criado em 2026-11-09. Fecha o que precisa estar decidido **antes** dos treinos do
artigo. O artigo é o núcleo da dissertação (FedCHAMP); a reformulação do título e da pergunta do
pré-projeto, que ainda falam em protótipos, fica para depois, com os orientadores.

| Item | Valor |
|---|---|
| Alvo | ESANN 2027 (Bruges, híbrido), 6 páginas |
| Submissão | **18/11/2026** (decisão em 22/01/2027; congresso de 21 a 23/04/2027) |
| Base experimental | A da E8: MTResNet34UNet (ImageNet), conjunto amplo da E6, TCGA/SIIM a 256 px, 400 rodadas |
| Seeds | **Uma (1993)**, declarada como limitação; seeds extras na versão final e na dissertação |
| GPU disponível | ~2 semanas; o plano usa ~2,5 dias |

## 1. Problema

Em aprendizado federado multitarefa (FMTL) entre domínios médicos, clientes de modalidades e
tarefas diferentes atualizam o mesmo tronco compartilhado. A literatura atribui a perda de
desempenho do federado frente ao treino local ao **conflito** entre esses updates. Nas nossas
medidas (E0, E8), os updates são quase **ortogonais**, não opostos: a média ponderada encolhe o passo
de cada cliente (**diluição**), sem cancelamento acima do esperado para vetores ortogonais.

## 2. Lacuna

Da síntese em `research_problem/negative_transfer/README.md` (§8):

1. Nenhum trabalho de FMTL **mede** o conflito entre clientes: FedHCA2 só o invoca na teoria,
   FedBone o define sem medir, OmniFM o afirma sem medir, FMTL-Bench não o mede.
2. Nenhum trabalho **separa** conflito de direção e diluição de magnitude.
3. Nenhum trabalho mistura **segmentação e classificação de modalidades diferentes** na mesma
   federação (OmniFM: uma tarefa por federação; FedBone: um domínio de imagem).
4. No FMTL-Bench, entre domínios, nenhum método federado supera o Local de forma relevante.
5. Falta rigor estatístico (várias seeds) na maior parte dos trabalhos — inclusive neste artigo,
   o que precisa ser dito.

## 3. Pergunta de pesquisa

> Num FedPer multitarefa entre quatro modalidades médicas, o custo da agregação é conflito de
> direção ou diluição de magnitude? A federação supera o treino local, e esse ganho depende de
> aprender as duas tarefas juntas? Corrigir a escala do agregado amplia o ganho?

## 4. Hipóteses e critério de decisão

Fixados **antes** dos runs novos. Uma seed: tudo é descritivo; nenhum p-valor é chamado de
significativo.

**Limiar de empate**, tirado do ruído entre seeds da E3 (3 seeds, MTnnUNet a 128 px; é a única
estimativa disponível e vem de outra base): o desvio-padrão da média das 8 células entre seeds ficou
entre 0,001 e 0,008 por braço; o desvio mediano por célula, entre 0,015 e 0,028. Daí:

- média das 8 células: |Δ| < **0,02** → empate;
- por célula: |Δ| < **0,03** → empate.

| # | Hipótese | Teste | Estado |
|---|---|---|---|
| H1 | O custo da agregação é diluição, não conflito | Cancelamento do controle ≤ referência ortogonal | Observado na seed 1993 (0,701 × 0,710) |
| H2 | A federação multitarefa supera o local-only | Fed-MTL − Local-MTL > 0,02 na média | Observado (+0,046; 18/24 pares) |
| H3 | A federação converge mais rápido | Rodadas até 95% do valor final de validação, fed × local | A medir (ISIC cls: fed estabiliza perto da rodada 300, local ainda sobe na 400) |
| H4 | O multitarefa ajuda dentro da federação | Fed-MTL × Fed-seg e Fed-MTL × Fed-cls, por célula | Runs novos |
| H5 | A federação ajuda também sem multitarefa | Fed-seg × Local-seg e Fed-cls × Local-cls | Runs novos |
| H6 | Corrigir a escala do agregado amplia o ganho | FedExP: o passo efetivo sobe **e** a média supera o controle por > 0,02 | Run novo |

Se H6 der empate, ele entra como resultado: a diluição existe mas não é o gargalo nesta base.

## 5. Definições operacionais

- **Célula:** um par dataset × tarefa (8 células). Métrica primária: `dice_positive` na seg,
  acurácia balanceada na cls, média dos clientes do dataset.
- **Transferência negativa:** federado − local-only < 0 numa célula, sob o mesmo algoritmo e o
  mesmo orçamento (NTG de Wang et al., 2019).
- **Cancelamento:** `1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖` sobre o conjunto compartilhado, com os pesos reais da
  agregação, decomposto em intra e inter-dataset e lido contra a referência ortogonal (D15).
- **Passo efetivo:** `η_g · ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖`; no FedAvg (η_g = 1) é `1 − cancelamento`.
- **Orçamento:** 10 passos locais por rodada × 400 rodadas, igual em todos os braços. Todo ganho
  é afirmado "sob orçamento fixo".
- **Secundárias:** Dice global, `empty_fp_image_rate`, macro-F1, AUC.

## 6. Contribuições

1. Um instrumento que separa conflito de diluição na agregação federada, por rodada e por bloco.
2. Evidência, em 4 modalidades e 2 tarefas, de que o custo da agregação é diluição, e de que mesmo
   assim o FMTL supera o treino local sob orçamento fixo, contra o padrão do FMTL-Bench.
3. Uma ablação 2×2 (tarefa única × multitarefa, local × federado) que separa o ganho da federação
   do ganho do multitarefa.
4. Um teste de intervenção (FedExP) que trata a diluição como causa, não só como correlação.

## 7. Método

- **FedPer hierárquico:** stem e cabeças pessoais; conjunto compartilhado de 17 módulos; peso do
  cliente `client_factor × Σ_t task_weight[t]·mass_t / n`, normalizado por dataset e depois por
  `dataset_weights` (1:1:1:1).
- **FedExP** (Jhunjhunwala et al., 2023), com os pesos reais da agregação:
  `η_g = max(1, Σ wᵢ‖Δᵢ‖² / (2(‖Σ wᵢΔᵢ‖² + ε)))` e `θ ← θ + η_g · Σ wᵢΔᵢ`, só no conjunto
  compartilhado; η_g é gravado por rodada. Com pesos uniformes (wᵢ = 1/M) é exatamente a fórmula
  do artigo, `Σ‖Δᵢ‖² / (2M(‖Δ̄‖² + ε))`.
- **η_g esperado na base da E8**, estimado dos cossenos gravados no controle supondo normas iguais
  (as normas por cliente não são gravadas): ~2,6 nas primeiras rodadas e **~5,6 da rodada 50 em
  diante**. O passo aplicado fica ~1,7× o passo médio de um cliente, contra ~0,3× no FedAvg.
  Normas desiguais só aumentam η_g.
- **Decisões de implementação a fechar:**
  - artefato final: o artigo do FedExP avalia a média dos dois últimos iterados globais. Proposta:
    seguir o artigo, o que exige guardar o iterado anterior também no checkpoint de resume;
  - sem teto para η_g (seguir o artigo); um teto entra só se o smoke divergir;
  - `ε` como no artigo.
- **Tarefa única:** um filtro de tarefa ativa na leitura do master (sem regenerar a partição). O
  oversampling e o conjunto compartilhado ficam iguais aos do controle; os blocos que a tarefa
  ausente não alcança recebem delta zero.

## 8. Braços

Todos com seed 1993, o master da E4 e o mesmo orçamento.

| # | Braço | Setup | Tarefas | Estado |
|---|---|---|---|---|
| A1 | `resnet34_256_local` | local-only | seg + cls | ✅ existe |
| A2 | `resnet34_256_control` | federado (FedAvg) | seg + cls | ✅ existe |
| A3 | `resnet34_256_layer_mixing` | federado + mistura por camada | seg + cls | ✅ existe (secundário) |
| A4 | FedExP | federado + FedExP | seg + cls | novo |
| A5 | Fed-seg | federado | só seg | novo |
| A6 | Fed-cls | federado | só cls | novo |
| A7 | Local-seg | local-only | só seg | novo |
| A8 | Local-cls | local-only | só cls | novo |

## 9. Análise

- **Tabela 1:** as 8 células × A1, A2, A4, com Δ e o número de clientes que melhoram.
- **Tabela 2:** a matriz 2×2 por tarefa (A1/A2 × A5–A8).
- **Figura 1:** cancelamento e passo efetivo por rodada (A2, A4, A5, A6), com a referência ortogonal.
- **Figura 2:** curvas de validação fed × local (convergência, H3).
- **Ressalvas:** uma seed; o local do ISIC cls ainda sobe na rodada 400; mais falsos positivos em
  máscaras vazias no SIIM e no BUSI; da E3 para a E8 mudaram vários fatores ao mesmo tempo.

## 10. Esqueleto do artigo (6 páginas)

| Seção | Conteúdo | Espaço |
|---|---|---|
| 1. Introdução | Problema, lacuna, contribuições | ~0,75 p |
| 2. Trabalhos relacionados | FMTL (FMTL-Bench, FedHCA2, FedBone), FL médico (OmniFM), conflito × diluição (PCGrad, FedExP) | ~0,5 p |
| 3. Método | FedPer hierárquico, medida de cancelamento, FedExP | ~1,25 p |
| 4. Configuração experimental | Datasets, partição, clientes, orçamento, métricas | ~0,75 p |
| 5. Resultados | Tabelas 1–2, Figuras 1–2 | ~1,5 p |
| 6. Conclusão | Achados e limitações | ~0,5 p |
| Referências | | ~0,75 p |

## 11. Cronograma

| Data | Trabalho | GPU |
|---|---|---|
| 09/11 (seg) | Fechar este documento | — |
| 10/11 (ter) | Implementar o filtro de tarefa e o FedExP; testes e smokes | Fed-seg à noite |
| 11–13/11 | Escrever introdução, relacionados e método | Fed-cls, FedExP, Local-seg, Local-cls (~55 h) |
| 14–16/11 | Análise, tabelas, figuras, resultados | Folga para resume |
| 17/11 | Revisão completa | — |
| 18/11 (qua) | Submissão | — |

Ordem de corte, se faltar tempo: os locais de tarefa única (A7, A8) saem primeiro, depois o FedExP.

## 12. Registro

- 2026-11-09 — recorte fechado com o usuário: artigo = núcleo da dissertação, base da E8, ESANN 2027,
  uma seed, mitigação da diluição (sem protótipos), ablação 2×2 de tarefa com os quatro braços.
