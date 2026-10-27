# Escopo e decisões da pesquisa — FedCHAMP

> Documento vivo, criado em 2026-10-27. Objetivo: **simplificar** o projeto de mestrado, ligar cada
> frente já desenvolvida no repositório à pergunta de pesquisa, e registrar o que fica fora do
> escopo **sem descartar** o que já está implementado.

## 1. Contexto fixado pelo grill

| Pergunta | Resposta |
|---|---|
| Qual é o projeto aprovado? | **FedCHAMP** — *Federated Clustering and Hierarchical Aggregation for Multi-Problems* (PPGEC/UPE). Pré-projeto em `docs/PPGEC_MODELO_QUALIFICAÇÃO_MESTRADO_2025.docx.pdf`. |
| A direção mudou? | Detalhes mudaram, a direção não: **mitigar a transferência negativa** na agregação federada multi-domínio e multitarefa. |
| Orientadores sabem das mudanças? | Sim (BUSI entrou; PANDA e detecção não foram feitos). **Qualificação já realizada.** |
| Prazo para a dissertação escrita | **4–6 meses.** |
| Implementar algo agora? | **Não.** Este documento só organiza o pensamento. |
| O que fazer com itens do pré-projeto não realizados? | Registrar como **ramos** (fora do escopo atual), sem apagar nada implementado. |

## 2. Pergunta de pesquisa (do pré-projeto)

> A separação do corpo e da cabeça de um modelo, somada ao agrupamento baseado em protótipos,
> garante que os diferentes conhecimentos não serão cancelados durante o aprendizado federado
> multi-task-domain?

- **Etapa 1** (desacoplamento corpo-cabeça) → **implementada** (FedPer com MTnnUNet, stems pessoais,
  agregação hierárquica por dataset).
- **Etapa 2** (protótipos + clusterização no servidor + fusão global com resolução de conflito) →
  **não implementada**.

## 3. Hipóteses: original → como entram agora

| | Pré-projeto | Decisão (2026-10-27) | Evidência atual |
|---|---|---|---|
| **H1** | Evitando a agregação ingênua, os conhecimentos não são cancelados. | **Ampliar o conceito.** "Transferência negativa" passa a ser toda perda de desempenho do federado frente ao local-only atribuível à agregação, decomposta em dois mecanismos: **(a) conflito** (cancelamento direcional e por curvatura) e **(b) diluição** (o passo de cada cliente é encolhido pelos pesos de agregação). A mitigação mira os dois. | E0: entre datasets o cosseno ≈ 0 e o conflito por curvatura existe com o Δ inteiro, mas some na escala aplicada (Δ×wᵢ). O custo dominante é a **diluição (~10×)**. E3 (3 seeds): o federado fica **abaixo do local-only** na maioria das 8 células → a transferência negativa *no desempenho* existe; o mecanismo medido é a diluição. |
| **H2** | Converge mais rápido. | Mantida, **em aberto**. Mede-se em rodadas até o platô a partir do `aggregation_history.json` já gravado. | Ainda sem análise. |
| **H3** | Supera standalone e treino separado em dados limitados. | Mantida como meta da **mitigação**. Hoje é a referência a ser superada. | E3: o federado atual não supera o local-only (melhor braço `init_seed_dataset`; média −0.03 a −0.05 por célula contra local-only). B0 (1 seed): BUSI cls foi a exceção. |

**Consequência para o texto:** os resultados de E0/B0/E3 não são um "resultado negativo" — viram o
**capítulo de diagnóstico** que motiva a Etapa 2 (o que exatamente precisa ser mitigado).

## 4. Mapa: frentes desenvolvidas → papel na dissertação

| Frente | O que fez | Papel na dissertação | Status |
|---|---|---|---|
| FedPer + agregação hierárquica | Corpo compartilhado, stems/cabeças pessoais, pesos por dataset/tarefa | Etapa 1 (Objetivo específico 1) | ✅ feito |
| Multi-dataset (BUSI, ISIC, TCGA, SIIM) | 4 modalidades, partição congelada | Cenário experimental (§4.3) | ✅ feito |
| **E0** conflito de gradientes | Cosseno M1, lookahead M2, controle nulo, razão de cancelamento | **Núcleo do diagnóstico de H1** (Obj. 3) | ✅ feito |
| **B0** baselines não federados | seg / cls / multitask por dataset | Referência local e evidência de MTL | ✅ feito |
| **E1** validação do treino | Normas de gradiente, D1–D15 | Garante que as comparações são válidas (Métodos) | ✅ feito |
| **E3** ablação de loss | Batch Dice; seed pessoal por dataset vence | Define o protocolo de treino; mostra fed < local | ✅ feito |
| **E5** fontes | `docs/DECISION_SOURCES.md` | Embasa a seção de Métodos | ✅ feito |
| E2 monitor | App local | Ferramenta (apêndice, se citar) | ✅ mantido |
| E4 pré-processamento (256 px) | TCGA/SIIM em 256 | Ablação de apoio | ⏳ treino pendente, mantido |
| E6 compartilhamento amplo | Decoder federado | Relaciona-se a H1: mais compartilhamento → mais diluição/conflito? | ⏳ treino pendente, mantido |
| E7 ResNet-34 ImageNet | Encoder pré-treinado | Substitui a varredura de backbones (uma variação só) | ⏳ treino pendente, mantido |
| Centralizado (BUSI) | Teto superior | Só no estudo BUSI legado | mantido |

## 5. Decisões de escopo

1. **Núcleo**: diagnóstico da transferência negativa (feito) + **um** mecanismo de mitigação na Etapa 2
   (a decidir — ver §7). Tudo o mais é apoio.
2. **Bases**: congelar **BUSI, ISIC 2018, TCGA-LGG e SIIM-ACR**.
3. **Tarefas**: segmentação + classificação.
4. **Backbone**: MTnnUNet (padrão) e ResNet-34 ImageNet (E7) como única variação.
5. **Estatística**: 3 seeds; p-valores exploratórios; nunca "estatisticamente significativo".
6. **Comparar só braços com o mesmo modo de budget** (`local_training.mode`).

## 6. Ramos registrados (fora do escopo atual, não descartados)

| Ramo | Por que saiu | O que precisaria para voltar |
|---|---|---|
| **PANDA** (histopatologia) | WSI gigapixel: patching e semanas de pré-processamento | Novo `*_preprocessing.py` (skill `dataset-onboarding`) + nova partição |
| **Detecção** | Nenhuma base atual tem caixas; cabeça e métricas novas | Base com bounding boxes + cabeça de detecção + mAP no `unified_eval` |
| **Varredura de backbones** (YOLO, MobileNetV2, VGG16, ViT, Swin) | Custo de GPU alto e fora da pergunta | `model_split` com nomes compatíveis, como já feito na E7 |
| **Baselines externos MAS e FedMTL** | Reimplementação cara | Ver pendência P3 |
| **Few-shot extremo** (clínicas muito pequenas) | Não configurado ainda | Dirichlet mais agressivo ou subamostragem por cliente, numa variante de partição |

## 7. Pendências (decisões em aberto)

- **P1 — Mecanismo da Etapa 2.** Protótipos + clusterização como no pré-projeto? E, dado que o mecanismo
  medido é a diluição, a fusão global precisa de algo além de projeção ortogonal (por exemplo, uma
  taxa de aprendizado do servidor) para não reduzir o passo de cada grupo?
- **P2 — Como medir H2**: rodadas até o platô de validação, com quais limiares?
- **P3 — Baselines externos**: incluir FedMTL (o mais próximo da proposta)? MAS? Antes de decidir,
  ler `literatura_mas_fedmtl.md`: o MAS supõe que todo cliente tem todas as tarefas, e o FedMTL
  agrega também as cabeças e não tem corpo global único, então nenhum dos dois encaixa direto no
  cenário FedPer multi-domínio.
- **P4 — Fechar E4/E6/E7**: rodar os treinos pendentes ou registrar como trabalho futuro, dado o prazo.
- **P5 — Coleta online de Δ×wᵢ** no M2, para ver a diluição ao longo do treino e não só no estado final.

## 8. Cronograma sugerido (4–6 meses)

| Mês | Entrega |
|---|---|
| 1 | Fechar P1–P3; escrever Introdução e Fundamentação (já há material em `docs/`). |
| 2 | Implementar a Etapa 2 mínima; smoke; decidir P4. |
| 3 | Rodar Etapa 2 × FedPer × local-only (3 seeds). |
| 4 | Análise (H1 via E0 + Etapa 2; H2 via histórico; H3 via deltas). |
| 5–6 | Escrita de Métodos/Resultados/Discussão e revisão com os orientadores. |
