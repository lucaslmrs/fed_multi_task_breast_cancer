# Pré-projeto aprovado × o que o repositório tem

Fonte: `PPGEC_MODELO_QUALIFICAÇÃO_MESTRADO_2025.docx.pdf` (nesta pasta), lido em 2026-10-26.

## O projeto em uma página

- **Título.** FedCHAMP: um framework de aprendizado federado multi-problema baseado em
  agrupamento de protótipos para imagens médicas (UPE/PPGEC; orientador Leandro Honorato de Souza
  Silva, coorientador Bruno José Torres Fernandes).
- **Problema.** Na agregação ingênua (FedAvg) com múltiplos domínios e tarefas, os conhecimentos se
  **cancelam** → transferência negativa.
- **Pergunta.** Separar corpo e cabeça **e** agrupar por protótipos garante que os conhecimentos
  não se cancelem no FL multi-task-domain?
- **Hipóteses.**
  - H1: evitando a agregação ingênua, não há mais cancelamento.
  - H2: o treino converge mais rápido.
  - H3: supera o standalone e o treino separado por tarefa em cenários de poucos dados.
- **Método.** Etapa 1: desacoplamento corpo-cabeça + escolha de backbone (YOLO, ResNet,
  MobileNetV2, VGG16, ViT, Swin). Etapa 2: protótipos locais → clusterização no servidor
  (cosseno) → agregação hierárquica em 2 fases (intra-cluster; global com resolução de conflito,
  ex. projeção ortogonal).
- **Dados.** ISIC 2018, TCGA-LGG, PANDA, SIIM-ACR; Dirichlet; clientes few-shot.
- **Baselines.** Standalone, FedAvg, MAS, FedMTL.
- **Cronograma.** 12 meses; prazo informado pelo Lucas: **4–6 meses** para a dissertação escrita.

## Matriz de rastreabilidade

✅ feito · 🟡 parcial / diferente do proposto · ❌ não existe · ➕ existe mas não estava no projeto

| Item do pré-projeto | Estado | Onde / observação |
|---|---|---|
| Desacoplamento corpo-cabeça (Etapa 1) | ✅ | FedPer: trunk `encoder2..5`+`bottleneck` federado; stem, decoder e cabeças locais (`model_split.py`) |
| Escolha de backbone (CNN × ViT) | 🟡 | MTnnUNet fixo. ResNet-34 ImageNet (E7) implementado, **sem resultado**. Nenhum ViT/YOLO |
| Protótipos locais | ❌ | Não há cálculo nem envio de protótipos |
| Clusterização de clientes no servidor | ❌ | — |
| Agregação hierárquica em 2 fases | 🟡 | Existe, mas o "cluster" é o **rótulo do dataset** (oráculo), não um agrupamento aprendido (`server.py::_hierarchical_weights`) |
| Resolução de conflito na fase global (projeção ortogonal etc.) | ❌ | Só medição (PCGrad-like M1), não correção |
| Medir cancelamento (H1) | ✅➕ | Muito além do proposto: cosseno de deltas, razão de cancelamento com referência ortogonal, M1, M2 com controle nulo (E0) |
| Convergência (H2) | 🟡 | Curvas existem (monitor, histories); comparação "rodadas até X" não feita; todos os braços sub-treinados (D15) |
| Poucos dados vs standalone (H3) | 🟡 | Comparação feita (local-only); resultado **desfavorável** exceto BUSI |
| ISIC 2018 / TCGA-LGG / SIIM-ACR | ✅ | Pré-processados, no master |
| PANDA (histopatologia) | ❌ | — |
| Curated BUSI | ➕ | Origem do repo; não está na lista do projeto (mas "ultrassom" está nos domínios) |
| Classificação + segmentação | ✅ | — |
| Detecção | ❌ | — |
| Dirichlet | ✅ | — |
| Clientes few-shot | ❌ | Só a escassez natural do BUSI (~115 amostras/cliente) |
| Baseline Standalone | ✅ | `federated.standalone: True` |
| Baseline FedAvg | ✅ | modo `flat` (`ablation_flat`) |
| Baseline MAS | ❌ | Artigo em `referencies/` |
| Baseline FedMTL | ❌ | Artigo em `referencies/` |
| Topologia multi_task (HC-FMTL), loss (E3), resolução (E4), decoder federado (E6), monitor, fontes (E5), paper BUSI (CIARP) | ➕ | Fora do projeto |

## A tensão central

O projeto supõe que **o problema é cancelamento**. A E0 mediu isso no cenário do projeto e encontrou:

- cosseno entre deltas de datasets diferentes ≈ 0 e cancelamento ≈ referência ortogonal
  (atualizações quase ortogonais, não opostas);
- interferência por curvatura quando o update de outro dataset é aplicado inteiro, mas
  **desprezível na escala real da agregação** (× wᵢ);
- **o custo dominante é a diluição do passo próprio (~10×)**, não o cancelamento.

Consequência: um mecanismo que só evita somar clientes "incompatíveis" ataca uma causa que, até
agora, não aparece. Isso não invalida a pesquisa — a pergunta é sim/não ("garante que não se
cancelam?") e a evidência é uma resposta — mas decide **como** H1 é formulada na dissertação.
Ressalva: E0 é uma seed, um holdout, estado final.
