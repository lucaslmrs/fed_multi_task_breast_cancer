# research_decisions — recorte da pesquisa de mestrado

Pasta criada em 2026-10-26 para separar as frentes de pesquisa que cresceram neste repositório e
decidir quais delas formam a dissertação. Material de trabalho, não é documento final.

| Arquivo | Conteúdo |
|---|---|
| `PPGEC_MODELO_QUALIFICAÇÃO_MESTRADO_2025.docx.pdf` | pré-projeto aprovado (FedCHAMP) |
| `00_projeto_vs_repositorio.md` | resumo do pré-projeto + matriz do que foi/não foi feito + a tensão central |
| `01_frentes_de_pesquisa.md` | inventário das frentes, o que cada uma já produziu e onde está |
| `02_opcoes_de_recorte.md` | recortes possíveis, recomendação e destino de cada trabalho |
| `03_perguntas_abertas.md` | perguntas da sessão `grill-me`, respondidas e em aberto |
| `mapa_de_escopo.html` | versão visual de tudo acima (abrir no navegador) |
| `arvore_de_decisoes.html` | árvore interativa: cada ramo do projeto com estado (feito, decidido, aberto, mantido, cortado) |

## Diagnóstico em 5 linhas

1. O projeto aprovado é o **FedCHAMP**: FL multi-domínio e multi-tarefa que mitiga a transferência
   negativa com **corpo-cabeça + agrupamento de clientes por protótipos**.
2. O repositório nasceu de um FedPer em ultrassom de mama (BUSI) e cresceu para 4 datasets, mas a
   **Etapa 2 (protótipos e clusterização) não existe**; o "cluster" atual é o rótulo do dataset.
3. Em vez disso, cresceram frentes fora do projeto: topologia HC-FMTL, ablações de loss, resolução,
   decoder federado, transfer learning, monitor e um paper só de BUSI.
4. A frente mais forte é o **diagnóstico de conflito (E0)**, e ela contradiz a premissa: entre
   datasets não há cancelamento relevante na escala da agregação; o custo dominante é a
   **diluição do passo (~10×)**. O federado ainda fica abaixo do local-only (−0.033 → −0.017).
5. Decidido em 2026-10-26: **transferência negativa ampliada** (cancelamento **ou** diluição);
   saem PANDA + detecção e a varredura de backbones; nada já implementado é descartado; nenhuma
   implementação nesta fase. Direção: **FedCHAMP enxuto** (ver `02_opcoes_de_recorte.md`).

## Regras que continuam valendo

Nenhuma decisão de recorte altera artefatos congelados: partição federada, `INTER_NEAREST` do BUSI,
`mapping.csv`. p-valores do estudo continuam exploratórios — nunca "estatisticamente significativo".
