# research_decisions — recorte da pesquisa de mestrado (FedCHAMP)

Pasta criada em 2026-10-26 para separar as frentes de pesquisa que cresceram neste repositório e
decidir quais delas formam a dissertação. Material de trabalho, não é documento final: nada aqui
muda código, configs ou artefatos congelados, e nenhuma implementação foi feita ou prometida.

O trabalho foi feito duas vezes, em computadores diferentes, a partir de sessões `grill-me` sobre o
pré-projeto e o `plano.md` (E0–E7): a **versão A** (2026-10-26, arquivos `00`–`03` e as duas
páginas HTML de recorte) e a **versão B** (2026-10-27, `escopo_e_decisoes.md`,
`literatura_mas_fedmtl.md` e `mapa_da_pesquisa.html`). As duas foram mantidas inteiras; a seção
"Como as duas versões se encaixam" abaixo diz o que é comum, o que só uma tem e o que divergia.

| Arquivo | Versão | Conteúdo |
|---|---|---|
| `PPGEC_MODELO_QUALIFICAÇÃO_MESTRADO_2025.docx.pdf` | A e B | pré-projeto aprovado (FedCHAMP) |
| `escopo_e_decisoes.md` | B | **ponto de entrada**: pergunta, hipóteses reformuladas, mapa frente → hipótese, decisões, ramos, pendências P1–P5 e cronograma |
| `00_projeto_vs_repositorio.md` | A | resumo do pré-projeto + matriz de rastreabilidade item a item + a tensão central |
| `01_frentes_de_pesquisa.md` | A | inventário das frentes F1–F7, evidência de cada uma e onde está |
| `02_opcoes_de_recorte.md` | A | recortes possíveis (Opções 1–3), recomendação e destino de cada trabalho |
| `03_perguntas_abertas.md` | A | perguntas da sessão `grill-me`, respondidas e em aberto (Q1–Q17) |
| `literatura_mas_fedmtl.md` | B | o que MAS, FedMTL e FMTL-Bench dizem de fato sobre transferência negativa e conflito, com páginas, lido dos PDFs |
| `mapa_da_pesquisa.html` | B | página única com a árvore de decisões interativa da versão B (offline, sem CDN) |
| `mapa_de_escopo.html` | A | versão visual de `00`–`03` |
| `arvore_de_decisoes.html` | A | árvore interativa da versão A: cada ramo com estado (feito, decidido, aberto, mantido, cortado) |

## Diagnóstico em 5 linhas

1. O projeto aprovado é o **FedCHAMP**: FL multi-domínio e multi-tarefa que mitiga a transferência
   negativa com **corpo-cabeça + agrupamento de clientes por protótipos**.
2. O repositório nasceu de um FedPer em ultrassom de mama (BUSI) e cresceu para 4 datasets, mas a
   **Etapa 2 (protótipos e clusterização) não existe**; o "cluster" atual é o rótulo do dataset.
3. Em vez disso, cresceram frentes fora do projeto: topologia HC-FMTL, ablações de loss, resolução,
   decoder federado, transfer learning, monitor e um paper só de BUSI.
4. A frente mais forte é o **diagnóstico de conflito (E0)**, e ela contradiz a premissa: entre
   datasets não há cancelamento relevante na escala da agregação; o custo dominante é a
   **diluição do passo (~10×)**. O federado ainda fica abaixo do local-only (lacuna média das 8
   células na E3: −0.033 no `primary` → −0.017 no `init_seed_dataset`). A literatura confirma o
   padrão: no FMTL-Bench, entre domínios, nenhum método federado supera o Local de forma relevante.
5. Decidido: **transferência negativa ampliada** (cancelamento/conflito **ou** diluição);
   saem PANDA + detecção e a varredura de backbones; nada já implementado é descartado; nenhuma
   implementação nesta fase. Direção: **FedCHAMP enxuto** (Opção 1 de `02_opcoes_de_recorte.md`).

## Como as duas versões se encaixam

**Concordam em tudo o que foi decidido:** projeto = FedCHAMP, direção inalterada (mitigar
transferência negativa); prazo de 4–6 meses; orientadores cientes e qualificação feita; Etapa 1
feita e Etapa 2 inexistente; H1 ampliada para cancelamento **e** diluição; saem PANDA, detecção e a
varredura de backbones; nada implementado é descartado; nenhuma implementação agora.

**Só a versão A tem:** a matriz de rastreabilidade item a item (`00`), o inventário F1–F7 com o
paper BUSI/CIARP e os achados transversais (`01`), as três opções de recorte com a recomendação e o
destino de cada trabalho (`02`), e as perguntas Q14 (que mecanismo ataca a diluição), Q15 (métrica
operacional de transferência negativa) e Q17 (cluster oráculo basta?) (`03`).

**Só a versão B tem:** H2 e H3 lado a lado com a evidência, os ramos registrados com o que cada um
precisaria para voltar, o cronograma mês a mês, as pendências P2 (medir H2) e P5 (coleta online de
Δ×wᵢ), e toda a leitura da literatura (`literatura_mas_fedmtl.md`).

**Mesma pergunta, nomes diferentes:**

| Versão A | Versão B | Tema |
|---|---|---|
| Q9, Q14, Q17 | P1 | mecanismo da Etapa 2 e se ele ataca a diluição |
| Q10 | P3 | baselines externos (FedMTL, MAS) |
| Q16 | P4 | papel e destino de E4, E6, E7 |
| F1–F7 | §4 (mapa de frentes) | o que o repositório já produziu |

**Onde divergiam, e como fica:**

| Ponto | Versão A | Versão B | Consolidado |
|---|---|---|---|
| Lacuna da E3 para o local-only | −0.033 → −0.017 | "média −0.03 a −0.05 por célula" | **−0.033 (`primary`) → −0.017 (`init_seed_dataset`)**, conferido na tabela da E3 em `plano.md`; o texto da versão B foi corrigido |
| MAS | não cortado, em aberto (Q10) | listado como ramo fora do escopo, decisão em P3 | **em aberto**, mas a leitura do artigo pesa contra: o MAS supõe que todo cliente tem todas as tarefas, o que não encaixa no cenário FedPer multi-domínio |
| Few-shot | não feito, sem decisão | ramo fora do escopo | **ramo registrado** (só a escassez natural do BUSI) |
| E7 (ResNet-34) | trabalho futuro ou o único ponto da escolha de backbone | a única variação de backbone (§5.4) | **única variação de backbone**, condicionada a P4 (rodar ou registrar como futuro) |
| E4 / E6 | trabalhos futuros / hipóteses de mitigação | mantidos, treino pendente | **mantidos**, papel decidido em Q16/P4 |

## Regras que continuam valendo

Nenhuma decisão de recorte altera artefatos congelados: partição federada, `INTER_NEAREST` do BUSI,
`mapping.csv`. p-valores do estudo continuam exploratórios — nunca "estatisticamente significativo"
(o FMTL-Bench usa a expressão; isso não se transfere para este projeto).
