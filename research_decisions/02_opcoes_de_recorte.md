# Opções de recorte para a dissertação

Revisado em 2026-10-26 **depois de ler o pré-projeto** (ver `00_projeto_vs_repositorio.md`). A versão
anterior tratava "mama" como eixo; o projeto aprovado é FedCHAMP (multi-domínio, multi-tarefa,
protótipos, transferência negativa), então as opções mudaram.

Restrições já respondidas: prazo de **4–6 meses**; a direção do projeto (mitigar transferência
negativa) continua a mesma.

## Opção 1 — FedCHAMP enxuto (recomendada)

Entregar o que o título promete, com o mínimo que responde à pergunta.

- **Núcleo.**
  1. Corpo-cabeça: pronto (FedPer + stem por modalidade).
  2. **Protótipos + clusterização no servidor** — a única peça nova. Versão mínima: protótipo por
     classe = média do bottleneck (GAP) por classe; cosseno entre clientes; agrupamento
     aglomerativo; agregação intra-cluster e depois global.
  3. H1 medida com o instrumental da E0 (cancelamento, M1, M2), que já é mais rigoroso que o
     proposto ("comprovar que evitou soma zero").
- **Comparação.** Standalone · FedAvg (`flat`) · hierárquico por dataset (**cluster oráculo**, já
  existe) · FedCHAMP (cluster por protótipo) · FedMTL se couber.
  A pergunta prática vira: *o agrupamento por protótipos recupera o cluster oráculo, e algum dos
  dois reduz cancelamento ou melhora sobre o FedAvg e o standalone?*
- **Dados/tarefas.** Os 4 datasets já processados (BUSI, ISIC, TCGA, SIIM); seg + cls.
- **Sai (documentado como redução de escopo).** PANDA, detecção, varredura de backbones
  (YOLO/VGG/MobileNet/ViT/Swin). *(Revisado: a proposta inicial também cortava MAS, few-shot, E4,
  E6 e E7; o Lucas decidiu manter o que já está implementado, e MAS e few-shot seguem em aberto.)*
- **Pró.** Fiel ao projeto aprovado; reaproveita quase tudo; o achado da diluição entra como
  resultado de H1 e não como desvio.
- **Contra.** Exige implementar uma coisa nova com 4–6 meses; o achado da E0 sugere que o ganho
  pode ser pequeno (aí a resposta à pergunta é "não, e o motivo é diluição").

## Opção 2 — Pivotar para "a transferência negativa não é cancelamento"

- **Núcleo.** E0 + E3 replicados com 3 seeds; protótipos **não** implementados.
- **Pró.** Menos trabalho novo; contribuição mecanística mais limpa.
- **Contra.** Abandona o título e a Etapa 2 do projeto aprovado → exige renegociar com orientador
  e coorientador, possivelmente refazer a qualificação. Arriscado no prazo.

## Opção 3 — Executar o pré-projeto inteiro

- PANDA, detecção, varredura de backbones, MAS e FedMTL, few-shot, protótipos.
- **Inviável em 4–6 meses** junto com a escrita; listada só para registrar por que foi descartada.

## Decisões já tomadas (sessão de 2026-10-26)

- **Transferência negativa ampliada:** federado abaixo do standalone, por cancelamento **ou**
  diluição. A diluição deixa de ser "desvio" e vira metade do problema.
- **Cortes:** PANDA + detecção; varredura de backbones. **Nada implementado é descartado**: E4, E6
  e E7 ficam como ramos (hipóteses de mitigação), não como frentes independentes.
- **MAS não foi cortado** (Q10 em aberto).
- **Nenhuma implementação agora**: esta fase é só de organização.
- Orientadores cientes; qualificação feita.

A Opção 1 abaixo continua sendo a direção, ajustada por essas decisões. A visualização dos ramos
está em `arvore_de_decisoes.html`.

## Recomendação

**Opção 1.** Critério para aceitar qualquer experimento daqui em diante: *ele testa H1, H2 ou H3
com o FedCHAMP ou um baseline da tabela acima?* Se não, vai para trabalhos futuros.

## O que acontece com o trabalho que não estava no projeto

| Trabalho | Destino proposto |
|---|---|
| FedPer em BUSI (experimentos A/B, CIARP) | Capítulo de resultados preliminares / validação da Etapa 1 |
| Instrumental de conflito (E0) | Método de avaliação de H1 (capítulo de metodologia) + resultado |
| Loss, Batch Dice, seed por dataset (E3) | Configuração experimental fixada, com uma tabela de ablação |
| Topologia multi_task (HC-FMTL) | Decisão de configuração justificada; não é contribuição |
| Baselines B0, E1, E5, monitor | Protocolo e controles; apêndice |
| E4, E6 | Trabalhos futuros |
| E7 (ResNet-34) | Trabalho futuro ou o único ponto da "escolha de backbone" |
