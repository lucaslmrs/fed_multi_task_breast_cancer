# Frentes de pesquisa desenvolvidas no repositório

Levantamento feito em 2026-10-26 a partir de `plano.md`, `docs/RESEARCH_NOTES.md`,
`docs/DECISION_SOURCES.md`, `docs/GRADIENT_CONFLICT_METHODS.md`, `papers/` e das notas de decisão
anteriores. Cada frente tem: pergunta, o que já existe, estado da evidência e onde está.

> **Atualização (mesmo dia, depois de ler o pré-projeto FedCHAMP):** os papéis abaixo foram
> escritos antes da leitura. O destino revisado de cada frente está em `02_opcoes_de_recorte.md`
> ("O que acontece com o trabalho que não estava no projeto"): F1 vira resultado preliminar, F2 e
> F4 são núcleo, F3 é configuração, F5 é apoio, F6 é futuro, F7 é protocolo.

Legenda de papel provisório: **N** = candidata a núcleo · **A** = capítulo de apoio ·
**F** = trabalho futuro · **I** = infraestrutura/rigor (sustenta, mas não é contribuição).

---

## F1 — FedPer multitarefa em ultrassom de mama (núcleo original) · N

- **Pergunta.** Um encoder federado com cabeças personalizadas (seg e cls) ajuda cada hospital
  em relação a treinar sozinho, e quanto fica do teto centralizado?
- **Existe.** Pipeline Flower + FedAvg; partição Dirichlet congelada; três setups (local-only,
  federado, centralizado) avaliados pelo mesmo `unified_eval`; experimentos A (2+2 clientes) e
  B (4+4) em Curated BUSI.
- **Evidência (descritiva).** Federado > local-only em todas as métricas de cls (AUC p=0.008 e
  0.039, exploratórios); seg neutra (±0.01); centralizado é teto claro; classe `normal` melhora
  (recall 0.31 → 0.47).
- **Ressalvas.** Runs históricos com o sistema antigo (`encoder1` compartilhado, `normal` fora da
  seg, Dice puro, Focal). Pares cliente×fold não independentes; uma seed.
- **Onde.** `docs/RESEARCH_NOTES.md` §4–5; `papers/first_try/`, `papers/ciarp2026/` (short paper).

## F2 — Federação multi-dataset e multimodal · N ou A

- **Pergunta.** Um trunk compartilhado entre datasets de modalidades diferentes (US mama,
  dermatoscopia, RM cerebral, raio-X tórax) transfere algo útil?
- **Existe.** Stem `encoder1` personalizado por modalidade (1 vs 3 canais); agregação hierárquica
  (normaliza por dataset, depois `dataset_weights`); 4 datasets pré-processados; supervisão parcial
  (ISIC tem seg e cls em imagens disjuntas).
- **Evidência.** Baselines B0: o federado fica **abaixo** do baseline em ISIC cls (0.35–0.39 vs
  0.48–0.55) e SIIM cls (0.59 vs 0.77); BUSI é a exceção (0.60–0.66 vs 0.47–0.51). Seg: baseline ≥
  federado. E3: lacuna média para o local-only −0.033 → −0.017 (3 seeds, descritivo).
- **Onde.** `RELATORIO_EXECUTIVO_FEDERACAO_MULTI_DATASET.html`, `plano.md` B0/E3, `data/*/PAPER_NOTES.md`.

## F3 — Topologia de clientes: single_task vs multi_task (HC-FMTL) · N ou A

- **Pergunta.** O desenho "um cliente = uma tarefa" duplica a mesma imagem em dois silos. O que
  muda quando um cliente tem várias tarefas sobre as mesmas imagens?
- **Existe.** Topologia `multi_task`, braços `single_task_primary/local` no manifest; loss com
  máscaras `has_mask`/`has_label`; enquadramento na literatura (MOCHA vs HC-FMTL/FedHCA², FedBone).
- **Evidência.** Contribuição **metodológica** (silo sem duplicação) mais que numérica; o trade-off
  aceito é que Dice/acc podem cair. Não há comparação fechada registrada no plano.
- **Onde.** Memória `client-topology-hcfmtl`; D11/D13 em `DECISION_SOURCES.md`.

## F4 — Diagnóstico de conflito de gradiente e transferência negativa · N ou A

- **Pergunta.** Quando a federação piora um cliente, é por conflito entre datasets/tarefas ou por
  outra coisa?
- **Existe.** Cosseno entre deltas por bloco; razão de cancelamento com referência ortogonal;
  M1 (cosseno no θ recebido, PCGrad) e M2 (lookahead TAG) com controle nulo; política de seed
  pessoal por dataset.
- **Evidência (o achado mais original do repo).**
  - O cosseno ≈ 0 não era cegueira da métrica: vinha de stems inicializados por cliente (H3).
    Com seed por dataset, o alinhamento intra-dataset aparece (0.39–0.57) e decai.
  - Entre datasets, o update inteiro interfere por **curvatura** (Z −16 a −22e-4 vs ≈0 aleatório),
    mas **na escala aplicada pela agregação** a interferência some.
  - **O custo dominante da federação é a diluição do passo próprio (~10×)**, não o conflito.
  - Uma seed, um holdout, estado final.
- **Onde.** `plano.md` E0; `docs/GRADIENT_CONFLICT_METHODS.md`; D14/D15/D26.

## F5 — Ablações de otimização: loss e orçamento · A

- **Pergunta.** Qual configuração de loss/λ/agregação/orçamento local é justa e funciona?
- **Existe / decidido (E3).** DiceBCE com **Batch Dice**; CE com `balanced_fold`; λ 4:1; seed
  pessoal por dataset adotada; Kendall, λ 1:1 e agregação 1:1 rejeitados. Budget `steps` vs
  `epochs` identificado como o confundidor mais forte (≈26× mais épocas para o dataset pequeno).
- **Evidência.** 3 seeds, descritivo. Todos os braços batem no teto de rodadas (D15).
- **Onde.** `plano.md` E3; `studies/e3_loss_ablation.yaml`.

## F6 — Arquitetura e escala: resolução, compartilhamento ampliado, transfer learning · F

- **Pergunta.** Mais resolução (E4), federar o decoder (E6) ou partir de ResNet-34 ImageNet (E7)
  fecha a lacuna para o local-only?
- **Estado.** Planejadas e implementadas em worktrees separadas, 400 rodadas, uma seed cada;
  **sem resultados registrados** no `plano.md` (status ⬜).
- **Observação.** Cada uma abre uma pergunta nova; nenhuma é necessária para responder F1, F3 ou F4.

## F7 — Infraestrutura e rigor · I

- B0 baselines não federados; E1 validação do treinamento (diagnósticos D1–D15); E2 app de
  monitoramento; E5 verificação das fontes de cada decisão; partição congelada, hash de estudo,
  `study_runner`; política estatística exploratória.
- **Papel.** Sustenta a credibilidade (capítulo de método / apêndice), mas não é pergunta de
  pesquisa. Na dissertação vira "protocolo experimental e controles de justiça".

---

## Achados transversais que qualquer recorte precisa carregar

- **SIIM seg** quase não é aprendida em nenhum setup a 128 px (problema da tarefa/loss, não da
  federação); Batch Dice e seed por dataset mitigam.
- **ISIC → MEL**: o federado desloca predições para MEL em todos os braços federados.
- **TCGA e SIIM** têm classe derivada da máscara: ganho de cls não é independente da seg.
- **Inferência**: tudo é descritivo (1 ou 3 seeds); confirmatório exige seeds independentes
  (perfil `final`: 1993/1994/1995) ou inferência por amostra.
