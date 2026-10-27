# Notas do apresentador — Avanços desde 28/09

## cover

Slide de abertura para quem não acompanha o dia a dia. Em três semanas: (1) o treino federado passou de 2 para 4 bases e 4 modalidades; (2) construímos ferramentas para verificar se o treino está correto e para acompanhá-lo; (3) a ablação da loss escolheu Batch Dice e a seed de personalização por dataset; (4) três análises novas, com 400 rodadas: decoder federado (E6), encoder ResNet-34 pré-treinado no ImageNet (E7) e resolução por base (E4, ainda rodando). A melhor média no teste, 0,700, é a da E7, com uma seed só e o dobro de rodadas da E3.

## linha-do-tempo

Ordem de execução do plano: E0, E1, E2, E5, E3 e depois E4 em paralelo com E6, seguida da E7. A B0 foi incluída a pedido em 15/10, antes de seguir. Os PRs #6 (Dice+BCE) e #7 (transferência negativa) entraram em 28/09 e 04/10; E0 a E3 foram mescladas na main até o PR #12. E4, E6 e E7 vivem nas branches exp/preprocessing-ablation, exp/wider-sharing e exp/transfer-learning, ainda não mescladas.

## datasets

BUSI e ISIC já estavam no estudo; TCGA-LGG e SIIM-ACR entraram em 28 e 29/09, com scripts próprios de pré-processamento. No TCGA e no SIIM a classe é derivada da máscara (tumor ou pneumotórax se a máscara não é vazia). O TCGA divide por paciente (stratified_group); no SIIM a anonimização deu um PatientID por DICOM, então não há agrupamento possível. Clientes por célula na E3: BUSI 2, TCGA 2, SIIM 4, ISIC seg 4 e ISIC cls 4. BUSI, TCGA e SIIM usam clientes multitarefa; o ISIC usa um cliente por tarefa porque os conjuntos de seg e cls são disjuntos.

## infraestrutura

E0: com 16 clientes, o cosseno entre deltas do trunk dava cerca de zero em todos os 120 pares. A E0.1 mostrou que a métrica é válida e que a ortogonalidade vinha da inicialização por cliente dos stems e cabeças (hipótese H3). Com seed por dataset, o cosseno intra-dataset sobe para 0,39 a 0,57 no início. Também quantificou o ruído de GPU entre runs: até cerca de 0,05 no BUSI e 0,01 nas outras bases. B0: a seg do SIIM chegou a 0,237 de Dice na época 14 e depois desaprendeu; isso levou ao Batch Dice da E3. E1: suíte de 197 testes na época, não-regressão bit a bit em CPU. E2: src/monitor, servidor da biblioteca padrão; 79 entradas listadas em 2,4 s. E5: docs/DECISION_SOURCES.md e docs/references.bib; 10 decisões verificadas, 5 adaptações, 6 próprias, 2 divergentes, 2 sem fonte.

## e3-fase-a

Fase A: 3 variantes de loss de segmentação (batch_dice, dice_heavy, focal_dice) nos 4 baselines centralizados, 12 runs, cerca de 3 horas de GPU, em 17/10. Seleção feita na validação; o teste não foi lido. O 0,000 de falso positivo do controle no SIIM vinha de não prever nada. BUSI e ISIC mudam dentro do ruído (±0,05 e ±0,01). Decisão: DiceBCE 0,5/0,5 com loss.dice_batch: true, que vai para o manifest da Fase B. Uma seed só, baseline centralizado; o efeito federado é medido na Fase B.

## e3-fase-b

Média de 3 seeds (1993 a 1995), 200 rodadas, holdout único; leitura descritiva, sem p-valores. Métrica primária: Dice positivo na seg e balanced accuracy na cls. A lacuna é a média simples dos Δ pareados das 8 células contra o local-only. Mecanismo: com seed por dataset, o cosseno M1 entre pares intra-dataset com as mesmas tarefas sobe de 0,002 a 0,003 para 0,098; o cancelamento fica abaixo do ortogonal no começo do treino. Os braços de loss não mudam a geometria. λ 1:1 troca seg por cls no SIIM e tem a pior média; agregação 1:1 não muda nenhuma célula; Kendall não melhora nenhuma. O diagnóstico D15 disparou nos 18 runs: o teto de 200 rodadas limitava todos os braços, e por isso E6 e E7 usam 400 rodadas. Decisão: personalized_init_seed: dataset virou o padrão do config.

## e6-compartilhamento

Seed 1993. A coluna E3 fed é o braço com seed por dataset, 200 rodadas; a E6 roda 400. A diferença mistura orçamento e intervenção, porque não existe controle de 400 rodadas. O mecanismo shared_modules é configurável: sem a chave, o compartilhamento e o hash de estudo continuam os de sempre. A lista ampla tem 17 módulos (encoder2–5, bottleneck, decoder1–5, upsample1–5, process_encoder_5 e process_decoder_5), 32 chaves e 64,07% dos valores. Durante a implementação foi corrigido o M1, que quebrava em silêncio nos clientes single_task do ISIC com o conjunto amplo. Ganhos maiores em BUSI cls e TCGA seg; quedas em ISIC seg e SIIM seg. A suspeita registrada é a diluição do ISIC no decoder compartilhado.

## e7-transfer

Seed 1993, 400 rodadas, comparado com a E6 como pede o plano. Arquitetura MTResNet34UNet: 202 tensores e 23,9 M de valores viajam no compartilhamento amplo; ficam pessoais encoder1, output1–4 e classifier. A BatchNorm do encoder é FrozenBatchNorm: estatísticas, escala e deslocamento congelados, então os clientes não divergem nela. Ganhos em 6 das 8 células, com o maior na classificação do ISIC (+0,185). O TCGA, que é ressonância de 3 canais, cai nas duas tarefas; uma hipótese registrada é a normalização do ImageNet aplicada a canais de RM, e outra é que o stem pessoal agora só alimenta o skip do decoder1. Ressalva: E7 contra E6 muda arquitetura e pré-treino ao mesmo tempo; separar os dois exige um braço ResNet-34 sem pré-treino.

## e4-resolucao

Status: E4 ainda rodando; a última leitura foi por volta da rodada 324 de 400. Todos os números deste slide são de validação, não de teste. A comparação justa com a E3 é na mesma janela de rodadas (191 a 200). Os ganhos aparecem justamente onde a resolução mudou: SIIM e TCGA. Atenção: o Dice a 256 px é calculado sobre outra rasterização da máscara de referência, então não é diretamente comparável ao Dice a 128 px. Dados: processed_256 gerado pela CLI de variantes sem nenhuma lesão sumida; master da E4 com 46.555 linhas, equivalente ao da E3. Uma seed (1993); a replicação multi-seed ficou pendente (E4.8).

## destaque

Este é o número que resume as três semanas. Na E3, nenhum braço federado superava o local-only. Com o decoder federado (E6) e depois o encoder pré-treinado (E7), a média dos braços federados passa a do local-only da E3. Mas a comparação tem duas confusões: os runs novos têm o dobro de rodadas, e a E7 muda arquitetura e pré-treino juntos. Não há controle local-only de 400 rodadas nem segunda seed.

## tabela-resultados

Todos os números são de teste, seed 1993, média por base e tarefa. As colunas E3 são o braço federado com seed por dataset e o local-only, na mesma seed; o registro da E3 no plano usa a média de 3 seeds, por isso os valores diferem um pouco do slide da Fase B. A E7 tem o maior valor em 6 linhas; a E6 vence nas duas células do TCGA. A seg do SIIM continua baixa em todos os braços, em torno de 0,2. A E4 não está aqui porque ainda não tem teste.

## grafico-media

Escala começando em zero para não exagerar as diferenças. Média simples das 8 células (4 bases × 2 tarefas), com Dice positivo na segmentação e balanced accuracy na classificação. Do federado da E3 para a E7 a média sobe 0,087; do local-only da E3 para a E7, 0,078. Lembrar que os dois runs novos têm 400 rodadas e que não existe controle de 400 rodadas.

## ressalvas

Estas ressalvas valem para todos os slides de resultados. O plano registra o ruído de GPU entre runs na E0 (até cerca de 0,05 no BUSI e 0,01 nas outras bases); a regra da E3 só contava um efeito acima desse ruído e do desvio entre seeds. Para separar orçamento de intervenção falta rodar o controle e o local-only com 400 rodadas. Para separar arquitetura de pré-treino falta um braço ResNet-34 sem pesos do ImageNet. No TCGA e no SIIM da E4, a máscara de referência a 256 px é outra rasterização, então o Dice não é diretamente comparável ao de 128 px.

## proximos-passos

Pendências registradas no plano (E7.9 e herdadas da E6): (a) estatísticas congeladas com γ e β treináveis e compartilhados; (b) FedBN, com BN pessoal por cliente; LR discriminativo entre encoder pré-treinado e decoder; entrada de 224 px, a resolução do ImageNet; estatísticas do ImageNet aplicadas aos canais de RM do TCGA; o stem pessoal que agora só alimenta o skip do decoder1; a diluição do ISIC no decoder e nos módulos process; a falta de controle a 400 rodadas; e a replicação multi-seed da E4 (E4.8). Abertos desde a E5: Div-1 (deep supervision) e Div-2 (Focal com pesos de classe). Também falta mesclar E4, E6 e E7 na main.
