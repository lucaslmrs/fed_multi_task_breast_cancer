# Alternativas para atacar a ortogonalidade e a diluição

Pesquisa de 2026-11-03, em três frentes paralelas: magnitude do passo, alinhamento de representação
e compartilhamento seletivo. As citações foram conferidas nas páginas do arXiv, PMLR, CVF ou
OpenReview. "Venue não confirmada" significa que não foi possível confirmar onde o trabalho saiu.
Os números vêm dos resumos e tabelas dos próprios papers. **Fichas completas ainda não foram feitas.**

## Diagnóstico de partida (E0, estado final, uma seed)

- Entre datasets, os deltas do trunk são **ortogonais** (cosseno ≈ 0): não há conflito, mas também não há
  transferência positiva de primeira ordem.
- Os passos dos outros clientes, escalados por wᵢ, contribuem com Z ≈ 0. O passo próprio fica **~10×
  diluído**.
- Na validação, o passo próprio inteiro é negativo (overfitting). A diluição pode estar protegendo a
  generalização.
- Federado < local-only (lacuna média de −0,017 a −0,033).

O problema tem duas faces, e cada família de método ataca uma:

| Face | Pergunta | Família |
|---|---|---|
| **Magnitude** | Como não perder o passo próprio na média? | A. Correção do passo no servidor, B. auto-peso / interpolação |
| **Ausência de sinal comum** | Como fazer os datasets terem algo para transferir? | C. Trunk comum pré-treinado, D. alinhamento explícito |
| **Com quem** | Como não forçar colaboração inútil? | E. Pesos de colaboração aprendidos |

## Ranking: chance de federado ≥ local-only

### 1. Interpolação validada por cliente e por bloco (FedALA / APFL) — custo mínimo

- **Ideia:** θⱼ = θ_local + βⱼ,b · (θ_fed − θ_local), com β escolhido na validação de cada cliente e em
  cada bloco b. Com β = 0 o resultado é o local-only, então na validação não fica pior.
- **Evidência:** FedALA (Zhang et al., AAAI 2023, arXiv:2212.01197), até +3,27% sobre 11 baselines;
  APFL (Deng et al., arXiv:2003.13461, venue não confirmada); Pillutla et al. (ICML 2022,
  arXiv:2204.03809), personalização parcial.
- **No repositório:** a branch `exp/validated-interpolation` (`src/experiments/validated_interpolation.py`)
  já faz a versão post-hoc.
- **Experimento mais barato:** interpolação post-hoc por bloco sobre os checkpoints pareados que já existem.
  Não precisa de treino novo.
- **Limite científico:** um ganho aqui é "a federação como regularizador", não transferência. Os β por bloco
  são, porém, uma leitura direta de *quais blocos transferem*, o que tem valor diagnóstico.
- **Risco:** na validação o resultado não piora por construção, mas no teste isso não é garantido.

### 2. Correção da magnitude no servidor (FedExP / agregação que preserva a norma) — ataca o mecanismo medido

- **FedExP** (Jhunjhunwala, Wang, Joshi, ICLR 2023, arXiv:2301.09604):
  η_g = max{1, Σwᵢ‖Δᵢ‖² / (2‖ΣwᵢΔᵢ‖²)}. Foi derivado exatamente para atualizações diversas: com N clientes
  ortogonais, η ≈ N/2, e cada cliente recupera cerca de metade do passo próprio. Em visão, o FedAvg
  alcança a mesma meta com 1,4–2× menos rounds.
- **Agregação que preserva a norma:** η = 1/(1−c), onde c é a **sua razão de cancelamento já registrada**.
  Fontes: FedMRUR (arXiv:2311.05924, venue não confirmada); Charles et al., NeurIPS 2021
  (arXiv:2106.07820).
- **Respaldo externo:** em model merging, task vectors ortogonais são *somados* com λ ajustado, não
  tirados na média (Task Arithmetic, ICLR 2023, arXiv:2212.04089; tangent-space, NeurIPS 2023,
  arXiv:2305.12827).
- **Contra-evidência importante:**
  - Charles et al. documentam **exatamente a sua diluição**: os updates são quase ortogonais e ‖média‖ cai
    como 1/√M. Mas restaurar a norma melhorou o *treino* mais do que o *teste*.
  - FedLAW (ICML 2023, arXiv:2302.10911) mostra que *encolher* o passo agregado pode ajudar a generalizar.
  - Isso conversa com o seu achado de overfitting na validação.
- **Experimento mais barato:**
  1. Calcular offline o η_t de cada round a partir do `aggregation_history.json`.
  2. Rodar um braço com η capado (4–8), com a média dos dois últimos iterados (como no FedExP).
  3. Comparar com η fixo ∈ {1, 2, √N} (FedOpt-SGD; Reddi et al., ICLR 2021, arXiv:2003.00295).
- **Valor para a tese:** é o teste de **intervenção** que falta para afirmar que a diluição *causa* o gap.
  Os dois resultados são informativos: se o gap fecha, a diluição é a causa; se o treino melhora mas o teste
  não, a diluição é protetora.

### 3. Trunk comum pré-treinado com adaptação eficiente (estende a E7) — a evidência mais sólida

- **Evidência:**
  - Nguyen et al., "Where to Begin?" (ICLR 2023, arXiv:2210.08090): o pré-treino reduz o dano da
    heterogeneidade, chega a +40% de acurácia e permite mais épocas locais sem degradação.
  - RadImageNet (Mei et al., Radiology: AI 2022): supera o ImageNet em +4,0% de AUC em massas de mama no
    ultrassom.
  - FFA-LoRA (ICLR 2024, arXiv:2403.12313), para federar só adaptadores.
- **Por que ataca a ortogonalidade:** um encoder comum fixo tira do trunk a liberdade de rotear cada
  modalidade para um subespaço próprio (hipótese 2a). Os adaptadores federados passam a aprender só o
  resíduo.
- **Experimento mais barato:**
  1. Rodar a E7 pareada (`exp/transfer-learning`).
  2. Depois, um braço com o trunk congelado e só adaptadores (ou o bottleneck) federados.
  3. Medir o cosseno entre datasets: ele fica positivo?
- **Risco:** o ganho pode vir só do pré-treino, sem que a federação acrescente nada. O braço local-only da E7
  é obrigatório.

### 4. Protótipos foreground/background por masked average pooling — **a opção mais original**

- **O ponto-chave da pesquisa:** os métodos de protótipo (FPL, CVPR 2023; FedProto, AAAI 2022; FedPCL,
  NeurIPS 2022) fazem a média por classe. Como os datasets não compartilham classes diagnósticas, um
  protótipo global de classe só agrega clientes do mesmo dataset e **não cria sinal entre datasets**.
- **O único espaço de rótulos comum aos datasets é a segmentação binária: lesão × fundo.** Protótipos
  fg/bg obtidos por masked average pooling das features do trunk dão duas "classes" compartilhadas por
  BUSI, TCGA, SIIM e pelos clientes de segmentação do ISIC.
- **Lacuna:** a pesquisa não encontrou um trabalho federado que faça exatamente isso entre modalidades.
- **Conexão com o FedCHAMP:** isso **resgata a ideia original de protótipos**, mas com outro papel. O
  protótipo deixa de servir para *agrupar clientes* (o que se reduziria aos datasets) e passa a servir para
  *criar uma direção comum* no trunk.
- **Experimento mais barato:**
  1. Uma loss auxiliar com peso 0,05–0,1 no encoder5 ou no bottleneck.
  2. O servidor faz a média dos protótipos fg/bg (alguns KB por round).
  3. Os clientes puxam as próprias features para eles (MSE estilo FedProto, ou InfoNCE).
  4. Reportar a variação do cosseno entre datasets junto com o Dice.
- **Riscos:**
  - Máscaras vazias (BUSI normal, SIIM e TCGA negativos) não geram foreground e precisam ser excluídas da
    loss.
  - Lesões minúsculas (SIIM) dão protótipos ruidosos.
  - Os clientes só de classificação não participam.
  - Pode elevar o cosseno sem fechar o gap. Por isso o local-only pareado é obrigatório.

### 5. Pesos de colaboração aprendidos (FedAMP / FedCollab)

- **Evidência:** nas tabelas do FedHCA2 (CVPR 2024), os únicos métodos que superam o Local em FMTL
  heterogêneo são os que aprendem com quem colaborar: FedAMP com +1,47 e +0,27, e o próprio FedHCA2 com
  +2,18 e +0,75. O FedPer fica com −1,11 e −0,16.
- **Fontes:**
  - FedAMP: Huang et al., AAAI 2021, arXiv:2007.03797.
  - FedCollab: Bao et al., ICML 2023, arXiv:2306.06508. Visa explicitamente o negative transfer contra o
    local e pode devolver coalizões unitárias, que equivalem ao local.
- **Experimento mais barato:** FedAMP no servidor. Cada cliente recebe um trunk ponderado pela similaridade
  com os demais, calculada a partir dos deltas já registrados. Não exige mudança nos clientes.
- **Risco:** os ganhos na literatura são de 1–2%. Com cosseno ≈ 0 entre datasets, ele deve convergir para o
  agrupamento por dataset, isto é, para algo próximo do `hierarchical` atual.

## Descartadas ou de baixa prioridade

| Opção | Motivo |
|---|---|
| PCGrad / CAGrad / GMA / FedBone (projeção de conflito) | Não há conflito para remover. O GMA zeraria quase todo o update entre datasets |
| O-LoRA / GPM / subespaços ortogonais | Forçam uma ortogonalidade que já existe |
| Clustered FL / IFCA | Os clusters seriam os datasets, que já é o `hierarchical` |
| FedBN dentro do trunk | O MTnnUNet usa InstanceNorm, sem estatísticas de execução: a motivação do FedBN não se aplica diretamente. Ainda seria possível personalizar os afins da norma, mas o `shared_modules` não expressa exclusão por chave |
| Destilação (FedDF, FedMD) | Não há um conjunto proxy público que cubra as quatro modalidades |
| Harmonização espectral (FedDG / HarmoFL / OmniFM) | Validada em multi-site dentro da mesma modalidade. Entre ultrassom e dermatoscopia a distância é muito maior, e o OmniFM não tem código nem números no resumo. Vale como trabalho relacionado e como opção secundária |
| SAM no treino local (FedSAM, ICML 2022) | Custo 2×, efeito indireto. Pode entrar como complemento |

## Proposta de objetivo de pesquisa

O que combina melhor com o diagnóstico e com o projeto aprovado:

> **Mitigar a transferência negativa em FMTL cross-modality atacando as duas faces medidas: criar
> sinal comum entre datasets (protótipos lesão/fundo, opção 4) e preservar o passo próprio na agregação
> (FedExP ou norma preservada, opção 2), avaliados contra o local-only pareado.**

- A opção 2 é o **teste de intervenção** que transforma a diluição de correlação em causa.
- A opção 4 é a **contribuição original**: ela resgata os protótipos do FedCHAMP e preenche a lacuna
  encontrada.
- As opções 1 e 3 entram como **baselines fortes e baratos**. Se a interpolação validada já superar o
  local-only, ela vira o piso que qualquer método proposto precisa vencer.

Ordem sugerida, da mais barata para a mais cara:
1. Interpolação post-hoc por bloco (sem treino).
2. η do FedExP calculado offline dos logs (sem treino).
3. Braço com FedExP capado.
4. Braço com protótipos fg/bg.
5. E7 pareada com adaptadores.

Todas as comparações seguem exploratórias: os p-valores levam a tag
`exploratory_only_non_independent_client_fold_pairs`. Uma afirmação positiva exige várias seeds.
