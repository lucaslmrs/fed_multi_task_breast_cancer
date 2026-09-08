# Federated Multi-Task Learning on Non-IID Data Silos: An Experimental Study

## Referência bibliográfica

Yang, Y.; Lu, Y.; Huang, S.; Sirejiding, S.; Lu, H.; Ding, Y. **Federated Multi-Task Learning on
Non-IID Data Silos: An Experimental Study**. In: *Proceedings of the 2024 International Conference
on Multimedia Retrieval (ICMR 2024)*, pp. 684–693, 2024.
[DOI: 10.1145/3652583.3657999](https://doi.org/10.1145/3652583.3657999).

- [Texto integral no arXiv](https://arxiv.org/html/2402.12876)
- [Código do FMTL-Bench](https://github.com/youngfish42/FMTL-Benchmark)

## Escopo desta nota

Esta nota possui dois objetivos. O primeiro é resumir a proposta, os cenários experimentais e as
conclusões do artigo. O segundo é registrar uma **fundamentação encontrada posteriormente** para
uma decisão específica do pipeline deste projeto: representar um silo que possui mais de uma
tarefa como **um único cliente multi-task**, treinando conjuntamente todas as tarefas disponíveis,
em vez de registrar o mesmo silo várias vezes como clientes single-task independentes.

A atribuição ao artigo é deliberadamente limitada a essa concepção do cliente multi-task como
detentor de um conjunto de tarefas e de uma função objetivo local conjunta. O artigo não é
apresentado como origem do MTnnUNet, do FedPer, da divisão entre trunk e componentes personalizados,
do mascaramento por supervisão, da agregação hierárquica ou dos pesos de tarefas adotados neste
projeto.

---

## 1. Resumo do artigo

### 1.1 Motivação e objetivo

O artigo parte da observação de que muitos trabalhos chamados de *Federated Multi-Task Learning*
(FMTL) tratavam cada cliente ou problema de personalização como uma “task”, embora todos os clientes
executassem, na prática, o mesmo tipo de tarefa supervisionada. Os autores concentram-se em um caso
mais explícito de aprendizado multi-task: um mesmo cliente pode possuir uma ou várias tarefas reais,
como estimativa de profundidade, detecção de bordas, estimação de normais de superfície e
segmentação semântica.

Para estudar sistematicamente esse cenário, o trabalho apresenta o **FMTL-Bench**, um benchmark que
organiza a heterogeneidade federada em três níveis:

1. **Dados:** quantidade de amostras, domínio do dataset e conjunto de tarefas disponível em cada
   cliente.
2. **Modelo:** backbone, número de decoders e compatibilidade dos parâmetros entre clientes.
3. **Otimização:** treinamento local, agregação federada, personalização e tratamento de conflitos
   entre gradientes das tarefas.

O objetivo principal não é propor um único algoritmo federado, mas fornecer cenários e critérios
reprodutíveis para avaliar como diferentes algoritmos se comportam quando os clientes possuem
conjuntos heterogêneos de tarefas, modelos e volumes de dados.

### 1.2 Formulação do cliente multi-task

O artigo representa cada cliente federado \(C_k\) como detentor de um dataset local
\(\mathbb{D}_k\) e de um conjunto de tarefas \(\mathcal{T}_k\). Sua função objetivo local é:

\[
\mathcal{L}_k(\theta_k)=
\frac{1}{\sum_{t\in\mathcal{T}_k}q_{k,t}}
\sum_{t\in\mathcal{T}_k}
q_{k,t}\,\ell_{k,t}(\theta_k),
\]

em que \(\ell_{k,t}\) é a loss da tarefa \(t\) no cliente \(k\), e \(q_{k,t}\) controla a
contribuição dessa tarefa para a otimização local. Por padrão, o benchmark atribui pesos iguais às
tarefas do cliente, isto é, \(q_{k,t}=1/|\mathcal{T}_k|\).

Essa formulação abrange diretamente dois casos:

- **Cliente single-task:** \(|\mathcal{T}_k|=1\). A atualização local é produzida por uma única
  tarefa.
- **Cliente multi-task:** \(|\mathcal{T}_k|>1\). As losses das tarefas pertencentes ao cliente são
  combinadas na mesma otimização local, produzindo uma atualização federada conjunta.

Portanto, um cliente multi-task não é definido como várias cópias do mesmo participante, uma para
cada tarefa. Ele continua sendo uma única unidade federada, com uma identidade, um dataset local e
um objetivo que combina as tarefas que possui.

### 1.3 Datasets e tarefas

O FMTL-Bench utiliza **NYUD-v2** e **PASCAL-Context**. Nos experimentos de domínio único, o NYUD-v2
fornece quatro tarefas de predição densa sobre cenas internas:

- estimativa de profundidade (*Depth*);
- detecção de bordas (*Edge*);
- estimação de normais de superfície (*Normals*);
- segmentação semântica (*SemSeg*).

Nos experimentos cross-domain, o PASCAL-Context introduz um domínio visual diferente e contribui
com estimação de normais e segmentação de partes humanas (*Parts*). Embora o PASCAL-Context também
possua outras anotações, esses são os tipos usados para construir os cenários cross-domain
descritos no artigo.

O conjunto de treinamento original é distribuído entre os clientes de acordo com cada cenário. Em
seguida, o dataset local de cada cliente é dividido na proporção 9:1 para treinamento e teste local.
O artigo considera tanto avaliação global, sobre o conjunto de teste original, quanto avaliação
personalizada/local, sobre os dados de cada cliente.

### 1.4 Os sete cenários de heterogeneidade

O benchmark organiza sete configurações principais:

1. **IID-1 SDMT — Single-Domain Multi-Task:** quatro clientes do mesmo domínio, com subconjuntos de
   dados não sobrepostos, executam o mesmo conjunto de quatro tarefas.
2. **NIID-2 SDST — Single-Domain Single-Task:** cada cliente executa uma tarefa distinta. É tratado
   como um caso extremo de heterogeneidade por tarefas.
3. **NIID-3 SDHT — Single-Domain Hybrid-Task:** reúne quatro clientes multi-task e quatro clientes
   single-task na mesma federação.
4. **NIID-4 UBSDMT — Unbalanced Single-Domain Multi-Task:** clientes multi-task do mesmo domínio
   recebem volumes diferentes de dados.
5. **NIID-5 UBSDST — Unbalanced Single-Domain Single-Task:** clientes single-task do mesmo domínio
   recebem volumes diferentes de dados.
6. **NIID-6 UBCDMT — Unbalanced Cross-Domain Multi-Task:** clientes multi-task de domínios distintos
   possuem quantidades e tipos heterogêneos de tarefas.
7. **NIID-7 UBCDST — Unbalanced Cross-Domain Single-Task:** combina clientes single-task de
   domínios diferentes e com volumes desiguais de dados.

O cenário híbrido NIID-3 é particularmente importante para esta nota porque demonstra que a
federação pode conter, simultaneamente, clientes com cardinalidades diferentes de
\(\mathcal{T}_k\). O servidor não precisa reinterpretar um cliente multi-task como vários clientes
single-task: a heterogeneidade é expressa no conjunto de tarefas pertencente a cada participante.

### 1.5 Arquiteturas avaliadas

O artigo avalia duas famílias de arquitetura multi-task:

- **MD — Multi-Decoder:** um encoder compartilhado alimenta um decoder específico para cada tarefa.
  O número e o tipo de decoders podem variar quando os clientes possuem conjuntos de tarefas
  diferentes.
- **TC — Task-Conditioned:** um encoder compartilhado é combinado com um único decoder condicionado
  pela tarefa executada. Essa alternativa reduz a heterogeneidade estrutural entre modelos de
  clientes com diferentes conjuntos de tarefas.

Os experimentos incluem backbones ResNet-18 e Swin-T. O estudo também considera modelos
single-task, permitindo observar como a arquitetura interage com a distribuição das tarefas entre
os clientes.

Uma consequência importante é que o artigo não impõe uma única forma física de executar o forward.
Na arquitetura multi-decoder, uma representação compartilhada pode alimentar simultaneamente os
decoders disponíveis. Na arquitetura task-conditioned, o decoder pode ser acionado sob diferentes
condições de tarefa. O elemento comum é a **otimização local conjunta das tarefas pertencentes ao
cliente**, e não a exigência de uma implementação específica do forward.

### 1.6 Algoritmos e agregação

O benchmark avalia nove baselines cobrindo treinamento local, aprendizado federado, aprendizado
federado personalizado, otimização multi-task e FMTL. Entre eles estão Local, FedAvg, FedProx,
FedAMP, FedRep, PCGrad, CAGrad, MaT-FL e FedMTL.

Como os modelos podem ter decoders diferentes, nem sempre todos os parâmetros são compatíveis para
agregação. FedRep e MaT-FL usam desacoplamento de parâmetros, transmitindo principalmente o encoder.
Nos cenários cross-domain com maior heterogeneidade de tarefas e modelos, variantes identificadas
por “-E” também restringem a comunicação aos parâmetros do encoder ou aos gradientes acumulados
correspondentes. PCGrad e CAGrad são usados para estudar conflitos entre gradientes de tarefas.

Assim, o FMTL-Bench não prescreve um único mecanismo de personalização ou agregação. Sua contribuição
é mostrar como diferentes famílias de métodos podem ser avaliadas sob a mesma taxonomia de clientes
single-task, multi-task, híbridos, desbalanceados e cross-domain.

### 1.7 Resultados e impacto

Os resultados mostram que a quantidade e o tipo de tarefas por cliente alteram substancialmente o
comportamento dos baselines. Nos cenários patológicos em que cada cliente possui somente uma tarefa,
combinações simples de agregação e modelos single-task podem sofrer forte degradação. Arquiteturas
condicionadas pela tarefa tendem a tolerar melhor a heterogeneidade estrutural, enquanto estratégias
de desacoplamento tornam possível comunicar apenas componentes compatíveis.

No cenário híbrido NIID-3, os experimentos indicam que clientes single-task e multi-task podem
participar da mesma federação, mas o resultado depende da arquitetura e do algoritmo empregado. Nos
cenários cross-domain, a mudança de domínio, volume de dados e conjunto de tarefas aumenta a
dificuldade da otimização. O artigo também amplia a avaliação além das métricas preditivas,
incluindo comunicação, tempo, consumo de energia e emissões estimadas.

O impacto científico do trabalho está principalmente na organização do problema: ele distingue
tarefas supervisionadas reais de uma interpretação puramente client-centric de “multi-task” e
fornece uma nomenclatura para experimentos nos quais cada cliente pode possuir um conjunto de
tarefas diferente. Isso permite descrever explicitamente cenários que combinam clientes single-task
e multi-task sem apagar sua heterogeneidade.

---

## 2. Contribuição para a construção do pipeline deste projeto

### 2.1 Natureza da contribuição

O FMTL-Bench foi encontrado **depois** da concepção inicial do pipeline. Portanto, ele é usado como
fundamentação conceitual posterior, e não como evidência de que a implementação tenha sido derivada
diretamente do código ou da arquitetura dos autores.

A contribuição atribuída ao artigo é específica:

> Um participante que possui várias tarefas deve ser representado como um único cliente federado
> com um conjunto de tarefas \(\mathcal{T}_k\) e uma função objetivo local conjunta, em vez de ser
> duplicado como vários clientes single-task.

Essa formulação esclareceu como representar, dentro da mesma federação, clientes que possuem apenas
uma tarefa e clientes que possuem várias tarefas.

### 2.2 Representação anterior: um silo registrado mais de uma vez

Na lógica anterior, um silo com imagens que possuíam máscara e rótulo de classe seria registrado
duas vezes no ambiente federado:

```text
Mesmo silo físico
├── cliente A-seg ── mesmas imagens + máscaras ── atualização de segmentação
└── cliente A-cls ── mesmas imagens + classes  ── atualização de classificação
```

Embora `A-seg` e `A-cls` correspondessem à mesma origem de dados, o servidor os enxergaria como dois
participantes independentes. Cada identidade faria seu próprio forward, calcularia uma loss isolada
e produziria uma atualização separada.

Essa representação tem consequências conceituais e operacionais:

- transforma um silo multi-task em dois clientes single-task artificiais;
- duplica a presença das mesmas imagens na topologia federada;
- separa os gradientes de segmentação e classificação antes da atualização local;
- permite que o servidor contabilize duas participações onde existe apenas um proprietário dos
  dados;
- impede que as duas tarefas regularizem conjuntamente a representação compartilhada durante o
  treinamento local.

Essa organização ainda pode ser útil como uma topologia experimental single-task, mas ela não deve
ser confundida com um cliente que efetivamente aprende várias tarefas.

### 2.3 Representação atual: um cliente, uma imagem e duas supervisões

Na representação atual, o silo aparece uma única vez e declara que possui segmentação e
classificação:

```text
cliente A / tarefas {seg, cls}
        │
        ├── uma imagem
        ├── uma máscara
        └── um rótulo de classe
                │
                ▼
          um forward do MTnnUNet
          ├── predição de segmentação
          └── predição de classificação
                │
                ▼
       L = λseg · Lseg + λcls · Lcls
                │
                ▼
          um backward local
                │
                ▼
       uma atualização federada do trunk
```

A mesma imagem é carregada uma única vez no batch e alimenta o modelo multi-task. O forward produz
as saídas de segmentação e classificação. Quando as duas supervisões estão disponíveis, ambas as
losses são calculadas e combinadas antes de um único `backward`. A atualização resultante contém a
interação local entre os gradientes das tarefas e é enviada ao servidor como a contribuição de um
único cliente.

Essa implementação é coerente com a função objetivo \(\mathcal{L}_k\) do artigo. O requisito
conceitual é que as tarefas pertencentes ao cliente sejam otimizadas conjuntamente. O uso de um
único forward que retorna simultaneamente classificação e segmentação é a realização específica
adotada neste projeto por meio do MTnnUNet; não é uma exigência universal imposta pelo FMTL-Bench.

### 2.4 Objetivo local implementado

No pipeline, um cliente single-task preserva o caminho histórico e otimiza somente a loss da tarefa
que possui:

\[
\mathcal{L}_k =
\begin{cases}
\mathcal{L}_{seg}, & \mathcal{T}_k=\{seg\},\\
\mathcal{L}_{cls}, & \mathcal{T}_k=\{cls\}.
\end{cases}
\]

Para um cliente multi-task:

\[
\mathcal{L}_k =
\lambda_{seg}\mathcal{L}_{seg}+
\lambda_{cls}\mathcal{L}_{cls},
\qquad
\lambda_{seg}+\lambda_{cls}=1.
\]

Os pesos \(\lambda_t\) são obtidos dos `task_weights` configurados e normalizados sobre as tarefas
que o cliente realmente possui. Na configuração 4:1, por exemplo,
\(\lambda_{seg}=0{,}8\) e \(\lambda_{cls}=0{,}2\). Esses pesos são uma decisão do projeto; o artigo
usa pesos iguais como padrão em sua formulação.

O código correspondente encontra-se em:

- [`src/federated/client.py`](../src/federated/client.py), que resolve o conjunto de tarefas e
  normaliza seus pesos;
- [`src/federated/local_trainer.py`](../src/federated/local_trainer.py), que executa o forward,
  seleciona as amostras supervisionadas e combina as losses;
- [`src/dataset/federated_partition.py`](../src/dataset/federated_partition.py), que constrói a
  propriedade conjunta das imagens no cliente multi-task.

### 2.5 Supervisão parcial

O pipeline representa a disponibilidade das supervisões por amostra usando `has_mask` e
`has_label`. Para cada tarefa, a loss é calculada somente nas amostras que possuem o alvo
correspondente. Se nenhuma amostra do batch supervisionar uma determinada tarefa, seu termo é
omitido; ele não é introduzido como loss zero.

Essa distinção é necessária porque ausência de máscara e máscara vazia não são equivalentes. Uma
máscara vazia pode ser um alvo legítimo, enquanto ausência de máscara significa que a tarefa não é
aplicável àquela amostra.

O mascaramento por `has_mask`/`has_label` é uma contribuição de implementação deste projeto. O
FMTL-Bench formula a heterogeneidade principalmente pelo conjunto de tarefas pertencente ao cliente
e não apresenta esse mesmo mecanismo para supervisão parcial arbitrária dentro de um batch.

### 2.6 Exemplo concreto: Curated BUSI

O Curated BUSI permite aplicar diretamente a representação multi-task porque a mesma imagem de
ultrassom pode possuir:

- máscara do tumor para segmentação;
- classe diagnóstica para classificação.

Na topologia multi-task, o pool de imagens é particionado uma única vez. Cada imagem pertence a um
único cliente, e esse cliente executa segmentação e classificação sobre ela. A tabela mestre ainda
pode registrar uma linha por par `(imagem, tarefa)`, mas o dataloader consolida essas linhas para
produzir uma única amostra com os indicadores de supervisão apropriados.

As imagens da classe `normal` constituem uma ressalva: elas possuem classe, mas são excluídas da
tarefa de segmentação pelo protocolo atual. Portanto, mesmo dentro de um cliente BUSI multi-task,
nem toda imagem necessariamente contribui para as duas losses. Isso é tratado pelos indicadores de
supervisão, sem duplicar o cliente ou a imagem.

### 2.7 Fronteira de aplicabilidade: ISIC 2018

No ISIC 2018 disponível neste projeto, segmentação e classificação usam pools disjuntos:

- Task 1 contém imagens com máscaras e sem classes diagnósticas;
- Task 3/HAM10000 contém imagens com classes e sem máscaras;
- a interseção entre os identificadores dos dois pools é zero.

Consequentemente, não existe uma imagem que possa produzir simultaneamente \(\mathcal{L}_{seg}\) e
\(\mathcal{L}_{cls}\) no mesmo forward. O ISIC permanece, por isso, na topologia single-task atual.
Transformá-lo em multi-task exigiria outro desenho experimental, como permitir que um mesmo cliente
possuísse pools de imagens diferentes para cada tarefa. Isso seria diferente da lógica adotada para
o BUSI e não deve ser inferido automaticamente a partir do FMTL-Bench.

### 2.8 Participação federada e agregação

Depois do treinamento local conjunto, o cliente multi-task envia **uma única atualização** do trunk
compartilhado. Os componentes personalizados permanecem locais. No protocolo multi-dataset atual:

- `encoder2..encoder5 + bottleneck` formam o trunk federado;
- `encoder1` permanece personalizado para acomodar modalidades e números de canais diferentes;
- decoders, saídas de segmentação e head de classificação permanecem personalizados;
- a atualização do cliente multi-task é ponderada considerando a massa de supervisão de suas
  tarefas;
- a agregação hierárquica normaliza contribuições dentro de cada dataset antes de aplicar os pesos
  dos datasets.

Essas escolhas são próprias deste projeto. O artigo sustenta a unidade lógica do cliente e a
combinação local das tarefas, mas não prescreve essa divisão FedPer nem a agregação hierárquica.

### 2.9 Antes e depois

| Aspecto | Representação anterior | Representação atual |
|---|---|---|
| Identidade federada | Um cliente lógico por tarefa | Um cliente lógico por silo |
| Cliente com `seg` e `cls` | Registrado duas vezes | Possui \(\mathcal{T}_k=\{seg,cls\}\) |
| Imagens | Reparticionadas/duplicadas entre tarefas | Particionadas uma única vez |
| Forward | Independente para cada cliente/tarefa | Um forward multi-task por batch |
| Loss | Uma loss isolada por participação | Soma ponderada das losses supervisionadas |
| Backward | Um por cliente single-task artificial | Um backward da função objetivo conjunta |
| Atualização enviada | Duas atualizações independentes | Uma atualização multi-task |
| Interação dos gradientes | Somente após a agregação no servidor | Durante a otimização local do cliente |

### 2.10 Delimitação da atribuição científica

Ao citar o artigo na descrição metodológica deste projeto, a afirmação recomendada é:

> Seguindo a formulação de clientes heterogêneos do FMTL-Bench, representamos cada participante
> por um conjunto local de tarefas \(\mathcal{T}_k\). Um participante com segmentação e
> classificação é tratado como um único cliente multi-task, que combina as losses das duas tarefas
> durante a otimização local e produz uma única atualização federada, em vez de ser duplicado como
> dois clientes single-task.

Uma versão mais cautelosa, que registra a ordem histórica correta, é:

> A formulação do FMTL-Bench oferece fundamentação posterior para a representação adotada: clientes
> que possuem mais de uma tarefa mantêm uma única identidade federada e otimizam localmente uma
> função objetivo conjunta sobre seu conjunto de tarefas.

Não devem ser atribuídos ao artigo:

- o uso específico do MTnnUNet;
- a execução simultânea dos heads de classificação e segmentação em um único forward;
- o mascaramento por `has_mask` e `has_label`;
- os pesos 4:1 entre segmentação e classificação;
- a divisão FedPer usada neste projeto;
- a personalização do stem de modalidade;
- a agregação hierárquica por dataset;
- a organização concreta dos datasets BUSI e ISIC.

---

## 3. Síntese

O FMTL-Bench fornece uma formulação útil para distinguir um conjunto de clientes single-task de um
cliente genuinamente multi-task. Na primeira situação, cada participação federada otimiza uma única
tarefa. Na segunda, um único cliente possui várias tarefas, combina suas losses localmente e envia
uma atualização conjunta.

Essa distinção fundamenta posteriormente a mudança realizada no pipeline: imagens BUSI com máscara
e classe deixam de representar duas participações federadas artificiais e passam a alimentar, uma
única vez, um cliente multi-task. O MTnnUNet produz as duas saídas, as losses supervisionadas são
combinadas e um único backward gera a atualização compartilhada. Clientes e datasets que possuem
somente uma tarefa, como os pools disjuntos do ISIC 2018 no protocolo atual, continuam participando
como single-task. Dessa forma, o ambiente federado aceita clientes com diferentes cardinalidades de
\(\mathcal{T}_k\) sem duplicar a identidade dos participantes.
