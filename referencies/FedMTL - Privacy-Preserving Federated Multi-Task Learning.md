# FedMTL: Privacy-Preserving Federated Multi-Task Learning

## Referência bibliográfica

Sen, P.; Borcea, C. **FedMTL: Privacy-Preserving Federated Multi-Task Learning**. In: *ECAI 2024 —
27th European Conference on Artificial Intelligence*. Frontiers in Artificial Intelligence and
Applications, v. 392, pp. 1993–2002, 2024.
[DOI: 10.3233/FAIA240715](https://doi.org/10.3233/FAIA240715).

- [PDF na página do autor](https://web.njit.edu/~borcea/papers/ecai24.pdf) — cópia nesta pasta.
- Chave BibTeX: `sen2024fedmtl` em [`docs/references.bib`](../docs/references.bib).

## Escopo desta nota

A nota resume o método, a versão segura e os resultados do artigo, e registra o que ele sugere para
a agregação deste projeto. **Nada do pipeline atual foi derivado do FedMTL.** O artigo foi lido em
2026-10-20.

**Atenção ao nome.** O FMTL-Bench (Yang et al., ICMR 2024) lista um baseline chamado "FedMTL"; ver a
[nota do FMTL-Bench](Federated%20Multi-Task%20Learning%20on%20Non-IID%20Data%20Silos%20-%20An%20Experimental%20Study.md).
O FMTL-Bench saiu antes deste artigo da ECAI (outubro de 2024), então o baseline dele provavelmente
é outro método com o mesmo nome. Isso **não foi confirmado**.

---

## 1. Resumo do artigo

### 1.1 Problema

Cada cliente treina um modelo multi-task sobre um **subconjunto próprio** das tarefas suportadas,
\(\mathcal{U}_i \subseteq \mathcal{U}\), e esse subconjunto varia entre clientes. Os autores apontam
duas falhas das abordagens anteriores:

- O FL convencional supõe o mesmo conjunto de tarefas em todos os clientes. O MAS (Zhuang et al.,
  2023) também supõe que todo cliente tem rótulos de todas as tarefas.
- O FMTL clássico (MOCHA, FedU, FedEM, agrupamento) trata cada cliente como **uma** tarefa.

O modelo local usa hard sharing: camadas compartilhadas \(W^S\) e uma camada própria por tarefa
\(W^{T_k}\). O alvo é um **modelo personalizado por cliente**, \(W^*_i\).

### 1.2 Observação que motiva o método

Os autores treinaram 60 clientes de CelebA em 6 grupos de tarefas (LeNet, 4 tarefas) e projetaram
os parâmetros com t-SNE (Fig. 2):

- **Camadas de tarefa:** os parâmetros da mesma tarefa ficam próximos, mesmo entre grupos que a
  treinam junto com tarefas diferentes.
- **Camadas compartilhadas do topo** (perto das cabeças): agrupam-se pelo conjunto de tarefas do
  cliente.
- **Camadas compartilhadas da base:** não mostram padrão.

Conclusão dos autores: parte do "compartilhado" é moldada pelas tarefas. Por isso, a similaridade
entre clientes deve vir das camadas de tarefa, e o trunk não deve receber o mesmo peso de todos.

### 1.3 Agregação FedMTL

A cada rodada, para cada par de clientes \((i, j)\):

1. **Similaridade entre tarefas.** Calcula o cosseno entre as camadas de tarefa de \(i\) e de \(j\):
   \(A_{i,j}[m][n] = \cos(W^T_i[m], W^T_j[n])\).
2. **Emparelhamento.** O algoritmo húngaro acha o pareamento um-a-um de maior soma \(H_{i,j}\). A
   similaridade do cliente é \(S_{i,j} = H_{i,j}/K_i\), que é assimétrica quando \(K_i \ne K_j\).
3. **Limiar adaptativo.** Zera \(S_{i,j}\) quando ela fica abaixo de \(Z_i\). O limiar sobe
   linearmente ao longo das rodadas (de 0,75 a 0,95 nos experimentos). O cliente aprende primeiro com
   muitos e depois só com os mais parecidos.
4. **Pesos.**
   - Para as camadas compartilhadas: \(p_{i,j} = S_{i,j}\,D_j\), com \(D_j\) o tamanho do dataset
     de \(j\).
   - Para a camada da tarefa \(k\): \(q_{i,j,k} = S_{i,j}\,D_j\) se \(T_k \in \mathcal{U}_j\), e
     0 caso contrário.
5. **Agregação personalizada.**

   \[
   W^{S*}_i = \frac{\sum_j p_{i,j} W^S_j}{\sum_j p_{i,j}}, \qquad
   W^{T_k*}_i = \frac{\sum_j q_{i,j,k} W^{T_k}_j}{\sum_j q_{i,j,k}}
   \]

Com \(S_{i,j}=1\), a fórmula reduz-se ao FedAvg. A similaridade usa os **parâmetros** das camadas de
tarefa, e não o ID da tarefa, porque a mesma tarefa em domínios diferentes gera parâmetros
diferentes.

### 1.4 Versão segura

- **Técnica:** SMPC com compartilhamento aditivo de segredos, entre \(P\) servidores agregadores
  (implementado com o CrypTen).
- **Modelo de ameaça:** servidores honestos mas curiosos; até \(P-1\) deles podem conspirar.
- **O que fica protegido:**
  - os parâmetros de cada cliente;
  - **os tipos de tarefa**, porque toda camada de tarefa é preenchida com zeros até o mesmo tamanho;
  - **o número de tarefas**, porque o cliente pode acrescentar tarefas falsas;
  - o tamanho do dataset.
- **Agregação no domínio secreto:** o húngaro e as somas ponderadas rodam sobre os segredos, e a
  acurácia é a mesma da versão em texto claro.
- **Custo:** cada cliente transmite cerca de \(2P\) vezes o tamanho do modelo por rodada. Com
  \(P=3\), \(N=80\), \(K=8\) e LeNet, a agregação leva ~45 s.

### 1.5 Protocolo experimental

| Item | Valor |
|---|---|
| Dados | CelebA e LFWA (8 tarefas, cada uma com 5 atributos faciais); FaceA = CelebA + LFWA; NYUD2 (4 tarefas de segmentação com 3 classes) |
| Clientes | 60 (CelebA/LFWA), 80 (FaceA), 20 (NYUD2); partição Dirichlet simétrica; 70/15/15 por cliente |
| Tarefas | D1: 2 tarefas por cliente; D2: de 1 a K tarefas por cliente, sorteadas |
| Modelos | LeNet e MobileNetV2 (faces), SegNet (NYUD2) |
| FL | 100% de participação, 1 época local, 20 rodadas, SGD |
| Repetições | Média de 3 execuções |
| Baselines | FedAvg, FedProx, MOCHA, pFedMe, FedFomo, FedAMP (PFLlib, hiperparâmetros padrão) |

### 1.6 Resultados principais

Distribuição D1 (Tabela 3):

| Método | CelebA (acc %) | LFWA (acc %) | FaceA (acc %) | NYUD2 (mIoU %) |
|---|---|---|---|---|
| **FedMTL** | **83,3** | **70,9** | **78,3** | **37,2** |
| FedAMP | 81,9 | 68,4 | 77,1 | 35,4 |
| MOCHA | 81,7 | 67,9 | 76,9 | 34,9 |
| FedFomo | 80,9 | 68,9 | 76,1 | 35,1 |
| FedAvg | 76,0 | 64,7 | 68,5 | 28,9 |

- **Número de tarefas diferente por cliente (D2):** 82,5% (CelebA), 70,5% (LFWA) e 36,7% de mIoU
  (NYUD2). Não há comparação, porque os baselines não suportam esse cenário.
- **Ablações** (FaceA, Fig. 8):
  - **FedMTL-T** (trunk por FedAvg; só as cabeças por similaridade) fica abaixo do FedMTL. O trunk
    compartilhado é prejudicado por clientes com tarefas diferentes.
  - **FedMTL-J/E** (similaridade pelo ID das tarefas) piora quando a mesma tarefa vem de outro
    domínio.
  - Um **limiar fixo** (0,8 ou 0,9) é pior que o limiar crescente.
- **Domínio + tarefa (FaceA):** clientes com as mesmas tarefas mas bases diferentes (CelebA vs LFWA)
  têm camadas de tarefa pouco semelhantes (Fig. 6). A similaridade por parâmetros separa os dois.
- **Leitura das métricas:** a acurácia é média por atributo. O artigo não reporta acurácia
  balanceada nem testes de significância.

---

## 2. Relação com este projeto

### 2.1 Pontos de contato

| Aspecto | FedMTL | Este projeto |
|---|---|---|
| Conjunto de tarefas por cliente | Heterogêneo, \(\mathcal{U}_i \subseteq \mathcal{U}\) | `{seg}`, `{cls}` ou `{seg, cls}` |
| Mesma tarefa, domínios diferentes | FaceA (CelebA + LFWA) | 4 bases de modalidades diferentes |
| Trunk | Personalizado por cliente (pesos \(p_{i,j}\)) | **Um** trunk global, com agregação hierárquica e pesos fixos |
| Cabeças | Agregadas entre clientes com a mesma tarefa, ponderadas pela similaridade | Pessoais (FedPer) |
| Pesos de agregação | Aprendidos a cada rodada pela similaridade | Fixos: `client_weighting`, `task_weights` e `dataset_weights` |
| Métrica de conflito | Matrizes de similaridade (Fig. 6), sem curva por rodada | Cosseno entre deltas e razão de cancelamento, por rodada |

### 2.2 O que o artigo sugere para nós

- **Mesma tarefa não basta para federar.** O FedMTL-J/E falha quando o domínio muda. Isso é
  coerente com o que observamos: o cancelamento entre bases é alto, e o federado fica abaixo do
  local-only mesmo em tarefas iguais (seg de BUSI vs seg de SIIM).
- **Trunk ponderado por similaridade.** O FedMTL-T mostra que agregar o trunk com o mesmo peso para
  todos piora o resultado. Um braço candidato seria um trunk **personalizado por base**, com pesos
  dados pela similaridade entre clientes:
  - a similaridade poderia vir dos parâmetros das cabeças, como no artigo, ou dos deltas que já
    medimos (D14);
  - o limiar subiria ao longo das rodadas.

  Isso é uma hipótese registrada, **não uma decisão**. Mudaria o artefato final (de "um trunk + N
  cabeças" para N trunks), e o FedMTL só foi avaliado com 20 rodadas e modelos pequenos.
- **Cabeças federadas por tarefa.** A E6 já mostrou que federar o decoder ajuda (+0,016 contra o
  controle de 400 rodadas). O FedMTL vai além e federa até as cabeças entre clientes da mesma
  tarefa, ponderando pela similaridade. É um passo possível depois da E6, não testado.

### 2.3 Métrica por rodada (item do To-do)

O FedMTL **não reporta uma métrica de conflito por rodada**. Ele mostra matrizes de similaridade
num instante (Fig. 6) e usa a similaridade só para ponderar a agregação. Isso responde ao item
"FedMTL / FedPer / HC-FMTL" de [`To-do.md`](To-do.md) para este artigo.

### 2.4 Privacidade

A versão segura protege **quais tarefas** cada cliente tem. Neste projeto o conjunto de tarefas de
cada cliente é público na partição, e a simulação não usa agregação segura. Se o projeto chegar a
discutir privacidade das tarefas (por exemplo, um hospital que não quer revelar que não tem
máscaras), o FedMTL é a referência.

### 2.5 O que não atribuir ao artigo

- A topologia `multi_task` e o mascaramento `has_mask`/`has_label`. A formulação do cliente com
  conjunto de tarefas vem do FMTL-Bench.
- A divisão FedPer, a agregação hierárquica e os pesos 4:1.
- O cosseno entre deltas e a razão de cancelamento (D14, D15).
- Qualquer resultado sobre imagens médicas: o FedMTL usa faces e NYUD2.

---

## 3. Síntese

O FedMTL trata clientes com conjuntos de tarefas diferentes e dá a cada cliente um modelo
personalizado. Os pesos de agregação do trunk e das cabeças vêm da similaridade entre as camadas de
tarefa, com um limiar que sobe ao longo das rodadas. As ablações mostram que ponderar o trunk igual
para todos piora o resultado e que o ID da tarefa não basta quando o domínio muda. Para este
projeto, ele é a referência mais próxima de um trunk ponderado por similaridade entre bases, uma
alternativa ainda não testada ao trunk global único.
