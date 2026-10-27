# MAS: Towards Resource-Efficient Federated Multiple-Task Learning

## Referência bibliográfica

Zhuang, W.; Wen, Y.; Lyu, L.; Zhang, S. **MAS: Towards Resource-Efficient Federated Multiple-Task
Learning**. In: *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 23357–23367,
2023. [DOI: 10.1109/ICCV51070.2023.02140](https://doi.org/10.1109/ICCV51070.2023.02140).

- [Texto integral no arXiv (2307.11285)](https://arxiv.org/abs/2307.11285) — PDF nesta pasta.
- Chave BibTeX: `zhuang2023mas` em [`docs/references.bib`](../docs/references.bib).

## Escopo desta nota

A nota resume o método, o protocolo e os resultados do artigo. Em seguida, registra onde ele toca
este projeto: a medida de afinidade entre tarefas (o nosso M2), a leitura de transferência negativa
e a ideia de agrupar tarefas em vez de federar todas juntas. **Nada do pipeline atual foi derivado
do MAS.** O artigo foi lido em 2026-10-20, depois das análises E3, E4, E6 e E7.

---

## 1. Resumo do artigo

### 1.1 Problema

O artigo trata de **várias tarefas de FL simultâneas** que disputam os mesmos dispositivos de borda.
O exemplo é um veículo autônomo que treina detecção de faixa, detecção de objetos e segmentação. Os
autores supõem que:

- cada dispositivo só consegue treinar uma tarefa de FL por vez (\(n_k = 1\));
- os modelos das tarefas compartilham o mesmo backbone e diferem apenas nos decoders;
- **todo cliente tem dados e rótulos de todas as tarefas.** O FedMTL (Sen & Borcea, 2024) critica
  justamente essa hipótese.

O objetivo é maximizar \(\sum_i M(\alpha_i;\omega_i)\), o desempenho somado das tarefas, com o menor
tempo de treino e o menor consumo de energia.

### 1.2 Duas soluções ingênuas e o seu trade-off

- **One-by-one:** treina as tarefas em sequência, cada uma no seu próprio FL. Considera só as
  diferenças entre as tarefas, e é lento.
- **All-in-one:** junta todas as tarefas numa rede multi-task (encoder compartilhado + um decoder por
  tarefa) e treina essa rede em FL. É rápido, mas a perda de teste piora, porque nem toda tarefa
  ajuda as outras: é a transferência negativa.

### 1.3 Método: Merge and Split

1. **Merge.** As \(n\) tarefas viram uma tarefa all-in-one \(\alpha_0\) com modelo
   \(\phi=\{\theta_s\}\cup\{\theta_{\alpha_i}\}\). A loss local é \(\sum_i \mathcal{L}_{\alpha_i}\),
   com pesos 1. A agregação é FedAvg sobre **todos** os parâmetros, decoders incluídos, já que todo
   cliente tem todas as tarefas.
2. **Medição de afinidade.** Durante o all-in-one, cada cliente \(k\) mede, a cada \(\rho\) batches,
   a afinidade da tarefa \(\alpha_i\) sobre \(\alpha_j\) com a definição do TAG (Fifty et al., 2021):

   \[
   S^{k,t}_{\alpha_i\to\alpha_j} = 1 -
   \frac{\mathcal{L}_{\alpha_j}(X^{k,t},\,\theta^{k,t+1}_{s|\alpha_i},\,\theta^{k,t}_{\alpha_j})}
        {\mathcal{L}_{\alpha_j}(X^{k,t},\,\theta^{k,t}_{s},\,\theta^{k,t}_{\alpha_j})}
   \]

   Um valor positivo indica que o passo de \(\alpha_i\) no trunk reduz a loss de \(\alpha_j\). O
   valor é a média sobre os passos, as épocas locais e os \(K\) clientes da rodada.
3. **Self-affinity.** O TAG fixa \(\hat S_{\alpha_i\to\alpha_i}=10^{-6}\), o que impede grupos de uma
   tarefa só. O MAS a substitui pela média normalizada das afinidades de ida e volta com as outras
   tarefas:

   \[
   \hat S_{\alpha_i\to\alpha_i} = \sum_{j\ne i}\frac{\hat S_{\alpha_i\to\alpha_j}+\hat S_{\alpha_j\to\alpha_i}}{2n-2}
   \]
4. **Split.** O servidor escolhe a partição **disjunta** das tarefas em \(x\) grupos que maximiza a
   afinidade total. A busca leva segundos; o TAG não termina 5 grupos de 9 tarefas em uma semana.
5. **Continuação.** Cada grupo continua o treino a partir dos parâmetros do all-in-one, e **não do
   zero**, como faz o TAG.

### 1.4 Protocolo experimental

| Item | Valor |
|---|---|
| Dados | Taskonomy (cenas internas), 32 clientes, um prédio por cliente (4 mil a 16 mil imagens) |
| Conjuntos de tarefas | `sdnkt` (5), `erckt` (5) e `sdnkterca` (9): segmentação, profundidade, normais, keypoints, bordas etc. |
| Modelo | Xception modificada como encoder; decoders de 4 deconv + 4 conv |
| FL | FedAvg, \(K=4\) clientes por rodada, \(E=1\) época, \(R=100\) rodadas |
| Otimização | SGD, momento 0,9, LR 0,1 com decaimento polinomial |
| Métrica | Soma das perdas de teste das tarefas; tempo de GPU; energia (kWh) |
| Repetições | A maioria dos resultados vem de 3 execuções independentes |

### 1.5 Resultados principais

Conjunto `sdnkt` (perda de teste total; quanto menor, melhor):

| Método | Perda de teste | Tempo (GPU·h) | Energia (kWh) |
|---|---|---|---|
| One-by-one | 0,603 ± 0,030 | 16,9 | 8,4 |
| All-in-one (FedAvg) | 0,677 ± 0,018 | 7,3 | 3,7 |
| All-in-one + GradNorm | 0,691 ± 0,013 | 7,8 | 4,1 |
| TAG-2 | 0,624 ± 0,015 | 17,4 | 9,8 |
| **MAS-2** | **0,578 ± 0,015** | 8,8 | 4,9 |
| **MAS-4** | **0,548 ± 0,001** | 12,9 | 6,7 |

- **Trade-off:** o MAS supera o one-by-one com cerca de metade do tempo e ~40% menos energia.
- **Ajuste fino da otimização não basta:** GradNorm e FedProx sobre o all-in-one quase não melhoram.
- **Inicializar os grupos pelo all-in-one** é melhor que treinar do zero em todas as divisões
  testadas (Tabela 1), e reduz a distância entre a melhor e a pior divisão.
- **A afinidade estabiliza cedo:** o padrão aparece já nas primeiras rodadas (Fig. 7), e o padrão
  usa as afinidades da rodada 10.
- **Momento de dividir:** com \(R=100\), o melhor \(R_0\) fica entre 30 e 40. Dividir tarde demais
  apaga o benefício de separar as tarefas.
- **Standalone** (cada cliente sozinho): perda 1,842, contra 0,677 do all-in-one e 0,578 do MAS-2.
- **Mais \(E\) ou mais \(K\)** melhora pouco a partir de certo ponto; o MAS rende mais com ~5× menos
  computação que \(E=5\).

---

## 2. Relação com este projeto

### 2.1 Diferenças de cenário

| Aspecto | MAS | Este projeto |
|---|---|---|
| Tarefas por cliente | Todas, sempre | `{seg}`, `{cls}` ou `{seg, cls}` |
| Domínio | Um dataset (Taskonomy) | 4 bases de modalidades diferentes |
| O que é agregado | Encoder **e** decoders | Só o trunk (E3); trunk + decoders na E6 |
| Personalização | Nenhuma | FedPer: stem e cabeças pessoais |
| Unidade agrupada | Tarefas | — (agrupamento ainda não testado) |
| Métrica | Soma das perdas de teste | Dice positivo e balanced accuracy, por célula |

### 2.2 Precedente federado para o M2

O nosso M2 ([`docs/GRADIENT_CONFLICT_METHODS.md`](../docs/GRADIENT_CONFLICT_METHODS.md)) usa a
mesma afinidade lookahead do TAG. O MAS mostra a mesma medida **calculada dentro dos clientes de um
FL e agregada entre eles**. Serve como precedente para usá-la em FL, mas o par medido é outro:

- no MAS, o par é **tarefa → tarefa dentro de um cliente**, no mesmo batch;
- no nosso M2, o par é **cliente → cliente**, com o delta Δᵢ do cliente i e o batch-sonda do
  cliente j.

O análogo direto do MAS seria a afinidade seg ↔ cls dentro de um cliente BUSI `multi_task`. Isso
não é medido hoje.

### 2.3 Métrica por rodada (item do To-do)

O MAS **reporta a afinidade ao longo das rodadas** (Fig. 7), e não só um valor final. Isso responde
em parte ao item "há alguma métrica de conflito por rodada?" de [`To-do.md`](To-do.md):

- existe uma curva por rodada, mas ela mede afinidade funcional (variação de loss), e não
  cancelamento geométrico;
- o artigo não fixa limiares; ele usa as afinidades só de forma relativa, para escolher a partição.

### 2.4 Ideia candidata: agrupar em vez de federar tudo

Os nossos resultados mostram transferência negativa entre bases: o cancelamento inter-dataset é
alto, e o federado fica abaixo do local-only na E3. O MAS sugere um braço possível:

- medir a afinidade entre **bases** (ou entre tarefas) nas primeiras rodadas;
- federar o trunk só dentro dos grupos com afinidade positiva;
- continuar cada grupo a partir do trunk já federado, e não do zero.

Isso é uma hipótese registrada, **não uma decisão**. Ela mudaria a pergunta do estudo (um trunk
global vs vários trunks), e o ganho do MAS foi medido num único domínio, com todas as tarefas em
todos os clientes.

### 2.5 O que não atribuir ao artigo

- A razão de cancelamento e a sua decomposição intra/inter (D15).
- O cosseno entre deltas como medida de conflito (D14).
- A divisão FedPer, a agregação hierárquica e os pesos 4:1.
- Qualquer resultado sobre imagens médicas: o MAS só usa Taskonomy.

---

## 3. Síntese

O MAS mostra que juntar todas as tarefas num único FL multi-task é rápido mas sofre transferência
negativa, e que separar as tarefas em grupos pela afinidade, medida cedo e continuando dos pesos
compartilhados, dá a menor perda com cerca de metade do tempo. Para este projeto, ele vale como
precedente federado da afinidade lookahead (M2), como exemplo de afinidade reportada por rodada e
como fonte de uma ideia ainda não testada: agrupar bases em vez de federar todas num só trunk.
