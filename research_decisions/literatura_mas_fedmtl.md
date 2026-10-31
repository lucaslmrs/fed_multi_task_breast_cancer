# MAS, FedMTL e FMTL-Bench: o que os artigos dizem sobre transferência negativa e conflito

> Leitura feita em 2026-10-27 **diretamente dos PDFs**, sem usar o que o repositório ou o
> pré-projeto dizem sobre eles. As páginas são as do PDF.
>
> - **MAS**: Zhuang, Wen, Lyu, Zhang. *MAS: Towards Resource-Efficient Federated Multiple-Task
>   Learning.* ICCV 2023. arXiv:2307.11285v1.
> - **FedMTL**: Sen, Borcea. *FedMTL: Privacy-Preserving Federated Multi-Task Learning.* ECAI 2024
>   (FAIA v. 392). PDF: `web.njit.edu/~borcea/papers/ecai24.pdf`.

## Resumo em uma linha

Nenhum dos dois **mede** conflito de gradientes nem usa a expressão "gradient conflict". O MAS
usa o termo "negative transfer" e o trata por **afinidade de loss (lookahead, estilo TAG) entre
tarefas**. O FedMTL não usa "negative transfer": fala em "conflicting tasks" e "adverse effects"
e os trata por **similaridade de cosseno entre os pesos das cabeças**.

## MAS

**Cenário.** Várias tarefas de FL **simultâneas** em dispositivos com recursos limitados. A
motivação principal é **tempo de treino e energia**; a transferência negativa é o motivo de não
bastar juntar tudo num só modelo. **Todo cliente tem dados e rótulos de todas as tarefas**: são 32
clientes no Taskonomy, um prédio por cliente, com tarefas densas de visão (segmentação semântica,
profundidade, normais etc.). A heterogeneidade é de dados entre clientes, não de tarefas
(p. 3 §3.1; p. 5 §4.1). O FedMTL critica exatamente esse ponto (FedMTL p. 2).

**O que diz sobre transferência negativa.**
- p. 2, Related Work: cita técnicas contra "the negative transfer problem [22, 60] among tasks":
  soft sharing, NAS e "dynamic loss reweighting strategies [24, 7, 52]". O [52] é o PCGrad; é a
  única citação ligada a gradientes, e aparece como reponderação de loss, não como medida.
- p. 4, §3.3: o treino "all-in-one" "leads to unsatisfactory performance on total test loss,
  because it only considers synergies among tasks, neglecting the negative transfer problems
  [22, 60] in multi-task learning".
- p. 1–2: "not all tasks are beneficial to the others when training together [22, 60]".
- **Conflito de gradientes não aparece** como termo nem como medida.

**Como mede.** Afinidade de uma tarefa αᵢ sobre αⱼ, no cliente k, a cada ρ batches (Eq. 3, p. 4):
`S = 1 − Lⱼ(X, θₛ depois do passo de αᵢ, θⱼ) / Lⱼ(X, θₛ antes, θⱼ)`. Positivo = αᵢ ajuda αⱼ.
- É medida **dentro do cliente, entre tarefas**, sobre os parâmetros compartilhados, e depois
  promediada entre clientes. Não mede interação **entre clientes** nem a agregação.
- Declara-se "inspired by TAG [12]" e lista as diferenças (p. 4): divisão sem sobreposição,
  autoafinidade nova (Eq. 4), média entre clientes e splits inicializados com os pesos do
  all-in-one.
- Usa as afinidades da **10ª rodada** por padrão (p. 7); o melhor ponto de split é R₀ = 30–40
  de 100 rodadas.

**Como mitiga.** Merge (um modelo multitarefa, encoder compartilhado e decoders por tarefa,
FedAvg) → Split em grupos que maximizam a afinidade total, com cada grupo virando um modelo
federado separado (Alg. 1, p. 5). Ou seja: **separa as tarefas em modelos distintos**.

**Evidência que apresenta** (sdnkt, p. 6, perda de teste total, 3 runs):
- One-by-one 0.603; all-in-one FedAvg 0.677; FedProx 0.711; GradNorm 0.691; MAS-2/3/4 0.578 /
  0.555 / 0.548.
- "Simply applying multi-task learning optimization (GradNorm) or federated optimization
  (FedProx) can hardly improve performance" (p. 6).
- Standalone 1.842 contra all-in-one 0.677 e MAS-2 0.578 (p. 8, Fig. 9): no cenário deles,
  qualquer FL supera muito o local.

## FedMTL

**Cenário.** Cada cliente treina um modelo multitarefa para **um subconjunto próprio de tarefas**
(heterogeneidade de tarefas entre clientes), com hard parameter sharing: camadas compartilhadas
Wˢ e uma camada por tarefa Wᵀ (p. 2 §3). Bases: CelebA, LFWA, a união FaceA e NYUD2; 20 rodadas,
1 época local (p. 7–8). O foco do artigo é igualmente **privacidade** (SMPC; §5–6).

**O que diz sobre transferência negativa / conflito.**
- O termo "negative transfer" **não aparece**. Nem "gradient conflict".
- p. 4, §4.2: zera a similaridade abaixo do limiar "to mitigate adverse effects arising from
  highly dissimilar models".
- p. 8, §7.2: FedAvg, FedProx e pFedMe ficam abaixo porque "the accuracy is impacted by the
  aggregation of model parameters from conflicting tasks".
- p. 9, ablação FedMTL-T: agregar as camadas compartilhadas como no FedAvg "fails to perform well
  because shared layers are generic for all clients and may be adversely affected by client
  models trained on dissimilar sets of tasks".
- p. 3, §4.1: por t-SNE, as camadas compartilhadas mais próximas das cabeças carregam assinatura
  da tarefa, "as these parameters are significantly updated by the gradients computed based on
  the losses associated with each task". É a única menção a gradientes, e é **qualitativa**.
- **Nenhuma medida de conflito** é reportada; o conflito é assumido e combatido.

**Como mede a compatibilidade.** Cosseno entre os **parâmetros** (não os deltas) das camadas de
tarefa de dois clientes, para todos os pares de tarefas; algoritmo Húngaro para o emparelhamento de
maior soma; Sᵢ,ⱼ = soma / Kᵢ (Eq. 4–6, p. 4). Compara parâmetros em vez de IDs de tarefa porque
"data from different domains can be used for the same task" (p. 4). Na FaceA, clientes com a
mesma tarefa mas bases diferentes têm cabeças pouco similares (p. 8, Fig. 6a).

**Como mitiga.**
- Sᵢ,ⱼ = 0 se Sᵢ,ⱼ < Zᵢ (Eq. 7). O limiar **não é fixo**: começa em Z_min e sobe
  (1 − Z_min)/R por rodada; nos experimentos, de 0.75 a 0.95 (p. 4 e p. 8). As versões de limiar
  fixo (FedMTL-0.8, 0.9) ficam abaixo (p. 9).
- Peso pᵢ,ⱼ = Sᵢ,ⱼ × Dⱼ para as camadas compartilhadas e qᵢ,ⱼ,ₖ para as cabeças (Eq. 8–9).
- **Tanto o corpo quanto as cabeças são agregados**, e cada cliente recebe **o seu próprio**
  agregado personalizado. Não existe um único corpo global.

**Evidência que apresenta** (Tabela 3, p. 8): FedMTL 83.3 / 70.9 / 78.3 % de acurácia em CelebA /
LFWA / FaceA e 37.2 % de mIoU no NYUD2, contra FedAvg 76.0 / 64.7 / 68.5 / 28.9. Não compara com
treino local isolado.

## O que muda em relação ao que estava escrito antes

| Afirmação anterior (pré-projeto ou conversa) | O que o artigo diz |
|---|---|
| FedMTL exclui clientes "abaixo de um limiar **predefinido**" | O limiar é **adaptativo e crescente** (0.75 → 0.95); o fixo é uma ablação pior. |
| FedMTL adota desacoplamento corpo-cabeça como o FedPer | Usa hard sharing, mas **agrega também as cabeças** e dá a cada cliente um corpo **personalizado**; não há cabeças estritamente locais nem corpo global único. |
| FedMTL trata "transferência negativa" | O termo não aparece; fala em "conflicting tasks" e "adverse effects", sem medir. |
| MAS mede "conflito de gradientes" | Mede **afinidade de loss** (lookahead TAG), entre tarefas dentro do cliente; não fala em conflito de gradientes. |
| MAS: clientes com tarefas diferentes | **Todos os clientes têm todas as tarefas**; a motivação principal é custo de treino. |

## FMTL-Bench (Yang et al., 2024)

> Yang, Lu, Huang, Sirejiding, Lu, Ding. *Federated Multi-Task Learning on Non-IID Data Silos: An
> Experimental Study.* arXiv:2402.12876v2 (16 abr. 2024). Lido do PDF em 2026-10-27.

**O que é.** Um **benchmark**, não um método. São NYUD-v2 e PASCAL-Context, com 4 a 8 clientes,
tarefas densas (profundidade, bordas, normais, segmentação semântica, partes humanas), até 100
rodadas, ResNet-18 ou Swin-T pré-treinados (p. 2–4). São 7 cenários (Fig. 2, p. 3), entre eles:
- **NIID-2 SDST**: cada cliente com uma tarefa diferente (equivale à topologia `single_task`);
- **NIID-6/7 UBCDMT/UBCDST**: domínios diferentes e volumes desbalanceados (o mais próximo do
  cenário multi-dataset deste projeto).

**Baselines** (p. 4, §3.2.3): Local, FedAvg, FedProx, FedAMP, FedRep, PCGrad, CAGrad, MaT-FL e
"FedMTL". **Atenção:** o "FedMTL" deles é a ref. [52], **Smith et al., NeurIPS 2017** (MOCHA),
e **não** o FedMTL de Sen & Borcea. O MAS não é avaliado. FedRep e MaT-FL usam desacoplamento de
parâmetros (só o encoder é transmitido), como o FedPer deste projeto.

**O que diz sobre transferência negativa e conflito.**
- O termo "negative transfer" **não aparece**.
- p. 7–8, §3.4.1: "the PCGrad algorithm, designed to resolve gradient conflict issues, generally
  underperforms". Na tabela do IID-1 (p. 5), o PCGrad fica em −6.6 % (MD) e −23.8 % (TC) contra o
  Local, com ResNet.
- p. 9, §3.4.4: o desacoplamento de parâmetros "can resist optimization direction conflicts from
  the MTL process and reduce model performance losses", além de reduzir a comunicação.
- **Nenhuma medida de conflito** é feita; o benchmark mede só desempenho final, curvas por rodada
  e custo.

**Resultados relevantes** (Δ% contra o Local; G-FL = teste global, P-FL = teste local).
- **Uma tarefa por cliente** (NIID-2, -5, -7): FedAvg, FedProx e CAGrad com decoders por tarefa
  "has extremely poor performance"; FedAMP e FedMTL só conseguem "a slight G-FL improvement
  compared to the Local algorithm" (p. 8). NIID-2 com ResNet: FedAvg −25.7 %.
- **Entre domínios** (NIID-7, ResNet, G-FL, Tab. 5, p. 6): Local 0.00; FedAMP +0.09; FedMTL +0.03;
  FedRep −2.08; MaT-FL −2.15; FedAvg −27.2 %. Conclusão deles (p. 8): "all baselines perform
  significantly worse in most tasks [...] cross-domain tasks pose a challenge for FMTL, thereby
  necessitating design of additional optimization strategies".
- **No IID** (IID-1), FedAvg e FedProx são os melhores (+13.5 %); com mais clientes e menos dados
  por cliente, todos menos FedAvg e FedProx caem (p. 9).
- **Pré-treino** e **backbone maior** melhoram tudo (p. 9).
- Com rótulos balanceados, vários modelos de tarefa única superam um modelo multitarefa local
  (p. 8).
- **Estatística:** usam Wilcoxon com Bonferroni e diagrama de diferença crítica (Nemenyi) e falam
  em "statistically significant" (Fig. 4, p. 7). Isso não vale para este projeto, que proíbe a
  expressão.

## Relação com este projeto (interpretação, não está nos artigos)

- **O FMTL-Bench é a evidência externa mais próxima dos resultados da E3.** No cenário entre
  domínios, nenhum método federado supera o Local de forma relevante, e o próprio artigo pede
  estratégias novas. Isso sustenta que "federado < local-only" é o comportamento esperado da
  literatura, e não um defeito da implementação, e dá um bom ponto de partida para a Etapa 2.
- O FMTL-Bench também mostra o **PCGrad**, um método feito para conflito de gradientes, **piorando**
  o desempenho. Isso é coerente com a E0: se o conflito não é o mecanismo dominante, resolver
  conflito não ajuda. Ainda assim, o artigo **não mede** conflito, então essa ligação é uma
  interpretação.
- O desacoplamento corpo-cabeça (FedRep, MaT-FL) é o que o artigo recomenda contra conflitos. É a
  Etapa 1, que já está feita.
- Nenhum dos três artigos considera a **diluição do passo**.

- O **M2 da E0** usa a mesma família de medida do MAS (lookahead do TAG), mas **entre clientes e
  datasets** e com **controle nulo** e **escala aplicada Δ×wᵢ**. Nenhum dos dois artigos faz isso.
  É uma diferença metodológica defensável.
- O critério do FedMTL é cosseno sobre parâmetros. Na E0, o cosseno entre deltas do tronco deu ≈ 0
  entre datasets; se o mesmo valer para as cabeças, um limiar de 0.75–0.95 isolaria cada dataset,
  o que **agravaria a diluição** em vez de corrigi-la. Precisa ser medido, não presumido.
- Nenhum dos dois artigos considera a **diluição do passo** como mecanismo; ambos assumem que a
  perda vem de somar coisas incompatíveis.
- O MAS mostra FL muito acima do standalone; os seus resultados mostram o contrário (E3). As
  configurações são muito diferentes (32 clientes com todas as tarefas e muitos dados, contra
  4 domínios médicos heterogêneos), e essa diferença é discutível no texto.
