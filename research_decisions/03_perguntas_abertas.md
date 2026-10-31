# Perguntas abertas (sessão grill-me)

O que o repositório não responde e decide o recorte. Cada resposta é registrada com data, e
`02_opcoes_de_recorte.md` e `arvore_de_decisoes.html` são atualizados.

## Respondidas (2026-10-26)

| # | Pergunta | Resposta |
|---|---|---|
| Q1 | O que diz o projeto aprovado? | Pré-projeto FedCHAMP, nesta pasta. "Algumas coisas mudaram ao longo do caminho, mas a direção seguiu a mesma: mitigação da transferência negativa." |
| Q2 | Prazo para a dissertação escrita? | 4–6 meses. |
| Q3 | Resultado negativo bem explicado é aceitável? | Substituída por Q8 depois da leitura do projeto. |
| Q8 | Como H1 entra diante da E0 (sem cancelamento entre datasets; custo = diluição)? | **Ampliar o conceito.** Transferência negativa = o federado rende menos que o standalone, por **cancelamento** (direção) **ou diluição** (magnitude). A mitigação precisa endereçar os dois. |
| Q9 | Implementar a Etapa 2 (protótipos + clusterização)? | **Não agora.** Esta pasta só organiza o pensamento e simplifica o projeto; nenhuma implementação nesta fase. A Etapa 2 segue como ramo aberto. |
| Q11 | O que sai do escopo? | **Saem:** PANDA + detecção; varredura de backbones (YOLO/VGG/MobileNet/ViT/Swin). **Nada já implementado é descartado** (E4, E6, E7 ficam como ramos). MAS não foi cortado. |
| Q12/Q13 | Orientadores sabem das mudanças? Qualificação feita? | Sabem; a qualificação já foi feita. O recorte pode ser decidido aqui e comunicado. |

## Em aberto

| # | Pergunta | Recomendação |
|---|---|---|
| Q10 | Quais baselines da literatura entram: FedMTL, MAS, os dois? | FedMTL primeiro (mais próximo do FedCHAMP); MAS se sobrar tempo. Ver `literatura_mas_fedmtl.md`: o MAS supõe todas as tarefas em todo cliente e o FedMTL agrega também as cabeças, então nenhum encaixa direto (= P3 em `escopo_e_decisoes.md`). |
| Q14 | Com o conceito ampliado, qual mecanismo ataca a **diluição**? Os protótipos sozinhos só decidem *quem* agregar com quem. | Ramo novo na árvore; candidatos: peso maior para o próprio cluster na fase global, trunk por cluster, ou a seed por dataset (já reduz a diluição alinhando os updates). |
| Q15 | Qual é a métrica operacional de transferência negativa na dissertação? | A lacuna federado − standalone por dataset/tarefa (já medida na E3), decomposta pelo instrumental da E0 (cancelamento vs diluição). |
| Q16 | E4, E6 e E7 ficam — mas em qual papel? | Como hipóteses de mitigação da diluição/lacuna (E6 compartilha mais, E7 parte de uma representação melhor), não como frentes independentes. |
| Q17 | O cluster oráculo (dataset) basta como "agrupamento" ou os protótipos são obrigatórios para o título? | Pergunta para a próxima reunião com os orientadores. |

As pendências P1–P5 de `escopo_e_decisoes.md` (versão B) cobrem as mesmas questões: Q9/Q14/Q17 ↔ P1,
Q10 ↔ P3, Q16 ↔ P4. P2 (medir H2) e P5 (coleta online de Δ×wᵢ) só existem lá.
