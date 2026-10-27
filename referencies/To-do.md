# To-do — referências

## Como a literatura reporta transferência negativa / cancelamento de updates

Hoje cada run federada reporta a **razão de cancelamento** do trunk compartilhado
(`src/federated/negative_transfer.py`):

- `overall = 1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖`, com os deltas do trunk por cliente e os pesos reais da agregação;
- decomposta em `intra` (clientes da mesma base) e `inter` (entre bases), com
  `(1 − overall) = (1 − intra)(1 − inter)`;
- resumida como média de todas as rodadas, média do 1º e do último terço, e média ± desvio entre folds.

Buscar como os autores medem e reportam isso, para alinhar ou justificar a escolha:

- [x] FedBone — conflito entre updates de tarefas heterogêneas no backbone compartilhado.
      → `cos < 0` (Def. 1), sem métrica por rodada. Ver `docs/DECISION_SOURCES.md` (D14).
- [x] PCGrad (Yu et al., 2020) — "gradient surgery"; cosseno negativo como definição de conflito.
      → Verificado em `docs/GRADIENT_CONFLICT_METHODS.md` (M1).
- [ ] GradNorm / CAGrad / Nash-MTL — métricas de conflito entre tarefas em MTL.
- [x] Gradient diversity (Yin et al., 2018) — `Σ‖gᵢ‖² / ‖Σ gᵢ‖²`, parente próximo da razão usada aqui.
      → Def. 4, AISTATS 2018. Nossa razão é uma adaptação (D15).
- [ ] FedMTL / FedPer / HC-FMTL — há alguma métrica de conflito por rodada?
      → FedMTL (Sen & Borcea, 2024): não; só matrizes de similaridade num instante, usadas para
        ponderar a agregação. Ver `FedMTL - Privacy-Preserving Federated Multi-Task Learning.md`.
      → MAS (Zhuang et al., 2023), lido junto: afinidade TAG por rodada (Fig. 7), funcional e sem
        limiares. Ver `MAS - Towards Resource-Efficient Federated Multiple-Task Learning.md`.
      → Falta: FedPer e HC-FMTL.
- [ ] Artigos que definem transferência negativa como perda de desempenho em relação ao local-only:
      como relacionam isso a medidas geométricas (cosseno, cancelamento)?
- [x] Decidir se reportamos por rodada, média final ou curva, e se usamos limiares qualitativos.
      → Curva por rodada ao lado da referência ortogonal, sem limiares (nenhuma fonte os oferece). D15.
