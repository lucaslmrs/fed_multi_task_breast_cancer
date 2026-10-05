# To-do — referências

## Como a literatura reporta transferência negativa / cancelamento de updates

Hoje cada run federada reporta a **razão de cancelamento** do trunk compartilhado
(`src/federated/negative_transfer.py`):

- `overall = 1 − ‖Σ wᵢΔᵢ‖ / Σ wᵢ‖Δᵢ‖`, com os deltas do trunk por cliente e os pesos reais da agregação;
- decomposta em `intra` (clientes da mesma base) e `inter` (entre bases), com
  `(1 − overall) = (1 − intra)(1 − inter)`;
- resumida como média de todas as rodadas, média do 1º e do último terço, e média ± desvio entre folds.

Buscar como os autores medem e reportam isso, para alinhar ou justificar a escolha:

- [ ] FedBone — conflito entre updates de tarefas heterogêneas no backbone compartilhado.
- [ ] PCGrad (Yu et al., 2020) — "gradient surgery"; cosseno negativo como definição de conflito.
- [ ] GradNorm / CAGrad / Nash-MTL — métricas de conflito entre tarefas em MTL.
- [ ] Gradient diversity (Yin et al., 2018) — `Σ‖gᵢ‖² / ‖Σ gᵢ‖²`, parente próximo da razão usada aqui.
- [ ] FedMTL / FedPer / HC-FMTL — há alguma métrica de conflito por rodada?
- [ ] Artigos que definem transferência negativa como perda de desempenho em relação ao local-only:
      como relacionam isso a medidas geométricas (cosseno, cancelamento)?
- [ ] Decidir se reportamos por rodada, média final ou curva, e se usamos limiares qualitativos.
