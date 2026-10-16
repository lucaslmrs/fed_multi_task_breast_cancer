# App local de monitoramento e comparação (E2)

```bash
python -m src.monitor --runs runs --port 8765
```

Abra `http://127.0.0.1:8765/`. O servidor é só leitura e escuta apenas em `127.0.0.1` (use
`--host` só se souber o que está expondo). Não há dependência Python nova: é `http.server` da
biblioteca padrão mais uma página estática (`src/monitor/static/`). Os gráficos usam Plotly.js de
CDN com versão fixa. Sem internet, as tabelas continuam funcionando e a página avisa que os
gráficos não carregaram.

## O que mostra

- **Lista (barra lateral).** Estudos → seed → braço, runs avulsos, baselines (B0) e
  validação/smokes. Cada run tem um status e o selo do diagnóstico salvo da E1:
  - `complete`: há `*_test_results.csv`;
  - `running`: sem resultados e com atividade há menos de `--stale-minutes` (padrão 15);
  - `stale`: sem resultados e parado;
  - `pending`: braço do `execution_plan.csv` sem pasta.
- **Run.**
  - Métricas finais por dataset/tarefa (média ± dp entre cliente × fold) e por cliente.
  - Curvas por dataset, uma linha por cliente. A validação é `post_aggregation_round`; o treino é
    a média dos passos locais da rodada.
  - Pesos de agregação, cancelamento contra a referência ortogonal, conflito M1/M2 (com a ressalva
    do Δ inteiro), telemetria de GPU e tempo por fase.
  - Diagnóstico D1–D15 com filtros e links para os relatórios estáticos (`run_report.html`,
    `negative_transfer.html`, `dashboard.html`, `report.html`).
- **Comparar.** Marque dois ou mais runs. A página mostra métricas finais lado a lado com Δ
  contra o primeiro (★) e curvas médias por dataset. Também mostra avisos quando:
  - `local_training.mode` difere;
  - os masters diferem (regra D12);
  - a topologia difere;
  - há run não concluído.

  A comparação é descritiva: nenhum p-valor é calculado.

## Ao vivo

Um run `running` é relido a cada `--refresh` segundos (padrão 10), conforme o `mtime` dos
arquivos. Durante o treino federado, crescem:

- `fold_*/client_*/training_history.csv`;
- a linha por rodada do `execution.log`;
- `gpu_telemetry.csv`.

`aggregation_history.json`, conflito e cancelamento só existem ao fim de cada fold. O botão
**Recalcular em memória** roda o diagnóstico da E1 sobre o estado atual sem gravar nada na pasta do
run. O diagnóstico em `diagnostics/` continua sendo o do fim do treino.

## Limitações

- Memória RSS por ator Ray não é registrada pelo treino (pendência E0.4 (e)).
- Runs antigos sem `training_history.csv` mostram só a linha por rodada do servidor (por
  dataset/tarefa), como o fallback da E1.
