# Paletas e estilo visual

A paleta padrão do projeto é a **Amazon-inspired** definida em `src/utils/plot_style.py` e
documentada na skill `amazon-colors` (`.agents/skills/amazon-colors/SKILL.md`). Aquela skill é a
dona das cores: valores, papéis, modo claro/escuro e o validador. Este arquivo resume o uso em
gráficos; se algo aqui divergir dela, ela vence.

Nunca escreva um hex literal em código de plot: importe o token nomeado de `plot_style.py`. Se
precisar de uma cor nova, adicione-a lá e rode
`python .agents/skills/amazon-colors/scripts/validate_palette.py --css`.

## Tokens principais

```python
from src.utils.plot_style import (
    ACCENT,             # laranja de peso de marca (#B75F0B): série focal, loss, destaque
    AMAZON_ORANGE,      # #FF9900: só preenchimento/faixa; 2,1:1 sobre branco, nunca linha ou texto
    CATEGORICAL_8,      # 8 matizes em ordem fixa (azul, laranja, magenta, ouro, teal, vermelho, roxo, verde)
    DIVERGING,          # azul -> cinza neutro -> laranja
    GRID, INK, MUTED_INK, REFERENCE_LINE, TINT,
    PRIMARY,            # #146EB4: método proposto / série principal
    SEQUENTIAL_BLUE, SEQUENTIAL_ORANGE,
    colormap,           # colormap("blue" | "orange" | "diverging" | "diverging_r")
    categorical_color, fold_style, metric_color, setup_color, split_style, style_axis,
    with_plot_style,
)
```

## Escolha semântica

| Necessidade | Escolha | Observação |
|---|---|---|
| Uma série principal | `PRIMARY` | Título nomeia a série; sem legenda |
| Método proposto + baseline | `setup_color()` (federado `PRIMARY`, local-only `MUTED_INK`) | Varie também marcador/traço |
| Destaque de um item | contexto em `REFERENCE_LINE`, foco em `ACCENT` | O destaque precisa de significado explícito |
| Várias categorias | `categorical_color(i)` na ordem, sem pular | Mais de 8: agrupe em "outros" ou facetas |
| Dispersão/small multiples | só os 3 primeiros slots | Só eles passam a checagem de todos os pares |
| Intensidade contínua | `colormap("blue")` | Claro = menor, escuro = maior |
| Desvio em torno de um centro real | `colormap("diverging")` ou `"diverging_r"` | Conflito (negativo) no polo laranja; `TwoSlopeNorm` |
| Estado (ok/alerta/crítico) | `STATUS["light"]` | Reservado; sempre com rótulo ou ícone |

## Matplotlib e Seaborn

Decore o construtor com `@with_plot_style` (usa `rc_context`, não muda `rcParams` global) e
finalize os eixos com `style_axis(ax)`. Para Seaborn, passe `palette=list(CATEGORICAL_8[:n])` e
`ax=` explicitamente.

## Plotly

```python
import plotly.express as px
from src.utils.plot_style import CATEGORICAL_8, DIVERGING, SEQUENTIAL_BLUE

fig = px.scatter(df, x="x", y="y", color="group",
                 color_discrete_sequence=list(CATEGORICAL_8), template="plotly_white")
continuous = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]
```

No app de monitoramento, use `seriesColor(i)` e as variáveis CSS (`--series-N`), que mudam com o
tema; veja a skill `amazon-colors`.

## Tipografia e densidade

- Use uma família já disponível; não introduza uma fonte externa só por estética.
- Dimensione primeiro a figura final. Para artigo, valide o tamanho em uma coluna ou duas colunas;
  para slides, aumente texto e espessuras.
- Título descreve a conclusão ou o conteúdo; eixos descrevem variáveis e unidades; caption registra
  agregação, intervalo, amostra e fonte quando necessário.
- Rótulos diretos podem substituir legendas em poucas séries. Em muitas séries, reduza o número de
  destaques e deixe o restante como contexto neutro.
- Grid deve ajudar a leitura de valores. Use apenas no eixo relevante e mantenha-o atrás dos dados.

## Acessibilidade e checagem

- A segurança para daltonismo e o contraste da paleta são **calculados** pelo validador da skill
  `amazon-colors`, não avaliados a olho.
- Texto usa `INK`/`MUTED_INK`, nunca a cor da série.
- Para duas ou mais séries importantes, varie também marcador, tracejado ou anotação direta.
- Tons claros das rampas são preenchimento ou faixa de incerteza, não traço fino.
- Em heatmaps, escolha a cor do texto das células pela luminância do fundo ou omita anotações em
  matrizes densas.
- Inspecione o arquivo salvo, não apenas a janela interativa.
