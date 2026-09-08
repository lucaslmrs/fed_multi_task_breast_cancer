# Paletas e estilo visual

As cores abaixo foram extraídas dos swatches enviados pelo usuário. Use os códigos exatamente como
definidos para manter consistência entre figuras.

## Paletas-base

```python
BLUES_6 = ["#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
EARTH_5 = ["#FCFAE1", "#EDE2B5", "#CABD91", "#6F6357", "#584A47"]
ACCENT_RED = "#9F2B2C"

# Ordem pensada para categorias sobre fundo claro. Não é uma escala contínua.
CATEGORICAL_8 = [
    "#08519C", "#9F2B2C", "#6F6357", "#3182BD",
    "#584A47", "#6BAED6", "#CABD91", "#9ECAE1",
]
```

### `BLUES_6`

Paleta sequencial do mais claro ao mais escuro. Use para magnitude ordenada, intensidade, frequência,
densidade ou progresso. Em linhas/categorias sobre fundo branco, comece em `#6BAED6` ou mais escuro;
os três primeiros tons funcionam melhor como preenchimento, fundo ou faixa de incerteza.

### `EARTH_5`

Paleta sequencial neutra e quente do marfim ao marrom. Use para contexto, baseline, grupos secundários,
fundos editoriais ou uma escala ordenada discreta. `#FCFAE1` e `#EDE2B5` não oferecem contraste
suficiente para texto pequeno sobre branco. `#584A47` é o tom preferido para texto escuro dentro desta
família.

### `ACCENT_RED`

`#9F2B2C` é um destaque único. Use para chamar atenção a uma série focal, erro, alerta, referência ou
diferença importante. Não use uma lista de tons vermelhos inventados para completar categorias; combine
o vermelho com azuis e neutros quando a semântica pedir contraste.

## Escolha semântica

| Necessidade | Escolha inicial | Observação |
|---|---|---|
| Uma série principal | `#08519C` | Alta legibilidade em fundo claro |
| Série principal + referência | `#08519C`, `#6F6357` | Referência neutra, sem competir |
| Método proposto + baseline | `#08519C`, `#584A47` | Adicione marcador ou traço distinto |
| Destaque de um item | contexto em `#CABD91`, foco em `#9F2B2C` | Vermelho deve ter significado explícito |
| Intensidade contínua | `BLUES_6` | Claro = menor, escuro = maior |
| Contexto editorial quente | `EARTH_5` | Evite textos nos tons claros |
| Valores abaixo/acima de um centro | azul escuro → neutro claro → vermelho | Só use se houver centro semântico real |

Para uma escala divergente, uma opção coerente com as referências é:

```python
DIVERGING_BLUE_RED = ["#08519C", "#6BAED6", "#FCFAE1", "#CABD91", "#9F2B2C"]
```

Defina explicitamente o centro (`TwoSlopeNorm` no Matplotlib, `color_continuous_midpoint` no Plotly).
Não trate uma paleta divergente como sequencial.

## Matplotlib e Seaborn

```python
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

BLUES_6 = ["#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
EARTH_5 = ["#FCFAE1", "#EDE2B5", "#CABD91", "#6F6357", "#584A47"]
ACCENT_RED = "#9F2B2C"

blues_cmap = LinearSegmentedColormap.from_list("reference_blues", BLUES_6)
earth_cmap = LinearSegmentedColormap.from_list("reference_earth", EARTH_5)

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette=["#08519C", ACCENT_RED, "#6F6357", "#3182BD", "#584A47"],
)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.labelcolor": "#584A47",
    "text.color": "#584A47",
    "xtick.color": "#6F6357",
    "ytick.color": "#6F6357",
    "grid.color": "#CABD91",
    "grid.alpha": 0.35,
    "legend.frameon": False,
    "savefig.facecolor": "white",
})
```

Não altere globalmente `rcParams` dentro de uma biblioteca reutilizável. Nesse caso, use
`matplotlib.rc_context` ou um contexto Seaborn para limitar o estilo à figura.

## Plotly

```python
import plotly.express as px

categorical = ["#08519C", "#9F2B2C", "#6F6357", "#3182BD", "#584A47"]
continuous_blues = [
    [0.00, "#EFF3FF"],
    [0.20, "#C6DBEF"],
    [0.40, "#9ECAE1"],
    [0.60, "#6BAED6"],
    [0.80, "#3182BD"],
    [1.00, "#08519C"],
]

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="group",
    color_discrete_sequence=categorical,
    template="plotly_white",
)
```

Para categorias persistentes, prefira um dicionário explícito a uma sequência posicional:

```python
COLOR_BY_SETUP = {
    "federated": "#08519C",
    "local-only": "#6F6357",
    "highlight": "#9F2B2C",
}
```

## Tipografia e densidade

- Use uma família já disponível; não introduza uma fonte externa só por estética.
- Dimensione primeiro a figura final. Para artigo, valide o tamanho em uma coluna ou duas colunas;
  para slides, aumente texto e espessuras.
- Título descreve a conclusão ou o conteúdo; eixos descrevem variáveis e unidades; caption registra
  agregação, intervalo, amostra e fonte quando necessário.
- Rótulos diretos podem substituir legendas em poucas séries. Em muitas séries, reduza o número de
  destaques e deixe o restante como contexto neutro.
- Grid deve ajudar a leitura de valores. Use apenas no eixo relevante e mantenha-o visualmente atrás
  dos dados.

## Acessibilidade e checagem

- Garanta contraste suficiente entre marca e fundo. Tons claros são preenchimentos, não traços finos.
- Para duas ou mais séries importantes, varie também marcador, tracejado ou anotação direta.
- Verifique a figura em escala de cinza quando a separação entre séries for crítica.
- Evite combinações em que azul e vermelho sejam a única distinção sem rótulos ou formas.
- Em heatmaps, escolha a cor do texto das células conforme a luminância do fundo ou omita anotações
  quando a matriz for densa.
- Inspecione o arquivo salvo, não apenas a janela interativa: exportação pode cortar rótulos ou mudar
  fontes e proporções.
