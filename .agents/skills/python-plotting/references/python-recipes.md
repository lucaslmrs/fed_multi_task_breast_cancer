# Receitas para gráficos em Python

Use este arquivo como referência de implementação. Adapte nomes, unidades e agregações aos dados reais;
os exemplos não autorizam inventar colunas ou estatísticas.

## Configuração estática reutilizável

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUES_6 = ["#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
EARTH_5 = ["#FCFAE1", "#EDE2B5", "#CABD91", "#6F6357", "#584A47"]
ACCENT_RED = "#9F2B2C"


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
```

Prefira funções que recebam os dados e retornem `Figure`/`Axes`; deixe escrita em disco no ponto de
orquestração. Isso facilita testes e composição em painéis.

## Linha e faixa de incerteza

```python
fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

for label, group in summary.groupby("setup", sort=False):
    group = group.sort_values("round")
    color = {"federated": "#08519C", "local-only": "#6F6357"}[label]
    ax.plot(group["round"], group["mean"], label=label, color=color, linewidth=2)
    ax.fill_between(
        group["round"],
        group["lower"],
        group["upper"],
        color=color,
        alpha=0.16,
        linewidth=0,
    )

ax.set(xlabel="Rodada", ylabel="Dice", title="Desempenho ao longo do treinamento")
ax.legend(title=None)
ax.grid(axis="y", alpha=0.3)
```

Informe o que `lower` e `upper` representam. Para trajetórias de poucas sementes/clientes, considere
mostrar linhas individuais claras por trás da média. Não use faixa de erro se ela não foi calculada.

## Distribuição entre grupos

```python
order = df.groupby("group", observed=True)["value"].median().sort_values().index
fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

sns.boxplot(
    data=df,
    x="group",
    y="value",
    order=order,
    color="#C6DBEF",
    width=0.55,
    fliersize=0,
    ax=ax,
)
sns.stripplot(
    data=df,
    x="group",
    y="value",
    order=order,
    color="#08519C",
    alpha=0.65,
    jitter=0.18,
    size=4,
    ax=ax,
)
ax.set(xlabel="Grupo", ylabel="Métrica (unidade)")
ax.tick_params(axis="x", rotation=30)
```

`stripplot` preserva as observações quando o tamanho amostral permite. Para muitos pontos, use violin,
boxen, ECDF, hexbin ou densidade, evitando uma nuvem ilegível.

## Barras categóricas

```python
ordered = summary.sort_values("value", ascending=True)
colors = ["#9F2B2C" if x == focus else "#6BAED6" for x in ordered["category"]]

fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
bars = ax.barh(ordered["category"], ordered["value"], color=colors)
ax.bar_label(bars, fmt="%.2f", padding=3, color="#584A47")
ax.set(xlabel="Valor (unidade)", ylabel=None)
ax.set_xlim(left=0)
ax.grid(axis="x", alpha=0.3)
```

Não use barras para esconder dispersão. Se as barras forem estimativas, acrescente intervalos e descreva
o estimador. Em barras horizontais, ordenar por valor costuma melhorar comparação; preserve ordem
semântica quando ela existir.

## Dispersão

```python
fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="group",
    style="group",
    palette={"A": "#08519C", "B": "#9F2B2C", "C": "#6F6357"},
    alpha=0.7,
    s=45,
    ax=ax,
)
ax.set(xlabel="X (unidade)", ylabel="Y (unidade)")
ax.legend(title="Grupo", frameon=False)
```

Para sobreposição intensa, reduza `alpha`, use marcadores menores ou agregue por bin/hexbin. Uma linha de
regressão deve vir acompanhada do modelo/escopo apropriado; correlação visual não implica causalidade.

## Heatmap

```python
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("reference_blues", BLUES_6)
fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
sns.heatmap(
    matrix,
    cmap=cmap,
    vmin=0,
    vmax=1,
    annot=matrix.size <= 100,
    fmt=".2f",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label": "Valor"},
    ax=ax,
)
ax.set(xlabel="Coluna", ylabel="Linha")
```

Defina `vmin`/`vmax` pela semântica e mantenha-os iguais entre painéis comparáveis. Para diferenças em
torno de zero, use uma paleta divergente e `TwoSlopeNorm(vcenter=0)`.

## Painéis

```python
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.0, 4.2),
    sharey=True,
    constrained_layout=True,
)

for ax, (dataset, group) in zip(axes, df.groupby("dataset", sort=False), strict=True):
    sns.lineplot(
        data=group,
        x="round",
        y="metric",
        hue="setup",
        palette=COLOR_BY_SETUP,
        errorbar=None,
        ax=ax,
    )
    ax.set_title(dataset)
```

Construa uma única legenda para o painel quando a codificação for compartilhada. Não repita títulos,
eixos e legendas sem necessidade. Se os limites não puderem ser compartilhados, deixe a diferença clara.

## Plotly interativo

```python
import plotly.express as px

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="group",
    symbol="group",
    color_discrete_map={"A": "#08519C", "B": "#9F2B2C", "C": "#6F6357"},
    hover_data={"sample_id": True, "x": ":.3f", "y": ":.3f"},
    labels={"x": "X (unidade)", "y": "Y (unidade)", "group": "Grupo"},
    template="plotly_white",
)
fig.update_layout(legend_title_text="Grupo")
fig.write_html("plot.html", include_plotlyjs=True)
```

Escolha campos de hover deliberadamente; não exponha identificadores sensíveis. Para arquivos menores e
ambiente conectado, `include_plotlyjs="cdn"` pode ser apropriado, mas o HTML deixa de ser autocontido.

## Exportação e teste

- PNG: inspeção, apresentações e documentos raster; use 300 dpi para publicação.
- SVG/PDF: vetor para linhas, texto e edição posterior; valide fontes e transparências no destino.
- HTML: Plotly interativo; confirme se precisa funcionar offline.
- Nomeie arquivos de forma estável e não sobrescreva resultados anteriores sem intenção explícita.
- Abra pelo menos uma saída representativa no tamanho final. Verifique clipping, sobreposição, legenda,
  unidade, escala, ordem das categorias, cores e correspondência com uma pequena amostra dos dados.
- Em geração em lote, feche figuras e teste com dados vazios, uma categoria e valores ausentes quando
  esses casos forem plausíveis.
