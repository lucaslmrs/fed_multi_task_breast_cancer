---
name: color-palette
description: Paleta de cores padrão do projeto (preto, vinho, bordô, rosé e creme) para qualquer trabalho visual. Use sempre que for escolher, aplicar, alterar ou revisar cores em gráficos Matplotlib/Seaborn/Plotly, src/utils/plot_style.py, relatórios HTML, o app de monitoramento (src/monitor/static), diagramas, slides ou páginas, e para validar daltonismo e contraste de uma paleta com o script incluído.
---

# Paleta de cores do projeto

Toda saída visual deste repositório usa esta paleta: figuras (`src/utils/plot_style.py`),
relatórios HTML estáticos, o app de monitoramento (`src/monitor/static/`) e qualquer artefato
visual novo (diagrama, slide, página). **Priorize sempre as cinco cores-base.** Cores derivadas só
entram quando uma função não cabe nas cinco (várias séries, status, rampas) e, nesse caso, vêm da
tabela abaixo, nunca de um hex escolhido a olho.

## As cinco cores-base

| Nome (`plot_style`) | Hex | Papel |
|---|---|---|
| `INK_BLACK` | `#0A0B0F` | Fundo da página no tema escuro; texto sobre rosé |
| `WINE_BLACK` | `#1E1216` | Texto no tema claro; superfície/painel no tema escuro; fim da rampa |
| `BURGUNDY` | `#7B2032` | Cor primária: série principal, método proposto, títulos, links, botão, destaque |
| `ROSE` | `#C2A19A` | Texto secundário no tema escuro, preenchimentos, faixas, meio das rampas |
| `CREAM` | `#FFF7F1` | Fundo de figuras, relatórios e do tema claro; texto no tema escuro; centro divergente |

Restrições medidas (não as contorne): rosé tem 2,2:1 sobre creme, então **nunca** é texto, linha
fina ou marcador sobre creme, apenas preenchimento ou faixa. Bordô sobre o painel escuro tem 1,8:1:
no tema escuro a série principal usa o bordô clareado `#B83D54`.

## Regra zero: calcule, não estime

Qualquer mudança de cor categórica, sequencial, divergente, de texto ou de status precisa passar em:

```bash
python .agents/skills/color-palette/scripts/validate_palette.py --css
```

Ele verifica, nos temas claro **e** escuro: faixa de luminosidade OKLCH e croma mínimo, separação
entre pares adjacentes sob protanopia/deuteranopia (Machado 2009, Delta E OKLab x100: alvo >= 8,
piso 6), piso de visão normal (>= 15, bloqueante), contraste WCAG >= 3:1 para marcas e >= 4,5:1
para texto/status, rampas sequenciais monotônicas de matiz único e ponto médio divergente neutro.
Sai com código 1 em qualquer FAIL. Um WARN exige canal secundário (marcador, traço, rótulo direto
ou tabela). O bordô aparece como WARN de luminosidade por escolha sua (fica abaixo da faixa
genérica); as demais checagens se aplicam a ele integralmente. Teste candidatos sem editar código
com `--categorical "#..,#.." --mode dark [--pairs all]`.

## Funções derivadas

**Categórica, ordem fixa** (bordô, azul, ocre, violeta, terracota, ciano, oliva, teal). Atribua
em ordem, sem pular, sem reordenar por valor, sem gerar uma 9ª cor:

| Slot | Claro (sobre `#FFF7F1`) | Escuro (sobre `#1E1216`) |
|---|---|---|
| 1 `PRIMARY` | `#7B2032` | `#B83D54` |
| 2 `ACCENT` | `#2B7EC9` | `#398AD6` |
| 3 | `#A46E01` | `#B37903` |
| 4 | `#8864C0` | `#9470CD` |
| 5 | `#BC5A29` | `#C96736` |
| 6 | `#089AC3` | `#0994BA` |
| 7 | `#7F8002` | `#8C8C06` |
| 8 | `#09A0A0` | `#05A7A7` |

Validado: pior par adjacente sob daltonismo 15,9 (claro) / 16,2 (escuro); visão normal >= 16,9;
contraste de marca >= 3,03:1 / 3,32:1. Os **três primeiros** slots também passam a checagem de
todos os pares (dispersão, small multiples) e não contêm par vermelho-verde; acima disso, facetas
ou "outros".

**Sequencial** `SEQUENTIAL = #FFF7F1, #E0CBC5, #C2A19A, #A16264, #7B2032, #4B1B24, #1E1216`:
creme, rosé, bordô e vinho com pontos médios em OKLab (cmap `palette_sequential`, `_r` invertido).

**Divergente** `DIVERGING = #1D4587, #7E93B9, #FFF7F1, #C2A19A, #7B2032` (azul, creme neutro,
rosé, bordô; cmap `palette_diverging`). Convenção do projeto para matrizes de cosseno/conflito:
**conflito (negativo) no polo bordô**, isto é, `colormap("diverging_r")` com
`TwoSlopeNorm(vcenter=0)`. Use só quando o centro tiver significado.

**Neutros.** Claro: texto `#1E1216`, secundário `#6B5559`, borda/grade `#E6D6D1`, tinta
`#F3E6E1` (cabeçalho de tabela, callout), linha de referência `#9B8385`, painel `#FFFDFB`.
Escuro: página `#0A0B0F`, painel `#1E1216`, painel-2 `#2E1F24`, texto `#FFF7F1`, secundário
`#C2A19A`, borda `#3A2A2E`, referência `#8C7A7C`, link `#E3A3AE`, acento `#B5506A`.

**Status (reservado, sempre com ícone ou rótulo, nunca cor de série):**

| Papel | Claro | Escuro |
|---|---|---|
| ok | `#23690D` | `#60A750` |
| warning | `#8D5E05` | `#C38406` |
| critical | `#B3261E` | `#E26275` |
| info | `#005A9D` | `#4697E4` |

## Mapeamento semântico (estável entre arquivos)

- Federado / método proposto: `PRIMARY` (bordô). Local-only: `MUTED_INK` (recua). Centralizado:
  `ACCENT` (azul). Use `setup_color()`, nunca um literal.
- Loss e série de contraste: `ACCENT`. Dice, balanced accuracy, AUC: `PRIMARY`; IoU, macro-F1:
  slot 5. Use `metric_color()`.
- Overlays de segmentação: acerto `PRIMARY`, falso positivo `ACCENT`, lesão perdida slot 3.
- Traços de contexto (curvas por cliente atrás da média, limites de rodada, referência
  ortogonal): `REFERENCE_LINE`, com alpha se preciso.
- Folds: `fold_style()` (cor + traço + marcador; identidade nunca só pela cor).

## Como aplicar

**Matplotlib/Seaborn.** Importe de `src.utils.plot_style` e decore o construtor com
`@with_plot_style` (`rc_context` escopado; fundo creme). Use `PRIMARY`, `ACCENT`,
`categorical_color(i)`, `colormap("sequential" | "diverging" | "diverging_r")`, `style_axis(ax)`.
Não escreva hex literal em código de plot: crie um token nomeado em `plot_style.py` e valide. A
skill `python-plotting` cuida da forma do gráfico e da integridade; esta skill é dona das cores.
Para uma figura de artigo que exija fundo branco, sobrescreva `figure.facecolor`/`axes.facecolor`
localmente; as séries continuam válidas (contraste sobre branco é maior que sobre creme).

**Relatórios HTML** (`run_report.py`, `analyze.py`, `training_curves.py`,
`negative_transfer.py`, `executive_report.py`, `scripts/*_report.py`): fundo `#FFF7F1`, texto
`#1E1216`, h1 com régua `#7B2032`, h2 e links `#7B2032`, bordas `#E6D6D1`, `th` e `.notice` em
`#F3E6E1` com régua bordô, secundário `#6B5559`, erro `#B3261E`, ok `#23690D`.

**App de monitoramento.** Só variáveis CSS (`var(--text)`, `var(--series-3)`, `var(--critical)`).
Traços Plotly usam `seriesColor(i)` em `app.js`, que lê `--series-N` do tema; heatmaps usam
`DIVERGING_SCALE`. A exportação PNG é sempre clara: `downloadLight` remapeia as séries escuras
para `LIGHT_SERIES`. O tema escuro é um conjunto validado à parte, não uma inversão.

**Qualquer outro visual** (diagrama, slide, página, mockup): fundo creme ou vinho, texto
vinho/creme, bordô como cor de ênfase, rosé para áreas e separadores; recorra à tabela categórica
só quando houver mais de uma categoria a distinguir.

## Alterando a paleta

1. Edite `plot_style.py` e espelhe em `style.css` (os dois blocos escuros) e nas constantes do
   topo de `app.js`.
2. Rode o validador com `--css` e depois `python -m unittest discover -v`.
3. Regere uma figura/relatório afetado e olhe para ele (rótulos sobrepostos, overlays sobre
   ultrassom em tons de cinza, legibilidade da legenda). O validador checa cor, não layout.
4. Atualize as tabelas acima.
