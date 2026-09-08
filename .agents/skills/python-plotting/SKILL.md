---
name: python-plotting
description: Criar, revisar ou padronizar gráficos e visualizações de dados em Python com Matplotlib, Seaborn ou Plotly. Use para figuras científicas, curvas de treino, distribuições, comparações, relações, heatmaps, painéis e gráficos interativos. Não use para diagramas conceituais, edição de imagens ou interfaces web completas.
---

# Gráficos em Python

Produza uma visualização que responda à pergunta analítica, preserve a semântica dos dados e seja
legível no meio de destino. Entregue o código reproduzível junto do artefato quando o pedido envolver
criação ou alteração de arquivos.

## Antes de plotar

1. Inspecione colunas, tipos, unidades, categorias, valores ausentes e cardinalidade. Não invente dados
   nem silencie observações inválidas.
2. Identifique a mensagem principal, o público e o destino: exploração, artigo, apresentação ou web.
   Se isso não estiver explícito e não mudar materialmente o resultado, adote uma figura estática limpa
   para leitura em tela e registre a suposição.
3. Escolha o gráfico pela relação a comunicar, não pela biblioteca disponível:
   - tendência/tempo: linha; não conecte categorias sem ordem;
   - relação entre variáveis: dispersão, com transparência ou agregação se houver sobreposição;
   - distribuição: pontos + box/violin, ECDF ou histograma; não esconda a distribuição em barras de média;
   - comparação categórica: pontos ou barras ordenadas; barras começam em zero;
   - matriz: heatmap com escala e limites semanticamente definidos;
   - composição: barras empilhadas; use pizza apenas para poucas partes e quando ângulos forem suficientes.

## Biblioteca

- Use **Matplotlib** para figuras estáticas, publicação e controle fino. Prefira a API orientada a
  objetos (`fig, ax = plt.subplots()`).
- Use **Seaborn sobre Matplotlib** para distribuições, relações estatísticas e facetas. Passe `ax=`
  explicitamente quando possível e finalize a figura pela API do Matplotlib.
- Use **Plotly** apenas quando hover, zoom, filtros, animação ou compartilhamento HTML forem parte do
  resultado. Evite milhões de pontos; agregue, faça amostragem justificada ou use renderização adequada.

Não instale ou atualize bibliotecas sem necessidade e autorização. Use primeiro o ambiente Python já
configurado no projeto.

## Paletas e estilo

Leia sempre [references/palettes-and-style.md](references/palettes-and-style.md) antes de criar ou
restilizar uma figura. Ele contém as paletas fornecidas pelo usuário, seus usos semânticos, mapeamentos
consistentes entre bibliotecas e restrições de contraste.

Regras essenciais:

- cor deve codificar informação; não a use apenas como ornamento;
- preserve o mesmo mapeamento categoria-cor entre painéis e arquivos relacionados;
- não dependa apenas de vermelho versus azul: combine cor com marcador, traço, rótulo ou posição;
- use o vermelho `#9F2B2C` para destaque, alerta ou contraste planejado, não como ciclo categórico padrão;
- reserve tons muito claros para fundo, faixa de incerteza ou preenchimento; não os use para texto fino
  sobre fundo branco;
- para escalas contínuas, use paleta sequencial; para desvios em torno de um centro real, construa uma
  escala divergente com ponto neutro explícito.

## Integridade analítica

- Mostre observações brutas quando forem legíveis. Se agregar, indique o estimador e a incerteza na
  legenda, rótulo ou caption.
- Não atribua causalidade, significância ou generalidade que a análise não sustenta. Uma anotação visual
  não amplia o escopo inferencial dos dados.
- Não corte eixos para exagerar diferenças. Se um recorte for necessário para leitura, torne-o evidente.
- Não suavize séries por padrão. Quando suavizar, mantenha ou disponibilize a série original e documente
  método e janela.
- Em painéis comparáveis, compartilhe limites de eixo quando isso não ocultar estrutura importante.
- Rótulos devem incluir unidade; legendas devem explicar codificações, não repetir o título.

## Implementação e entrega

Para padrões de código e tipos de gráfico, leia
[references/python-recipes.md](references/python-recipes.md) somente quando precisar implementar ou
revisar uma figura.

Ao gerar uma figura estática:

1. dimensione para o destino antes de ajustar fontes e espessuras;
2. use `constrained_layout=True` ou ajuste deliberadamente o layout;
3. salve PNG a 300 dpi para inspeção e SVG/PDF quando o destino aceitar vetor;
4. use `bbox_inches="tight"` e feche a figura em geração em lote;
5. abra o artefato gerado e verifique cortes, sobreposição, contraste, ordem, legenda e fidelidade dos dados.

Para Plotly, salve HTML autocontido quando o usuário precisar compartilhar interatividade; exporte uma
imagem estática apenas se o ambiente já tiver o renderizador necessário.

## Utilitário de paletas

Execute `scripts/preview_palettes.py --output <arquivo.png>` para gerar uma folha de referência das
paletas e validar que Matplotlib consegue renderizá-las. O script não é necessário para gráficos comuns;
ele existe para inspeção visual e consistência.
