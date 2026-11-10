"""Gera paper/artigo_esann2027_v1.docx: primeira versão do artigo (até a metodologia), em português.

CONGELADO desde 10/11/2026: o .docx passou a ser a fonte e é editado diretamente. Este script
só gera a versão inicial e se recusa a sobrescrever um .docx existente.

Rodar com um Python que tenha python-docx (fora do venv do projeto, para não mudar as dependências):
    <python-com-docx> paper/build_artigo_v1.py
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

OUT = Path(__file__).with_name("artigo_esann2027_v1.docx")
if OUT.exists():
    raise SystemExit(f"{OUT} já existe e agora é editado diretamente; não regenerar.")
NOTE = RGBColor(0x8A, 0x1C, 0x3A)  # vinho da paleta do projeto, para as notas de rascunho

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
for name in ("Heading 1", "Heading 2", "Title"):
    doc.styles[name].font.name = "Times New Roman"
    doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)


def para(text, *, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    return p


def note(text):
    p = doc.add_paragraph()
    run = p.add_run(f"[Nota de rascunho] {text}")
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = NOTE
    return p


def equation(text, label):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"{text}        ({label})")
    run.font.name = "Cambria Math"
    return p


def bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()
    return t


# ---------------------------------------------------------------------------------------------
title = doc.add_heading(
    "Diluição, não conflito: aprendizado federado multitarefa entre modalidades de imagem médica",
    level=0,
)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Lucas Rodrigues¹ · [orientadores]", align=WD_ALIGN_PARAGRAPH.CENTER)
para("¹ Programa de Pós-Graduação em Engenharia da Computação (PPGEC), Universidade de Pernambuco",
     align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
note("Versão de 10/11/2026: metodologia completa e resultados até o FedExP. Alvo: ESANN 2027 (6 páginas, em inglês; "
     "submissão 18/11/2026). Os braços de tarefa única estão marcados com [aguardando]. "
     "Título provisório.")

doc.add_heading("Resumo", level=1)
para(
    "No aprendizado federado multitarefa (FMTL) entre domínios, a perda de desempenho do modelo "
    "federado frente ao treino local costuma ser atribuída ao conflito entre os updates de clientes "
    "com tarefas e dados diferentes. Investigamos um cenário com quatro modalidades de imagem médica "
    "(ultrassom de mama, dermatoscopia, ressonância magnética de cérebro e radiografia de tórax), "
    "com segmentação e classificação na mesma federação, num esquema FedPer com agregação "
    "hierárquica. Medindo a geometria da agregação a cada rodada, mostramos que os updates dos "
    "clientes são quase ortogonais: o cancelamento observado não excede o esperado para vetores "
    "ortogonais, e o custo da agregação é a diluição do passo, não o conflito de direção. "
    "Mesmo assim, o modelo federado supera o treino local em sete de oito pares dataset × tarefa "
    "sob o mesmo orçamento. Corrigir a diluição com o passo extrapolado do FedExP multiplica o "
    "passo efetivo por cerca de cinco sem criar conflito, mas não melhora a métrica primária: "
    "muda o perfil de erro da segmentação. [aguardando: ablação de tarefa única.]"
)
note("Resumo a fechar com os números finais. Limite do ESANN: ~150 palavras.")

# ---------------------------------------------------------------------------------------------
doc.add_heading("1. Introdução", level=1)
para(
    "O aprendizado federado (FL) permite que instituições treinem um modelo em conjunto sem "
    "compartilhar dados [McMahan et al., 2017]. Em imagem médica, isso é atraente porque os dados "
    "são sensíveis e cada instituição costuma ter poucos exemplos anotados. Na prática, porém, os "
    "participantes raramente têm o mesmo problema: diferem na modalidade de aquisição, no órgão "
    "estudado e na tarefa (segmentar uma lesão ou classificar um diagnóstico). O aprendizado "
    "federado multitarefa (FMTL) trata esse caso compartilhando parte do modelo entre clientes "
    "heterogêneos e personalizando o restante [Arivazhagan et al., 2019; Smith et al., 2017]."
)
para(
    "O risco é a transferência negativa: o modelo federado ficar pior do que o mesmo algoritmo "
    "treinado só com os dados locais [Wang et al., 2019]. Os trabalhos recentes de FMTL atribuem "
    "esse risco ao conflito entre os updates de clientes com tarefas diferentes e propõem "
    "mecanismos para resolvê-lo, como a agregação com resolução de conflito do FedHCA² "
    "[Lu et al., 2024] e a projeção de gradientes do FedBone [Chen et al., 2024], inspirada na "
    "cirurgia de gradientes do PCGrad [Yu et al., 2020]. Mas esse conflito raramente é medido: "
    "o FedHCA² o invoca na análise teórica, o FedBone o define sem quantificá-lo, o OmniFM "
    "[Liu et al., 2026] o afirma sem medi-lo, e o FMTL-Bench [Yang et al., 2024] não o mede. "
    "No mesmo FMTL-Bench, entre domínios, nenhum método federado supera o treino local de forma "
    "relevante."
)
para(
    "A média de vetores pode encolher por dois motivos diferentes. Se os updates apontam em "
    "direções opostas, eles se cancelam (conflito). Se apontam em direções independentes, quase "
    "ortogonais, a média também encolhe, sem que nenhum cliente atrapalhe o outro: com M clientes "
    "ortogonais de mesma norma, o passo agregado tem 1/√M da norma de cada update (diluição). Os "
    "dois casos pedem remédios distintos. Projetar ou descartar componentes conflitantes não ajuda "
    "quando não há conflito; reescalar o passo do servidor ajuda quando há diluição."
)
para("Este trabalho faz quatro contribuições:")
bullets([
    "um instrumento que separa conflito de diluição na agregação federada, por rodada e por bloco "
    "do modelo, lendo a razão de cancelamento contra a referência ortogonal;",
    "evidência, num cenário com quatro modalidades e duas tarefas, de que o custo da agregação é "
    "diluição e não conflito, e de que mesmo assim o FMTL supera o treino local sob orçamento "
    "fixo;",
    "uma ablação 2×2 (tarefa única × multitarefa, local × federado) que separa o ganho da "
    "federação do ganho do aprendizado multitarefa;",
    "um teste de intervenção, com o passo de servidor extrapolado do FedExP "
    "[Jhunjhunwala et al., 2023], que trata a diluição como causa e não só como correlação.",
])
note("As contribuições 2–4 dependem dos resultados da E9; ajustar a redação ao que os números "
     "sustentarem. Uma seed: tudo é descritivo.")

# ---------------------------------------------------------------------------------------------
doc.add_heading("2. Trabalhos relacionados", level=1)
para(
    "Aprendizado federado personalizado. O FedPer [Arivazhagan et al., 2019] compartilha as "
    "camadas de base e mantém as camadas finais em cada cliente; o FedRep [Collins et al., 2021] e "
    "o FedBN [Li et al., 2021] seguem a mesma ideia de separar o que é comum do que é local. Nosso "
    "arcabouço é um FedPer com stem de entrada e cabeças personalizadas e agregação hierárquica "
    "por dataset."
)
para(
    "Aprendizado federado multitarefa. O FedBone [Chen et al., 2024] e o FedHCA² "
    "[Lu et al., 2024] tratam clientes com tarefas diferentes sobre um backbone compartilhado e "
    "propõem mecanismos contra o conflito entre tarefas. O FMTL-Bench [Yang et al., 2024] compara "
    "métodos de FMTL em silos não-IID e relata que, entre domínios, eles raramente superam o "
    "treino local. O OmniFM [Liu et al., 2026] é o trabalho mais próximo em imagem médica entre "
    "modalidades, mas cada federação tem uma única tarefa. Não encontramos trabalho que combine "
    "segmentação e classificação de modalidades diferentes na mesma federação."
)
para(
    "Conflito e diluição. Em MTL centralizado, o conflito é medido pelo cosseno negativo entre "
    "gradientes de tarefas [Yu et al., 2020]. Em FL, a razão entre a norma da soma e a soma das "
    "normas dos updates aparece como divergência de direção dos updates [Nagura et al., 2020] e "
    "se relaciona à diversidade de gradientes [Yin et al., 2018]. O FedNova [Wang et al., 2020] "
    "corrige a inconsistência causada por números de passos locais diferentes, e o FedExP "
    "[Jhunjhunwala et al., 2023] extrapola o passo do servidor quando os updates são "
    "heterogêneos. Até onde sabemos, nenhum trabalho de FMTL usa essas medidas para decidir se o "
    "custo da agregação é conflito ou diluição."
)

# ---------------------------------------------------------------------------------------------
doc.add_heading("3. Metodologia", level=1)

doc.add_heading("3.1 FedPer multitarefa com agregação hierárquica", level=2)
para(
    "Cada cliente i possui um modelo θᵢ = (φ, ψᵢ), em que φ é o conjunto compartilhado, agregado "
    "pelo servidor, e ψᵢ é a parte personalizada, que nunca sai do cliente: o stem de entrada "
    "(adaptado ao número de canais da modalidade), as saídas de segmentação e a cabeça de "
    "classificação. Em cada rodada t, o servidor envia φᵗ; cada cliente treina τ passos locais a "
    "partir de (φᵗ, ψᵢ) e devolve o update dᵢ = φᵢᵗ⁺¹ − φᵗ."
)
para(
    "Um cliente pode ter uma tarefa (segmentação ou classificação) ou as duas sobre as mesmas "
    "imagens. Neste caso, ele otimiza L = Σₜ λₜ Lₜ, com cada termo calculado só sobre as amostras "
    "que a tarefa supervisiona; uma tarefa sem amostra supervisionada no batch é omitida, nunca "
    "contada como zero."
)
para(
    "O servidor agrega em dois níveis. Dentro de cada dataset, o peso do cliente é proporcional "
    "a Σₜ wₜ·mₜ/n, em que mₜ é o número de amostras do cliente que supervisionam a tarefa t e "
    "wₜ é o peso da tarefa; os pesos são normalizados dentro do dataset. Entre datasets, cada "
    "um recebe a mesma participação (pesos 1:1:1:1), para que um dataset grande não domine o "
    "conjunto compartilhado. O resultado são pesos finais wᵢ, com Σᵢ wᵢ = 1, e o update agregado"
)
equation("d̄ = Σᵢ wᵢ dᵢ,    φᵗ⁺¹ = φᵗ + η_g · d̄", "1")
para("No FedAvg, η_g = 1.")

doc.add_heading("3.2 Separando conflito de diluição", level=2)
para(
    "Medimos, a cada rodada, a razão de cancelamento do conjunto compartilhado sob os pesos reais "
    "da agregação:"
)
equation("C = 1 − ‖Σᵢ wᵢ dᵢ‖ / Σᵢ wᵢ ‖dᵢ‖", "2")
para(
    "C = 0 quando todos os updates são paralelos e C = 1 quando se anulam por completo. O valor "
    "de C sozinho não distingue conflito de diluição, porque updates ortogonais também produzem "
    "C > 0. Por isso, cada valor é lido contra a referência ortogonal C⊥, o cancelamento que os "
    "mesmos pesos e normas produziriam se os updates fossem mutuamente ortogonais (Gram "
    "diagonal). C ≈ C⊥ indica diluição sem conflito; C > C⊥ indica conflito além do esperado. "
    "C também é decomposto em uma parcela intra-dataset e uma inter-dataset, com "
    "(1 − C) = (1 − C_intra)(1 − C_inter). Os cossenos entre os updates de cada par de clientes "
    "são guardados por bloco do modelo, o que permite repetir a leitura por módulo."
)
para(
    "A partir de C definimos o passo efetivo, a fração do passo médio de um cliente que o servidor "
    "de fato aplica:"
)
equation("ρ = η_g · ‖Σᵢ wᵢ dᵢ‖ / Σᵢ wᵢ ‖dᵢ‖", "3")
para("No FedAvg, ρ = 1 − C.")

doc.add_heading("3.3 Extrapolação do passo do servidor (FedExP)", level=2)
para(
    "Se a perda vem da diluição, reescalar o passo agregado deve recuperá-la. O FedExP "
    "[Jhunjhunwala et al., 2023] escolhe η_g a cada rodada a partir da extrapolação do método de "
    "projeções em conjuntos convexos. Generalizamos a fórmula do artigo (Eq. 9) para os pesos da "
    "agregação hierárquica:"
)
equation("η_g = max{ 1,  Σᵢ wᵢ ‖dᵢ‖² / [ 2 ( ‖Σᵢ wᵢ dᵢ‖² + ε ) ] }", "4")
para(
    "Com pesos uniformes (wᵢ = 1/M), a expressão coincide com a original. O fator η_g fica perto "
    "de 1 quando os updates estão alinhados e cresce até cerca de M/2 quando são ortogonais; o "
    "max garante que o servidor nunca ande menos que no FedAvg, e ε evita que η_g exploda "
    "quando o agregado é muito pequeno. Usamos ε = 10⁻³, o valor inicial recomendado pelos "
    "autores. Como os iterados do FedExP oscilam, o modelo final é a média dos dois últimos "
    "iterados globais, como no artigo original. A extrapolação é aplicada só a φ; ψᵢ não muda."
)
note("Observado: η_g médio de 4,34 nas 400 rodadas (de 2,9 no início a ~5,0 no fim; Tabela 5). "
     "Diferença central para o artigo original: aqui há partes personalizadas, treinadas contra "
     "o φ anterior.")

doc.add_heading("3.4 Configuração experimental", level=2)
para(
    "Dados. Usamos quatro datasets públicos, cada um com máscara de segmentação binária e rótulo "
    "de classificação (Tabela 1). No ISIC 2018, as imagens com máscara (Tarefa 1) e as imagens "
    "com diagnóstico (Tarefa 3, HAM10000) são conjuntos disjuntos, então cada cliente do ISIC tem "
    "uma única tarefa. Nos outros três, cada cliente tem as duas tarefas sobre as mesmas imagens. "
    "No TCGA-LGG e no SIIM-ACR, a classe é derivada da máscara (lesão presente ou ausente). "
    "Máscaras vazias são alvos válidos e supervisionam a segmentação."
)
table(
    ["Dataset", "Modalidade", "Classes", "Resolução", "Clientes", "Treino / val. / teste"],
    [
        ["Curated BUSI [Al-Dhabyani et al., 2020]", "Ultrassom de mama", "3", "128 px",
         "2 (seg+cls)", "252 / 63 / 135"],
        ["ISIC 2018 [Codella et al., 2019; Tschandl et al., 2018]", "Dermatoscopia",
         "7 (cls)", "128 px", "4 seg + 4 cls", "seg 2.067 / 518 / 1.109; cls 5.578 / 1.411 / 3.026"],
        ["TCGA-LGG [Buda et al., 2019]", "RM de cérebro", "2", "256 px", "2 (seg+cls)",
         "2.203 / 738 / 988"],
        ["SIIM-ACR [SIIM-ACR, 2019]", "Radiografia de tórax", "2", "256 px", "4 (seg+cls)",
         "6.743 / 1.687 / 3.614"],
    ],
)
para("Tabela 1: datasets, clientes e tamanhos das partições (16 clientes no total).", italic=True,
     size=9)
para(
    "Partição. Uma única partição congelada é usada por todos os braços. Cada dataset é dividido "
    "em 70% para treino e validação e 30% para teste (holdout). No TCGA-LGG, a divisão agrupa por "
    "paciente; no ISIC (classificação), por lesão. O pool de treino é repartido entre os clientes "
    "por Dirichlet com α = 100 (quase IID), e 20% de cada cliente é separado para validação."
)
para(
    "Modelo. Usamos uma U-Net multitarefa com encoder ResNet-34 pré-treinado no ImageNet [He et "
    "al., 2016] e BatchNorm congelada, decoder com supervisão profunda em quatro escalas e uma "
    "cabeça de classificação que funde atributos do encoder e do decoder. O conjunto "
    "compartilhado tem 17 módulos (encoder, gargalo, decoder e blocos de fusão), 85,6% dos "
    "parâmetros; o stem de entrada, as saídas de segmentação e o classificador são "
    "personalizados."
)
para(
    "Treino. Cada cliente faz τ = 10 passos de Adam (lr = 10⁻⁴, batch de 32) por rodada, ao "
    "longo de 400 rodadas, em precisão bf16. A segmentação usa Dice+BCE (0,5/0,5, Dice "
    "calculado sobre o batch), com as saídas profundas ponderadas por 1/(n+1). A classificação "
    "usa entropia cruzada com pesos de classe balanceados. Nos clientes multitarefa, "
    "λ_seg = 0,8 e λ_cls = 0,2. Todos os clientes participam de todas as rodadas. A semente é "
    "1993 em todos os braços, e a inicialização, a ordem dos batches e as transformações são "
    "pareadas entre os braços federado e local."
)
note("Conferir se o otimizador local usa o scheduler plateau dentro do cliente e se vale "
     "mencioná-lo.")

doc.add_heading("3.5 Braços experimentais", level=2)
para(
    "Todos os braços usam a mesma partição, o mesmo modelo e o mesmo orçamento (Tabela 2). O "
    "braço local-only treina cada cliente sozinho, sem nunca receber φ do servidor; é o mesmo "
    "algoritmo sem os dados dos outros clientes, o que torna a diferença federado − local a "
    "medida de transferência negativa de Wang et al. (2019). Os braços de tarefa única removem a "
    "supervisão da outra tarefa na leitura dos dados, sem alterar a partição: os clientes do ISIC "
    "da tarefa removida saem da federação, e os clientes multitarefa mantêm as imagens mas "
    "supervisionam só a tarefa ativa."
)
table(
    ["Braço", "Setup", "Tarefas", "Servidor"],
    [
        ["Local-MTL", "local-only", "seg + cls", "—"],
        ["Fed-MTL (controle)", "federado", "seg + cls", "FedAvg"],
        ["Fed-MTL + FedExP", "federado", "seg + cls", "FedExP"],
        ["Local-seg / Local-cls", "local-only", "só seg / só cls", "—"],
        ["Fed-seg / Fed-cls", "federado", "só seg / só cls", "FedAvg"],
    ],
)
para("Tabela 2: braços do estudo. Todos com 400 rodadas × 10 passos locais e semente 1993.",
     italic=True, size=9)
note("A mistura global/local por camada (E8) empatou com o controle e pode entrar como resultado "
     "secundário ou ficar para a dissertação.")

doc.add_heading("3.6 Métricas e protocolo de análise", level=2)
para(
    "A unidade de análise é a célula dataset × tarefa (oito células). A métrica primária é o Dice "
    "sobre as imagens com lesão na segmentação e a acurácia balanceada na classificação, "
    "calculadas no teste de cada cliente e promediadas entre os clientes do dataset. Como métricas "
    "secundárias, relatamos o Dice global, a taxa de falsos positivos em imagens sem lesão, o "
    "macro-F1 e a AUC. A geometria da agregação é descrita por C, C⊥, C_intra, C_inter, ρ e η_g "
    "por rodada, e a convergência pelo número de rodadas até 95% do valor final de validação."
)
para(
    "Os resultados vêm de uma única semente e de um holdout, e por isso são descritivos. Fixamos "
    "o critério de decisão antes dos treinos, a partir da variação entre três sementes observada "
    "num estudo anterior do mesmo pipeline: diferenças menores que 0,02 na média das oito células, "
    "ou que 0,03 numa célula, são tratadas como empate. Relatamos também o número de clientes em "
    "que cada diferença tem o mesmo sinal. Não usamos testes de significância: os pares "
    "cliente × tarefa não são réplicas independentes."
)
note("O estudo anterior usa outra arquitetura (MTnnUNet a 128 px); dizer isso no texto. "
     "Intervalos por bootstrap nas imagens de teste podem entrar como complemento, deixando claro "
     "que medem a variação da amostra de teste e não a de semente.")

# ---------------------------------------------------------------------------------------------
doc.add_heading("4. Resultados", level=1)
note("Atualizado em 10/11/2026 com os braços Local-MTL, Fed-MTL e FedExP. Os braços de tarefa "
     "única ainda estão em treino; seus espaços estão marcados com [aguardando]. Uma semente e um "
     "holdout: todos os números são descritivos.")

doc.add_heading("4.1 Federado × local e efeito do FedExP", level=2)
para(
    "A Tabela 3 compara os três braços multitarefa no teste. O federado supera o local-only em "
    "sete das oito células, com ganho médio de +0,046 (18 de 24 pares cliente × tarefa na mesma "
    "direção). O maior ganho está na classificação do ISIC (+0,195), onde o local-only ainda "
    "melhorava na rodada 400 enquanto o federado estabilizava perto da rodada 300: sob o mesmo "
    "orçamento, a federação converge mais rápido. Sem essa célula, o ganho médio das outras sete "
    "é de cerca de +0,025."
)
table(
    ["Dataset", "Tarefa", "Local-MTL", "Fed-MTL", "Fed-MTL + FedExP", "FedExP − Fed"],
    [
        ["BUSI", "cls", "0,812", "0,856", "0,844", "−0,012"],
        ["BUSI", "seg", "0,741", "0,737", "0,754", "+0,018"],
        ["ISIC", "cls", "0,368", "0,563", "0,569", "+0,006"],
        ["ISIC", "seg", "0,869", "0,871", "0,876", "+0,005"],
        ["SIIM", "cls", "0,756", "0,804", "0,779", "−0,026"],
        ["SIIM", "seg", "0,218", "0,286", "0,207", "−0,080"],
        ["TCGA", "cls", "0,902", "0,909", "0,897", "−0,012"],
        ["TCGA", "seg", "0,689", "0,696", "0,709", "+0,013"],
        ["Média", "", "0,669", "0,715", "0,704", "−0,011"],
    ],
)
para("Tabela 3: métrica primária no teste (Dice positivo na segmentação, acurácia balanceada na "
     "classificação), média dos clientes de cada dataset.", italic=True, size=9)
para(
    "Pelo critério fixado antes dos treinos, o FedExP empata com o FedAvg na média (−0,011, "
    "abaixo do limiar de 0,02) e supera o controle em 11 de 24 pares. Só uma célula passa do "
    "limiar de 0,03: a segmentação do SIIM, que piora (−0,080). O FedExP continua acima do "
    "local-only (+0,035; 19 de 24 pares)."
)
note("A mistura global/local por camada (E8) também empatou com o controle (média 0,717; +0,002). "
     "Decidir se entra como uma linha desta tabela ou só no texto.")

doc.add_heading("4.2 Geometria da agregação", level=2)
para(
    "A Tabela 4 resume a geometria da agregação nas 400 rodadas. No FedAvg, o cancelamento fica "
    "na referência ortogonal (0,701 contra 0,710): os updates não se opõem mais do que vetores "
    "independentes, e o servidor aplica só 30% do passo médio de um cliente. O FedExP corrige essa "
    "diluição: η_g cresce de 2,9 nas primeiras 50 rodadas para cerca de 5,0 no fim, e o passo "
    "efetivo sobe para 1,51, cerca de cinco vezes o do FedAvg. O cancelamento continua abaixo da "
    "referência ortogonal, ou seja, a extrapolação não introduziu conflito."
)
table(
    ["Braço", "C", "C⊥", "C_intra", "C_inter", "η_g médio", "ρ médio"],
    [
        ["Fed-MTL (FedAvg)", "0,701", "0,710", "0,398", "0,502", "1", "0,30"],
        ["Fed-MTL + FedExP", "0,647", "0,705", "0,349", "0,457", "4,34", "1,51"],
        ["Fed-seg", "[aguardando]", "", "", "", "1", ""],
        ["Fed-cls", "[aguardando]", "", "", "", "1", ""],
    ],
)
para("Tabela 4: médias por rodada da razão de cancelamento (Eq. 2), da referência ortogonal, das "
     "parcelas intra e inter-dataset, do fator do servidor (Eq. 4) e do passo efetivo (Eq. 3).",
     italic=True, size=9)
table(
    ["Rodadas", "1–50", "51–100", "101–150", "151–200", "201–250", "251–300", "301–350", "351–400"],
    [["η_g médio", "2,95", "3,73", "4,15", "4,46", "4,67", "4,91", "4,92", "4,96"],
     ["ρ médio", "1,22", "1,37", "1,46", "1,53", "1,57", "1,62", "1,64", "1,65"]],
)
para("Tabela 5: evolução do fator do servidor e do passo efetivo no braço FedExP.", italic=True,
     size=9)
note("Figura 1 (a fazer): C, C⊥ e ρ por rodada para Fed-MTL, FedExP, Fed-seg e Fed-cls.")

doc.add_heading("4.3 Perfil de erro da segmentação", level=2)
para(
    "A média da Tabela 3 esconde uma mudança no tipo de erro. Com o FedExP, o modelo de "
    "segmentação fica mais conservador: a taxa de falsos positivos em imagens sem lesão cai de "
    "0,58 para 0,16 no BUSI e de 0,23 para 0,07 no SIIM, e o Dice global sobe 0,07 e 0,11, "
    "respectivamente (Tabela 6). Em troca, o modelo deixa passar lesões pequenas, o que explica a "
    "queda do Dice positivo no SIIM, onde o pneumotórax costuma ocupar uma área pequena da "
    "radiografia."
)
table(
    ["Dataset", "Dice global (Fed)", "Dice global (FedExP)", "FP em vazias (Fed)",
     "FP em vazias (FedExP)"],
    [
        ["BUSI", "0,692", "0,766", "0,583", "0,161"],
        ["ISIC", "0,871", "0,876", "—", "—"],
        ["SIIM", "0,666", "0,774", "0,226", "0,065"],
        ["TCGA", "0,832", "0,848", "0,093", "0,075"],
    ],
)
para("Tabela 6: Dice global e taxa de imagens sem lesão com alguma predição positiva. O ISIC não "
     "tem imagens sem lesão no teste.", italic=True, size=9)
note("O BUSI tem ~10 imagens sem lesão por cliente; a queda de 0,58 para 0,16 se apoia em poucos "
     "casos. A hipótese das lesões pequenas no SIIM precisa ser conferida pela área das lesões "
     "perdidas.")

doc.add_heading("4.4 Ablação de tarefa única", level=2)
para(
    "A Tabela 7 separa o efeito da federação do efeito do aprendizado multitarefa. "
    "[aguardando: Fed-seg, Fed-cls, Local-seg e Local-cls]"
)
table(
    ["Dataset", "Tarefa", "Local (única)", "Fed (única)", "Local-MTL", "Fed-MTL"],
    [
        ["BUSI", "seg", "[ ]", "[ ]", "0,741", "0,737"],
        ["ISIC", "seg", "[ ]", "[ ]", "0,869", "0,871"],
        ["SIIM", "seg", "[ ]", "[ ]", "0,218", "0,286"],
        ["TCGA", "seg", "[ ]", "[ ]", "0,689", "0,696"],
        ["BUSI", "cls", "[ ]", "[ ]", "0,812", "0,856"],
        ["ISIC", "cls", "[ ]", "[ ]", "0,368", "0,563"],
        ["SIIM", "cls", "[ ]", "[ ]", "0,756", "0,804"],
        ["TCGA", "cls", "[ ]", "[ ]", "0,902", "0,909"],
        ["Média", "", "[ ]", "[ ]", "0,669", "0,715"],
    ],
)
para("Tabela 7: matriz 2×2 tarefa × federação, mesma métrica primária da Tabela 3.", italic=True,
     size=9)
note("Perguntas que esta tabela responde: (H4) Fed-MTL × Fed-única, o multitarefa ajuda dentro da "
     "federação? (H5) Fed-única × Local-única, a federação ajuda sem multitarefa? Comparar também o "
     "cancelamento de Fed-seg e Fed-cls com o de Fed-MTL (Tabela 4).")

doc.add_heading("4.5 Convergência", level=2)
para("[aguardando: rodadas até 95% do valor final de validação, por braço e célula (H3)]")
note("Figura 2 (a fazer): curvas de validação Fed-MTL × Local-MTL, destacando a classificação do "
     "ISIC.")

doc.add_heading("5. Discussão e conclusão", level=1)
para(
    "Os resultados até aqui sustentam três pontos. Primeiro, num FMTL com quatro modalidades e "
    "duas tarefas, a agregação não gera conflito: o cancelamento fica na referência ortogonal, e "
    "o custo é a diluição do passo. Segundo, mesmo com essa diluição, o FedPer multitarefa supera "
    "o treino local sob orçamento fixo, ao contrário do padrão relatado pelo FMTL-Bench. Terceiro, "
    "corrigir a diluição com o FedExP multiplica o passo efetivo por cerca de cinco sem criar "
    "conflito, mas não melhora a métrica primária: muda o perfil de erro da segmentação, trocando "
    "falsos positivos por lesões pequenas perdidas. Isso indica que, nesta configuração, a "
    "diluição não é o gargalo de desempenho. Uma hipótese é que as partes personalizadas, "
    "treinadas contra o tronco anterior, não acompanham passos maiores do servidor."
)
para("[aguardando: conclusão da ablação de tarefa única]")
note("Limitações a declarar: uma semente; um holdout; ε do FedExP sem busca; limiar de empate "
     "estimado em outra arquitetura; ganho do ISIC cls ligado ao orçamento fixo.")

# ---------------------------------------------------------------------------------------------
doc.add_heading("Referências", level=1)
note("Entradas marcadas [conferir] não estão em docs/references.bib; verificar no PDF antes de "
     "submeter.")
refs = [
    "Al-Dhabyani, W. et al. (2020). Dataset of breast ultrasound images. Data in Brief.",
    "Arivazhagan, M. G. et al. (2019). Federated learning with personalization layers. arXiv.",
    "Buda, M. et al. (2019). Association of genomic subtypes of lower-grade gliomas with shape "
    "features automatically extracted by a deep learning algorithm. Computers in Biology and "
    "Medicine.",
    "Chen, Y. et al. (2024). FedBone: towards large-scale federated multi-task learning.",
    "Codella, N. et al. (2019). Skin lesion analysis toward melanoma detection 2018: a challenge "
    "hosted by the ISIC. arXiv.",
    "Collins, L. et al. (2021). Exploiting shared representations for personalized federated "
    "learning. ICML.",
    "He, K. et al. (2016). Deep residual learning for image recognition. CVPR. [conferir]",
    "Jhunjhunwala, D., Wang, S., Joshi, G. (2023). FedExP: speeding up federated averaging via "
    "extrapolation. ICLR. [conferir: não está no .bib]",
    "Li, X. et al. (2021). FedBN: federated learning on non-IID features via local batch "
    "normalization. ICLR.",
    "Liu et al. (2026). OmniFM: toward modality-robust and task-agnostic federated learning for "
    "heterogeneous medical imaging. CVPR. [conferir]",
    "Lu, Y. et al. (2024). FedHCA²: towards hetero-client federated multi-task learning. CVPR.",
    "McMahan, B. et al. (2017). Communication-efficient learning of deep networks from "
    "decentralized data. AISTATS.",
    "Nagura et al. (2020). FedNNNN. [conferir]",
    "SIIM-ACR (2019). SIIM-ACR Pneumothorax Segmentation. Kaggle.",
    "Smith, V. et al. (2017). Federated multi-task learning. NeurIPS.",
    "Tschandl, P. et al. (2018). The HAM10000 dataset. Scientific Data.",
    "Wang, J. et al. (2020). Tackling the objective inconsistency problem in heterogeneous "
    "federated optimization. NeurIPS.",
    "Wang, Z. et al. (2019). Characterizing and avoiding negative transfer. CVPR. [conferir]",
    "Yang et al. (2024). Federated multi-task learning on non-IID data silos: an experimental "
    "study (FMTL-Bench). [conferir]",
    "Yin, D. et al. (2018). Gradient diversity: a key ingredient for scalable distributed "
    "learning. AISTATS.",
    "Yu, T. et al. (2020). Gradient surgery for multi-task learning. NeurIPS.",
]
for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(9)

doc.save(OUT)
print(OUT)
