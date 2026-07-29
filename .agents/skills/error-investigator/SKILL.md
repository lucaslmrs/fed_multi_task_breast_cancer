---
name: error-investigator
description: Investigar a causa raiz de um erro, exceção ou traceback antes de corrigir. Use quando o usuário colar um traceback, relatar um crash, um teste falhando, um resultado numérico inesperado, ou quando uma ferramenta retornar erro em tempo de execução. Gatilhos - "deu erro", "traceback", "exception", "está quebrando", "não roda", "investigar bug", "por que falhou".
---

# Investigador de erros

Diagnostique a causa raiz antes de mexer em qualquer linha. Este repositório produz artefatos
congelados (partições federadas, CSVs de resultado) que uma correção apressada pode invalidar
silenciosamente — o custo de um diagnóstico errado aqui é alto.

## Princípios

- **Nunca assuma nem invente.** Se falta informação para uma conclusão sólida, pergunte antes de prosseguir.
- **Investigue antes de agir.** Leia o código e o contexto relevante antes de propor qualquer modificação.
- **Explique a cadeia causal.** "O erro ocorre porque X, causado por Y, originado em Z."
- **Confirme antes de mudanças impactantes** — fluxo principal, interfaces públicas, esquema de dados ou comportamento observável.

## Fluxo

### 1. Coleta de contexto
- Leia o traceback inteiro. Cada frame pode conter informação relevante; o frame mais profundo raramente é a causa.
- Identifique tipo do erro, arquivo, linha, módulo e a chamada que originou o problema.
- Localize o código envolvido com as ferramentas de busca e leitura.
- Se algo estiver ausente ou ambíguo, **pergunte** — não preencha lacunas com suposições.

### 2. Diagnóstico
- Trace a execução desde a entrada até o ponto de falha.
- Separe **sintoma** de **causa raiz**.
- Verifique as suspeitas frequentes deste repositório antes de partir para hipóteses exóticas:

| Sintoma | Causa provável |
|---|---|
| `FileNotFoundError` em `mapping.csv` / `federated_mapping.csv` | dataset não pré-processado, ou `data.dataset`/`data.variant` apontando para outro lugar. `paths.require_*` falha de propósito em vez de regenerar silenciosamente |
| `[ABORT] This script preprocesses ...` | `data.dataset` no `config.yaml` não bate com o script executado — proteção contra sobrescrever a variante de outro dataset |
| `[ABORT] ... already holds images of size ...` | tentativa de escrever outra resolução dentro de uma variante existente |
| Shape mismatch de canais (1 vs 3) | `federated.share_stem: True` num run misto BUSI+ISIC. O stem precisa ser pessoal |
| `NaN` inesperado nas colunas de máscara | linha do ISIC com `task: cls` — não tem máscara, e `NaN` ali é intencional, não bug |
| Erro de ABI em pandas/monai | `numpy` subiu para `>=2` |
| RAM/VRAM estourando na simulação Flower | `ray_num_cpus` / `client_resources` altos demais |
| Métrica de classificação absurda mas AUC razoável | não é bug: é colapso de limiar por perda ponderada com pouco treino |

### 3. Análise de soluções
- Liste as opções com trade-offs; indique a recomendada e por quê.
- Destaque se a solução muda comportamento, contrato de interface, dados persistidos ou lógica crítica.

### 4. Aprovação antes de implementar
Confirme com o usuário antes de qualquer correção que:
- altere o fluxo principal de execução;
- modifique função/classe usada em múltiplos lugares;
- mude esquema de dados, arquivos de configuração ou parâmetros de I/O;
- **toque em qualquer coisa que regenere ou invalide `federated_mapping.csv`** — isso quebra a comparabilidade entre braços já executados.

Para mudanças localizadas e de baixo impacto (typo, variável local), implemente direto.

### 5. Implementação
- Implemente apenas o que foi acordado.
- Não refatore código não relacionado. Não adicione funcionalidade além do necessário.

### 6. Verificação
- Reproduza o cenário que falhava e confirme que não ocorre mais.
- Rode os testes: `python -m unittest discover -v`.
- Verifique se a mudança não quebrou fluxos relacionados (busque referências).
- Reporte: "Resolvido porque [explicação]. Verifiquei que [evidência]." Se não puder confirmar automaticamente, dê ao usuário o passo exato para validar.

## Restrições

- **NÃO** suponha versões de biblioteca ou comportamento de ambiente sem evidência concreta.
- **NÃO** vá além do escopo do erro reportado.
- **NÃO** marque como resolvido sem a etapa de verificação.
- **NÃO** apague arquivos, remova branches nem execute ações destrutivas sem confirmação explícita.
