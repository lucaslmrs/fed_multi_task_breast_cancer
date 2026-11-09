---
name: keep-session-alive
description: Manter a sessão do agente ativa por horas ou dias (acesso remoto pelo Remote Control, treinos longos de fim de semana), com um keep-alive periódico que impede a suspensão do computador e confere, de hora em hora, um treino em andamento. Use quando o usuário pedir para manter a sessão/o computador acordado, acompanhar um treino longo à distância, ou mudar o intervalo do keep-alive. Gatilhos - "manter a sessão ativa", "keep alive", "keep awake", "não deixar a sessão dormir", "vou acessar remotamente", "acompanhe durante o fim de semana".
---

# Manter a sessão ativa

Uma sessão ociosa deixa de responder ao Remote Control e o computador pode suspender. O recurso
que segura isso (`mcp__ccd_host__request_keep_awake`) só vale até ~5 min de ociosidade, então ele
precisa ser renovado por uma tarefa periódica. Esta skill monta esse arranjo em três passos.

Argumento opcional: o intervalo em minutos (padrão **20**) e/ou o arquivo de status a vigiar.

## 1. Segurar a suspensão agora e ligar o Remote Control

- `mcp__ccd_host__request_keep_awake` com `until: "session_idle"` e um `reason` curto.
- Se o usuário vai acompanhar de outro dispositivo, `mcp__ccd_session_mgmt__set_remote_control`
  com `session_id: "self"`, `enabled: true` (ferramentas adiadas: carregue com `ToolSearch`).

## 2. Criar (ou trocar) a tarefa periódica

1. Carregue `CronCreate`, `CronDelete` e `CronList` com `ToolSearch`.
2. `CronList`: se já existir um keep-alive desta sessão, apague-o com `CronDelete` antes de criar
   outro (nunca deixe dois).
3. `CronCreate` com `recurring: true`, num minuto que não seja 0/30. Para N minutos:
   `3-59/N * * * *` (N=10 → `3-59/10`, N=20 → `3-59/20`, N=30 → `3-59/30`).
4. Prompt da tarefa (troque `<STATUS>` pelo arquivo do treino, ou remova a parte do treino se não
   houver nenhum):

   > Keep-alive: chame mcp__ccd_host__request_keep_awake (until session_idle). Se o minuto atual
   > for < N, rode `tail -2 <STATUS>` e verifique se src.training_federated segue vivo; reporte só
   > se houver problema, retomada ou conclusão. Caso contrário, responda em uma linha curta.

5. Para treinos federados com retomada, `<STATUS>` é o `.status` que
   `scripts/run_with_resume.sh` escreve ao lado do run (ex.: `runs/e8_resume/*.status`).

## 3. A cada disparo da tarefa

- Sempre: `request_keep_awake` de novo. Nada mais se o minuto não cair na conferência horária.
- Na conferência: `tail -2 <STATUS>`, `pgrep -f "training_federated --config"` e o último
  `[round N]` do log da tentativa atual. Responda em **uma linha** (hora, braço, round).
- Só escreva mais quando algo mudar: tentativa nova (`ACTOR_DIED` → retomada), `DONE`,
  `GAVE_UP`, processo morto sem `DONE`. Aí investigue o log antes de reportar.
- Não marque wrap-up nem plano nesses disparos; são respostas de uma linha.

## Limites (diga ao usuário uma vez, ao montar)

- A tarefa vive só nesta sessão: some se o app fechar, e expira sozinha em 7 dias.
- O disparo pode atrasar alguns minutos; com intervalo > 5 min há janelas sem a trava de
  suspensão. Para dias seguidos, o mais seguro é a suspensão do Windows em "Nunca" (no WSL, também
  `vmIdleTimeout=-1` em `C:\Users\<user>\.wslconfig`).
- Tampa fechada ou suspensão manual suspendem de qualquer jeito.
- O treino em si roda desacoplado (`nohup setsid`); a sessão só é necessária para acompanhar.
- `/compact` só o usuário pode digitar; a tarefa e os watchers continuam ativos depois dele.

## Desligar

`CronList` → `CronDelete` do keep-alive. A trava de suspensão cai sozinha ~5 min depois.
