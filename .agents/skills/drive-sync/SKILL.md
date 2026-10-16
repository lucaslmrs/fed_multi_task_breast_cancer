---
name: drive-sync
description: Espelhar o repositório no Google Drive com rclone (remote `gdrive:`, pasta `fed_mtl/current`), e configurar o rclone pela primeira vez numa máquina nova. Use ao fazer backup, sincronizar runs/código com o Drive, verificar se há algo pendente para sincronizar, ou restaurar o projeto do Drive. Gatilhos - "sync", "rclone", "sincronizar com o drive", "backup no drive", "há algo para sincronizar".
---

# Sync com o Google Drive (rclone)

Destino: `gdrive:fed_mtl/current` — espelho do repo filtrado por `.rclone-filter` (na raiz,
versionado; a mesma cópia é enviada ao Drive). O filtro exclui `.venv`, `.git`, `__pycache__`,
`data.zip`, `data/*/{raw,archives,processed_128,curated}` e checkpoints `runs/**/*.pt`.
Os datasets pré-processados ficam como `.tar` em `gdrive:fed_mtl/packs/`, fora do sync.

Rode tudo a partir da raiz do repo **dentro do WSL** (é lá que o rclone está configurado).

## Fluxo

1. **Dry-run primeiro, sempre** — `sync` apaga no Drive o que não existe mais localmente:

   ```bash
   rclone sync . gdrive:fed_mtl/current --filter-from .rclone-filter --dry-run --log-file /tmp/rclone_dry.log
   grep -c "Skipped copy" /tmp/rclone_dry.log; grep -c "Skipped delete" /tmp/rclone_dry.log
   ```

2. **Se houver deleções, mostre ao usuário quais pastas somem e peça confirmação.** Runs antigos
   apagados localmente podem existir só no Drive. Alternativa sem deleção: `rclone copy` com os
   mesmos argumentos. Deleções vão para a lixeira do Drive (padrão do backend `drive`).

3. **Sync real** (pode passar de 10 min com runs grandes — rode em background):

   ```bash
   rclone sync . gdrive:fed_mtl/current --filter-from .rclone-filter --stats-one-line -v --log-file /tmp/rclone_sync.log
   ```

   Confira `tail /tmp/rclone_sync.log`; não confie só na saída do terminal (via `wsl.exe` ela
   pode vir vazia).

## Ao relatar

- Um treino em andamento (`pgrep -af training_`) continua gravando em `runs/`: avise que um novo
  sync será necessário quando ele terminar.
- `Can't follow symlink` em `.claude/skills/*` é esperado: o conteúdo real está em `.agents/skills/`.
- `ERROR : Attempt 2/3 succeeded` com exit 0 significa sucesso após retry.
- "Há algo para sincronizar?" = só o dry-run; reporte as contagens de cópias/deleções.

## Primeira configuração (máquina nova)

```bash
sudo apt install rclone            # ou: curl https://rclone.org/install.sh | sudo bash
rclone config                      # n → nome: gdrive → storage: drive → client_id/secret em branco
                                   # → scope: 1 (drive) → resto padrão → auto config:
                                   #   sem navegador (WSL/servidor): "n" e rode no Windows
                                   #   `rclone authorize "drive"`, colando o token de volta
rclone lsd gdrive:fed_mtl          # deve listar current/ e packs/
```

Restaurar numa máquina nova: `git clone` do repo, depois
`rclone copy gdrive:fed_mtl/current . --filter-from .rclone-filter` para os runs/artefatos e
`rclone copy gdrive:fed_mtl/packs /tmp/packs` + `tar -xf /tmp/packs/<pack>.tar` **na raiz do
repo** (os caminhos internos começam em `data/`). Os `mapping.csv` usam caminhos relativos à raiz,
então funcionam sem regenerar.
