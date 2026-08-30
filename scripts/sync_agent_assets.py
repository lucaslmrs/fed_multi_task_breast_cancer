#!/usr/bin/env python
"""Keep the agent-facing assets consistent across Claude Code and Codex.

Both tools read the SAME skill format -- `<skill>/SKILL.md` with YAML frontmatter carrying
`name` and `description` -- but they discover skills in different directories:

    .claude/skills/<name>/SKILL.md    Claude Code
    .codex/skills/<name>/SKILL.md     Codex

So the skills are authored once under `.agents/skills/`. Each discovery directory contains
relative symlinks to those canonical skill directories, avoiding duplicated instructions while
also exposing future `scripts/`, `references/`, `assets/`, and `agents/` resources to both tools.

Instructions follow the other half of the same split: `AGENTS.md` is the canonical project brief
(Codex reads it natively, hierarchically) and `CLAUDE.md` pulls it in with an `@AGENTS.md` import.
That import is the one fragile link in the setup, so this script verifies it rather than trusting
it: the import line must be present, its target must exist, and the guard-rail block that CLAUDE.md
keeps inline as a fallback must appear byte-identically inside AGENTS.md.

Usage:
    python -m scripts.sync_agent_assets            # write the mirrors
    python -m scripts.sync_agent_assets --check    # verify only; non-zero exit on drift
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_SKILLS = REPO_ROOT / ".agents" / "skills"
SKILL_MIRRORS = (REPO_ROOT / ".claude" / "skills", REPO_ROOT / ".codex" / "skills")

AGENTS_FILE = REPO_ROOT / "AGENTS.md"
CLAUDE_FILE = REPO_ROOT / "CLAUDE.md"
IMPORT_LINE = "@AGENTS.md"

# The guard-rails are mirrored inline in CLAUDE.md on purpose: if the @import ever stops
# resolving, the invariants whose violation destroys frozen artifacts must still be in context.
# Duplication is only safe because this script proves the two copies are identical.
GUARD_START = "<!-- GUARD-RAILS:START -->"
GUARD_END = "<!-- GUARD-RAILS:END -->"

# Frontmatter rules enforced by BOTH tools. Codex's validator rejects any key outside this set
# (a Copilot-style `tools:` key, for instance, fails), so the mirrors must stay within it.
ALLOWED_KEYS = {"name", "description", "license", "allowed-tools", "metadata"}
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


class Problem(Exception):
    """A drift or validation failure worth failing the build over."""


# --------------------------------------------------------------------------- frontmatter

def parse_frontmatter(text: str, where: Path) -> dict:
    """Minimal top-level YAML mapping reader.

    Deliberately not PyYAML: this must run in `--check` mode from a bare test process without
    assuming the project's dependencies are installed.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise Problem(f"{where}: sem frontmatter YAML delimitado por '---' no topo do arquivo")

    fields: dict[str, str] = {}
    key = None
    for raw in match.group(1).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[0] not in " \t":                      # top-level key
            if ":" not in raw:
                raise Problem(f"{where}: linha de frontmatter sem ':' -> {raw!r}")
            key, _, value = raw.partition(":")
            key = key.strip()
            fields[key] = value.strip().strip('"').strip("'")
        elif key is not None:                        # continuation / nested block
            fields[key] = (fields[key] + " " + raw.strip()).strip()
    return fields


def validate_skill(skill_md: Path) -> str:
    """Validate one SKILL.md against the rules both tools apply. Returns the skill name."""
    text = skill_md.read_text(encoding="utf-8")
    fields = parse_frontmatter(text, skill_md)

    unknown = set(fields) - ALLOWED_KEYS
    if unknown:
        raise Problem(
            f"{skill_md}: chave(s) de frontmatter não suportada(s): {sorted(unknown)}. "
            f"Permitidas: {sorted(ALLOWED_KEYS)}")

    for required in ("name", "description"):
        if not fields.get(required):
            raise Problem(f"{skill_md}: falta '{required}' no frontmatter")

    name = fields["name"]
    if not NAME_RE.match(name):
        raise Problem(
            f"{skill_md}: name '{name}' precisa ser hyphen-case "
            f"(minúsculas, dígitos e hifens simples)")
    if len(name) > MAX_NAME_LEN:
        raise Problem(f"{skill_md}: name tem {len(name)} caracteres (máximo {MAX_NAME_LEN})")
    if name != skill_md.parent.name:
        raise Problem(
            f"{skill_md}: name '{name}' difere do diretório '{skill_md.parent.name}'")

    description = fields["description"]
    if len(description) > MAX_DESCRIPTION_LEN:
        raise Problem(
            f"{skill_md}: description tem {len(description)} caracteres "
            f"(máximo {MAX_DESCRIPTION_LEN})")
    # Codex's own validator rejects these outright; Claude tolerates them, so this rule would
    # otherwise only surface as a skill that silently fails to load on one of the two tools.
    if "<" in description or ">" in description:
        raise Problem(
            f"{skill_md}: description não pode conter '<' nem '>' (regra do validador do Codex). "
            f"Escreva o caminho sem os sinais, ex.: 'data/DATASET/VARIANT/'")

    body = text[FRONTMATTER_RE.match(text).end():]
    if not body.strip():
        raise Problem(f"{skill_md}: corpo vazio")

    return name


def canonical_skills() -> list[Path]:
    if not CANONICAL_SKILLS.is_dir():
        raise Problem(f"diretório canônico ausente: {CANONICAL_SKILLS}")
    found = sorted(p for p in CANONICAL_SKILLS.glob("*/SKILL.md"))
    if not found:
        raise Problem(f"nenhum SKILL.md encontrado em {CANONICAL_SKILLS}")
    return found


def expected_link(mirror_root: Path, name: str) -> Path:
    """Return the relative link stored at `<mirror_root>/<name>`."""
    return Path("../..") / CANONICAL_SKILLS.relative_to(REPO_ROOT) / name


# --------------------------------------------------------------------------- instructions

def extract_guard_rails(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start, end = text.find(GUARD_START), text.find(GUARD_END)
    if start == -1 or end == -1:
        raise Problem(
            f"{path.name}: bloco de guarda-corpos ausente "
            f"(precisa conter {GUARD_START} ... {GUARD_END})")
    if end < start:
        raise Problem(f"{path.name}: marcadores de guarda-corpos fora de ordem")
    return text[start + len(GUARD_START):end].strip()


def check_instructions() -> list[str]:
    """Verify the CLAUDE.md -> AGENTS.md link and its fallback. Returns human-readable notes."""
    notes = []
    if not AGENTS_FILE.exists():
        raise Problem("AGENTS.md ausente na raiz — é o arquivo canônico de instruções")
    if not CLAUDE_FILE.exists():
        raise Problem("CLAUDE.md ausente na raiz")

    claude_text = CLAUDE_FILE.read_text(encoding="utf-8")

    if not any(line.strip() == IMPORT_LINE for line in claude_text.splitlines()):
        raise Problem(
            f"CLAUDE.md não contém a linha de import '{IMPORT_LINE}' isolada — "
            f"sem ela o Claude Code não carrega o conteúdo de AGENTS.md")
    notes.append(f"import '{IMPORT_LINE}' presente e resolvendo para {AGENTS_FILE.name}")

    if "Read" not in claude_text or "AGENTS.md" not in claude_text:
        raise Problem("CLAUDE.md perdeu a instrução de fallback para ler AGENTS.md manualmente")
    notes.append("fallback explícito de leitura manual presente")

    claude_guards = extract_guard_rails(CLAUDE_FILE)
    agents_text = AGENTS_FILE.read_text(encoding="utf-8")
    if claude_guards not in agents_text:
        raise Problem(
            "os guarda-corpos de CLAUDE.md não aparecem literalmente em AGENTS.md — "
            "as duas cópias divergiram; alinhe-as")
    notes.append("guarda-corpos idênticos nos dois arquivos")

    for skill in canonical_skills():
        name = skill.parent.name
        if name not in agents_text:
            raise Problem(f"AGENTS.md não menciona a skill '{name}' — índice desatualizado")
    notes.append("índice de skills em AGENTS.md completo")

    return notes


# --------------------------------------------------------------------------- sync

def sync(check_only: bool) -> list[str]:
    notes = []
    names = [validate_skill(p) for p in canonical_skills()]
    notes.append(f"{len(names)} skills válidas em .agents/skills: {', '.join(sorted(names))}")

    for mirror_root in SKILL_MIRRORS:
        label = mirror_root.relative_to(REPO_ROOT).as_posix()
        mirror_root.mkdir(parents=True, exist_ok=True)
        present = {p.name for p in mirror_root.iterdir()}
        stale = present - set(names)

        if check_only:
            for name in names:
                target = mirror_root / name
                link = expected_link(mirror_root, name)
                if not target.is_symlink():
                    raise Problem(
                        f"{label}/{name} não é link para a skill canônica — rode sync_agent_assets")
                if Path(target.readlink()) != link:
                    raise Problem(
                        f"{label}/{name} aponta para '{target.readlink()}', esperado '{link}'")
                if not (target / "SKILL.md").is_file():
                    raise Problem(f"{label}/{name} é um link quebrado ou não contém SKILL.md")
            if stale:
                raise Problem(f"{label}: skills órfãs {sorted(stale)} — rode sync_agent_assets")
            notes.append(f"{label}: em dia")
        else:
            for name in sorted(stale):
                target = mirror_root / name
                target.unlink() if target.is_symlink() or target.is_file() else shutil.rmtree(target)
                notes.append(f"{label}: removida skill órfã '{name}'")
            for name in names:
                target = mirror_root / name
                link = expected_link(mirror_root, name)
                if target.is_symlink() and Path(target.readlink()) == link:
                    continue
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.exists():
                    shutil.rmtree(target)
                target.symlink_to(link, target_is_directory=True)
                notes.append(f"{label}: vinculado '{name}' -> {link}")
            notes.append(f"{label}: sincronizado")

    notes.extend(check_instructions())
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="apenas verifica; sai com código != 0 se houver divergência")
    args = parser.parse_args()

    try:
        for note in sync(check_only=args.check):
            print(f"  {note}")
    except Problem as exc:
        print(f"[FALHA] {exc}", file=sys.stderr)
        if args.check:
            print("        rode `python -m scripts.sync_agent_assets` para regenerar.",
                  file=sys.stderr)
        return 1

    print("OK" if args.check else "Sincronizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
