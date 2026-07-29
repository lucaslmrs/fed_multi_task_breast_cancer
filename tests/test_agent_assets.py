"""Agent-facing assets must stay consistent across Claude Code and Codex.

Two failure modes this guards against, both silent:

1. A skill edited in one tool's directory and not the other, so the two agents disagree about
   how to do the same task.
2. `CLAUDE.md` losing its `@AGENTS.md` import (or the import target moving), which would leave
   Claude Code with no project brief at all -- and nothing would visibly break until an agent
   made a decision the brief was supposed to prevent.
"""

import unittest
from pathlib import Path

from scripts.sync_agent_assets import (CANONICAL_SKILLS, CLAUDE_FILE, AGENTS_FILE, IMPORT_LINE,
                                       SKILL_MIRRORS, Problem, canonical_skills,
                                       check_instructions, extract_guard_rails, rendered, sync,
                                       validate_skill)


class TestSkillMirrors(unittest.TestCase):
    def test_canonical_skills_exist(self):
        self.assertTrue(canonical_skills(), "nenhuma skill canônica em .agents/skills")

    def test_every_skill_validates_under_both_tools(self):
        for skill_md in canonical_skills():
            with self.subTest(skill=skill_md.parent.name):
                validate_skill(skill_md)

    def test_mirrors_match_canonical(self):
        names = {p.parent.name for p in canonical_skills()}
        for mirror_root in SKILL_MIRRORS:
            for name in names:
                target = mirror_root / name / "SKILL.md"
                with self.subTest(mirror=mirror_root.name, skill=name):
                    self.assertTrue(
                        target.exists(),
                        f"{target} ausente — rode `python -m scripts.sync_agent_assets`")
                    self.assertEqual(
                        target.read_text(encoding="utf-8"),
                        rendered(CANONICAL_SKILLS / name / "SKILL.md"),
                        f"{target} divergiu — rode `python -m scripts.sync_agent_assets`")

    def test_no_orphan_skills_in_mirrors(self):
        names = {p.parent.name for p in canonical_skills()}
        for mirror_root in SKILL_MIRRORS:
            orphans = {p.parent.name for p in mirror_root.glob("*/SKILL.md")} - names
            with self.subTest(mirror=mirror_root.name):
                self.assertFalse(orphans, f"skills órfãs em {mirror_root}: {sorted(orphans)}")


class TestInstructionImport(unittest.TestCase):
    """The @AGENTS.md link is the one fragile piece of the setup; verify it explicitly."""

    def test_agents_file_is_the_canonical_brief(self):
        self.assertTrue(AGENTS_FILE.exists(), "AGENTS.md ausente na raiz")
        text = AGENTS_FILE.read_text(encoding="utf-8")
        for section in ("## Commands", "## Architecture Overview",
                        "## Federated training", "## Comparison experiment"):
            with self.subTest(section=section):
                self.assertIn(section, text, f"AGENTS.md perdeu a seção {section!r}")

    def test_claude_md_imports_agents_md(self):
        lines = [ln.strip() for ln in CLAUDE_FILE.read_text(encoding="utf-8").splitlines()]
        self.assertIn(IMPORT_LINE, lines,
                      "CLAUDE.md precisa da linha de import '@AGENTS.md' isolada")

    def test_import_target_resolves(self):
        target = (CLAUDE_FILE.parent / IMPORT_LINE.lstrip("@")).resolve()
        self.assertTrue(target.is_file(), f"alvo do import não existe: {target}")

    def test_fallback_instruction_survives(self):
        """If the import silently fails, CLAUDE.md must still tell the agent to read the file."""
        text = CLAUDE_FILE.read_text(encoding="utf-8")
        self.assertIn("Read tool", text,
                      "CLAUDE.md perdeu a instrução de ler AGENTS.md manualmente")

    def test_guard_rails_are_byte_identical(self):
        guards = extract_guard_rails(CLAUDE_FILE)
        self.assertTrue(guards, "bloco de guarda-corpos vazio em CLAUDE.md")
        self.assertIn(guards, AGENTS_FILE.read_text(encoding="utf-8"),
                      "guarda-corpos de CLAUDE.md divergiram de AGENTS.md")

    def test_skill_index_is_complete(self):
        text = AGENTS_FILE.read_text(encoding="utf-8")
        for skill_md in canonical_skills():
            with self.subTest(skill=skill_md.parent.name):
                self.assertIn(skill_md.parent.name, text,
                              "AGENTS.md não indexa esta skill")


class TestSyncCheckMode(unittest.TestCase):
    def test_check_mode_passes(self):
        """The same check the CLI runs, so `--check` and the suite cannot disagree."""
        try:
            sync(check_only=True)
            check_instructions()
        except Problem as exc:  # pragma: no cover - only on real drift
            self.fail(f"assets de agente fora de sincronia: {exc}")


class TestCopilotRemoval(unittest.TestCase):
    def test_no_copilot_agent_definitions(self):
        """Copilot support was dropped; its frontmatter (`tools:`, `model:`) breaks Codex."""
        stale = list((CLAUDE_FILE.parent / ".github").glob("agents/*.agent.md"))
        self.assertFalse(stale, f"definições de agente do Copilot ainda presentes: {stale}")


if __name__ == "__main__":
    unittest.main()
