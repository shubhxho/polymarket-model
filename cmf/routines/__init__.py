from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from cmf.routines.base import Ctx, Vote
from cmf.routines.us_session import UsSessionRoutine

ROOT = Path(__file__).resolve().parents[2]
USER_DIR = ROOT / "routines"


def builtin() -> list:
    return [UsSessionRoutine()]


def load_user() -> list:
    found = []
    if not USER_DIR.exists():
        return found
    for path in sorted(USER_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"user_routine_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for obj in vars(mod).values():
            if isinstance(obj, type) and hasattr(obj, "vote") and hasattr(obj, "name"):
                if obj.__name__ in {"Routine"}:
                    continue
                try:
                    inst = obj()
                except Exception:
                    continue
                found.append(inst)
    return found


class RoutineBank:
    def __init__(self) -> None:
        self.items = builtin() + load_user()

    def list(self) -> list[dict]:
        return [{"name": r.name, "enabled": getattr(r, "enabled", True)} for r in self.items]

    def set(self, name: str, enabled: bool) -> None:
        for r in self.items:
            if r.name == name:
                r.enabled = enabled
                return

    def apply(self, ctx: Ctx, p: float) -> tuple[float, list[Vote]]:
        votes = []
        for r in self.items:
            if not getattr(r, "enabled", True):
                continue
            v = r.vote(ctx)
            votes.append(v)
            if v.veto:
                return p, votes
            p = p + v.delta
        return float(max(0.02, min(0.98, p))), votes
