"""US cash-session routine.

RTH is 13:30–20:00 UTC. During those hours US equities and the related
crypto-beta are denser; require a fatter edge. Overnight, allow the lag
trade more freely. Drop more files in /routines to extend this base.
"""

from __future__ import annotations

from cmf.routines.base import Ctx, Vote


class UsSessionRoutine:
    name = "us_rth"
    enabled = True

    def vote(self, ctx: Ctx) -> Vote:
        minutes = ctx.utc_hour * 60 + ctx.utc_minute
        rth = ctx.weekday < 5 and 13 * 60 + 30 <= minutes < 20 * 60
        if rth:
            return Vote(self.name, delta=-0.02, note="US RTH: demand 2¢ more edge")
        if ctx.weekday >= 5:
            return Vote(self.name, delta=0.01, note="weekend: lag more persistent")
        return Vote(self.name, delta=0.015, note="US off-hours: allow lag trade")
