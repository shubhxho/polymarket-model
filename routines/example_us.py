"""Example user routine. Copy and edit. Loaded automatically from this folder."""

from cmf.routines.base import Ctx, Vote


class ExampleUsOpen:
    name = "example_us_open"
    enabled = False

    def vote(self, ctx: Ctx) -> Vote:
        # 13:30–14:00 UTC = first half-hour of US cash
        minutes = ctx.utc_hour * 60 + ctx.utc_minute
        if ctx.weekday < 5 and 13 * 60 + 30 <= minutes < 14 * 60:
            return Vote(self.name, delta=-0.03, note="first 30m US: fade")
        return Vote(self.name, 0.0, note="")
