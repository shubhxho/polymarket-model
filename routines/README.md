# User routines

Drop a `.py` file here. Any class with `name` and `vote(ctx) -> Vote` is loaded at desk start.

```python
from cmf.routines.base import Ctx, Vote

class MyUsOpen(Routine-like):
    name = "my_us_open"
    enabled = True

    def vote(self, ctx: Ctx) -> Vote:
        # ctx.utc_hour, ctx.asset, ctx.p_ens, ctx.ask, ...
        return Vote(self.name, delta=0.0, note="noop")
```

Built-in: `us_rth` (US regular-hours edge adjustment).
