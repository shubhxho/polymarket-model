const $ = (id) => document.getElementById(id);

function render(s) {
  $("badge").textContent = s.live_armed ? "LIVE ARMED" : "PAPER";
  $("badge").classList.toggle("live", !!s.live_armed);
  $("cash").textContent = s.paper_cash != null ? `$${Number(s.paper_cash).toFixed(2)}` : "—";
  $("max").textContent = s.max_usd != null ? `$${Number(s.max_usd).toFixed(2)}` : "—";
  $("sdk").textContent = s.sdk || "—";
  $("addr").textContent = s.address || s.reason || "no live key";
  $("cards").innerHTML = (s.cards || [])
    .map((c) => {
      const yes = Number(c.ask || c.mid || 0.5);
      const no = 1 - Number(c.bid || c.mid || 0.5);
      return `<article class="card">
        <h2>${c.asset} Up or Down · 15m</h2>
        <div class="row">binance <b>${Number(c.bn || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</b></div>
        <div class="row">ensemble / digital / fusion <b>${Number(c.p_up).toFixed(2)} / ${Number(c.digital || 0).toFixed(2)} / ${Number(c.fusion || 0).toFixed(2)}</b></div>
        <div class="row">tte <b>${Number(c.tte_min).toFixed(1)}m</b></div>
        <div class="pair">
          <div class="yn yes"><span>Yes</span><b>${(100 * yes).toFixed(1)}¢</b></div>
          <div class="yn no"><span>No</span><b>${(100 * no).toFixed(1)}¢</b></div>
        </div>
        <div class="act">${c.action}${c.reason ? " · " + c.reason : ""}</div>
      </article>`;
    })
    .join("") || "<p class='hint'>waiting for 15-minute books…</p>";
  $("routines").innerHTML = (s.routines || [])
    .map(
      (r) => `<div class="rt"><span>${r.name}</span>
        <button type="button" data-rt="${r.name}" data-on="${r.enabled ? "0" : "1"}">${r.enabled ? "on" : "off"}</button></div>`
    )
    .join("") || "<p class='hint'>no routines loaded</p>";
  $("fills").innerHTML = (s.fills || [])
    .slice()
    .reverse()
    .map((f) => `<div>${f.venue} ${f.asset} ${f.side} $${Number(f.usd).toFixed(2)} @ ${Number(f.price).toFixed(3)}</div>`)
    .join("");
  $("log").innerHTML = (s.log || [])
    .slice()
    .reverse()
    .map((l) => `<div>${l.ts} ${l.msg}</div>`)
    .join("");
  document.querySelectorAll("[data-rt]").forEach((btn) => {
    btn.onclick = async () => {
      await fetch("/api/routines", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: btn.dataset.rt, enabled: btn.dataset.on === "1" }),
      });
      pull();
    };
  });
}

async function pull() {
  try {
    const r = await fetch("/api/state");
    if (!r.ok) throw new Error("no api");
    render(await r.json());
  } catch {
    render({
      live_armed: false,
      paper_cash: 5,
      max_usd: 5,
      sdk: "preview",
      reason: "uv run cmf desk",
      routines: [{ name: "us_rth", enabled: true }, { name: "example_us_open", enabled: false }],
      cards: [
        { asset: "BTC", bn: 97420, mid: 0.54, bid: 0.53, ask: 0.55, p_up: 0.61, digital: 0.58, fusion: 0.62, tte_min: 8.1, action: "HOLD", reason: "no edge after spread" },
      ],
      fills: [],
      log: [{ ts: "—", msg: "static preview" }],
    });
  }
}

$("paper")?.addEventListener("click", async () => {
  await fetch("/api/paper", { method: "POST" });
  pull();
});
$("arm")?.addEventListener("click", async () => {
  if (!window.confirm("Real CLOB orders if CMF_LIVE=1 and a key is set.")) return;
  const r = await fetch("/api/arm", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ confirm: "I_UNDERSTAND_REAL_ORDERS" }),
  });
  const j = await r.json();
  if (!r.ok) window.alert(j.error || "arm failed");
  pull();
});
pull();
setInterval(pull, 1000);
