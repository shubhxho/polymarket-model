const $ = (id) => document.getElementById(id);

function render(s) {
  $("badge").textContent = s.live_armed ? "LIVE ARMED" : "PAPER";
  $("badge").classList.toggle("live", !!s.live_armed);
  $("cash").textContent = s.paper_cash != null ? `$${Number(s.paper_cash).toFixed(2)}` : "—";
  $("max").textContent = s.max_usd != null ? `$${Number(s.max_usd).toFixed(2)}` : "—";
  $("sdk").textContent = s.sdk || "—";
  $("addr").textContent = s.address || s.reason || "no live key";
  const cards = $("cards");
  cards.innerHTML = (s.cards || [])
    .map(
      (c) => `<article class="card">
        <h2>${c.asset}</h2>
        <div class="row">binance <b>${Number(c.bn || 0).toFixed(2)}</b></div>
        <div class="row">pm mid <b>${Number(c.mid).toFixed(3)}</b></div>
        <div class="row">bid / ask <b>${Number(c.bid).toFixed(3)} / ${Number(c.ask).toFixed(3)}</b></div>
        <div class="row">ensemble <b>${Number(c.p_up).toFixed(3)}</b></div>
        <div class="row">digital / fusion / lag <b>${Number(c.digital||0).toFixed(2)} / ${Number(c.fusion||0).toFixed(2)} / ${Number(c.lag||0).toFixed(2)}</b></div>
        <div class="row">up edge <b>${(100 * Number(c.up_edge)).toFixed(1)}¢</b></div>
        <div class="row">yes+no ask <b>${Number(c.complement||0).toFixed(3)}</b></div>
        <div class="row">tte <b>${Number(c.tte_min).toFixed(1)}m</b></div>
        <div class="act">${c.action}</div>
      </article>`
    )
    .join("") || "<p class='row'>waiting for 15-minute books…</p>";
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
      sdk: "static preview",
      reason: "run uv run cmf desk for live books + model",
      cards: [
        { asset: "BTC", bn: 97420, mid: 0.52, bid: 0.51, ask: 0.53, p_up: 0.61, up_edge: 0.08, tte_min: 8.2, action: "HOLD" },
        { asset: "ETH", bn: 3320, mid: 0.47, bid: 0.46, ask: 0.48, p_up: 0.44, up_edge: -0.04, tte_min: 8.2, action: "HOLD" },
      ],
      fills: [],
      log: [{ ts: "—", msg: "static GitHub preview — start the desk locally to trade" }],
    });
  }
}

$("paper")?.addEventListener("click", async () => {
  await fetch("/api/paper", { method: "POST" });
  pull();
});
$("arm")?.addEventListener("click", async () => {
  const ok = window.confirm("This sends real Polymarket CLOB orders if CMF_LIVE=1 and a key is set. Continue?");
  if (!ok) return;
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
