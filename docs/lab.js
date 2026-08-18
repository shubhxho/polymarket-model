function erf(x) {
  const s = Math.sign(x);
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  return s * (1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
}
function phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
function digital(s, k, tau, vol) {
  tau = Math.max(tau, 1e-6);
  vol = Math.max(vol, 1e-8);
  const d2 = (Math.log(s / k) - 0.5 * vol * vol * tau) / (vol * Math.sqrt(tau));
  return Math.min(0.98, Math.max(0.02, phi(d2)));
}

function episode(lag, vol, enter, usrth, weekday) {
  const n = 240;
  let s = 100, path = [s];
  for (let i = 1; i < n + 16; i++) {
    s *= Math.exp(-0.5 * vol * vol + vol * (Math.random() * 2 - 1));
    path.push(s);
  }
  const open = path[0];
  const resolved = path[path.length - 1] > open ? 1 : 0;
  let pos = 0, entry = 0, pnl = 0, trades = 0;
  for (let t = 16; t < n; t++) {
    const spot = path[t];
    const tau = n - t;
    const dig = digital(spot, open, tau, vol);
    const lead = (spot - path[t - 8]) / path[t - 8];
    const lagP = Math.min(0.98, Math.max(0.02, dig + 6 * lead * (lag / 8)));
    let p = 0.55 * dig + 0.45 * lagP;
    const hour = 14;
    if (usrth && weekday < 5) p -= 0.02;
    const ask = Math.min(0.99, p + 0.012 + Math.abs(lead) * 2);
    const bid = Math.max(0.01, p - 0.012);
    const up = p - ask;
    const dn = 1 - p - (1 - bid);
    let act = 0;
    if (pos === 0) {
      if (up > enter) act = 1;
      else if (dn > enter) act = -1;
    }
    if (act !== 0 && pos === 0) {
      pos = act;
      entry = act > 0 ? ask : 1 - bid;
      trades += 1;
    }
  }
  if (pos !== 0) {
    const exit = pos > 0 ? resolved : 1 - resolved;
    pnl = (exit - entry) / entry;
  }
  return { pnl, trades, win: pnl > 0 };
}

function run() {
  const N = +document.getElementById("episodes").value;
  const lag = +document.getElementById("lag").value;
  const vol = +document.getElementById("vol").value * 1e-4;
  const enter = +document.getElementById("enter").value / 1000;
  const usrth = document.getElementById("usrth").checked;
  const pnls = [];
  let trades = 0, wins = 0, eq = [0], acc = 0;
  for (let i = 0; i < N; i++) {
    const e = episode(lag, vol, enter, usrth, i % 7);
    pnls.push(e.pnl);
    trades += e.trades;
    if (e.win) wins += 1;
    acc += e.pnl;
    eq.push(acc);
  }
  const mean = acc / N;
  const sd = Math.sqrt(pnls.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(N - 1, 1));
  const sharpe = mean / (sd + 1e-8) * Math.sqrt(N);
  document.getElementById("pnl").textContent = (mean >= 0 ? "+" : "") + mean.toFixed(3);
  document.getElementById("sh").textContent = sharpe.toFixed(2);
  document.getElementById("wr").textContent = (100 * wins / N).toFixed(1) + "%";
  document.getElementById("tr").textContent = (trades / N).toFixed(2);
  const c = document.getElementById("eq");
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const min = Math.min(...eq), max = Math.max(...eq);
  g.strokeStyle = "#1f6b62";
  g.lineWidth = 1.5;
  g.beginPath();
  eq.forEach((y, i) => {
    const x = (i / (eq.length - 1)) * (c.width - 8) + 4;
    const yy = c.height - 8 - ((y - min) / (max - min + 1e-9)) * (c.height - 16);
    i ? g.lineTo(x, yy) : g.moveTo(x, yy);
  });
  g.stroke();
}

["episodes", "lag", "vol", "enter"].forEach((id) => {
  const el = document.getElementById(id);
  const map = { episodes: "ev", lag: "lv", vol: "vv", enter: "en" };
  const fmt = { enter: (v) => (v / 10).toFixed(1), vol: (v) => v, lag: (v) => v, episodes: (v) => v };
  const paint = () => { document.getElementById(map[id]).textContent = fmt[id](+el.value); };
  el.addEventListener("input", paint);
  paint();
});
document.getElementById("run").addEventListener("click", run);
run();
