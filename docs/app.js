function ticks(n, lag) {
  const out = [];
  let p = 100;
  for (let i = 0; i < n; i += 1) {
    p *= 1 + (Math.sin(i / 9) * 0.0008 + (i % 17 === 0 ? 0.0012 : 0));
    const delayed = i < lag ? 100 : 100 * Math.exp(Math.log(p / 100) * 0.92);
    const mid = 1 / (1 + Math.exp(-18 * (delayed / 100 - 1)));
    out.push({ i, p, mid });
  }
  return out;
}

function renderTape() {
  const data = ticks(80, 7);
  const fast = data
    .map((d) => {
      const cls = d.p >= 100 ? "up" : "dn";
      return `<span class="${cls}">BN ${d.p.toFixed(2)}</span>`;
    })
    .join("");
  const slow = data
    .map((d) => `<span>PM ${d.mid.toFixed(3)}</span>`)
    .join("");
  const a = document.getElementById("tape-fast");
  const b = document.getElementById("tape-slow");
  if (a) a.innerHTML = `<b>FAST · BINANCE</b>${fast}<b>FAST · BINANCE</b>${fast}`;
  if (b) b.innerHTML = `<b>SLOW · POLYMARKET + τ</b>${slow}<b>SLOW · POLYMARKET + τ</b>${slow}`;
}

function theme() {
  const root = document.documentElement;
  const stored = localStorage.getItem("cmf-theme");
  if (stored) root.dataset.theme = stored;
  document.getElementById("theme")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : root.dataset.theme === "light" ? "dark" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("cmf-theme", next);
  });
}

function tocSpy() {
  const links = [...document.querySelectorAll("nav.toc a[href^='#']")];
  const ids = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const on = () => {
    let cur = ids[0];
    for (const el of ids) {
      if (el.getBoundingClientRect().top < 120) cur = el;
    }
    links.forEach((a) => {
      a.removeAttribute("aria-current");
      if (a.getAttribute("href") === `#${cur.id}`) a.setAttribute("aria-current", "true");
    });
  };
  document.addEventListener("scroll", on, { passive: true });
  on();
}

document.addEventListener("DOMContentLoaded", () => {
  renderTape();
  theme();
  tocSpy();
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
    });
  }
});
