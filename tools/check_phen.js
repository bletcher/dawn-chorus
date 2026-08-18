// Does the browser's recompute reproduce Python's morning_summary?
//
// The detection floor is now applied in the page, not at build time, so the page has its
// own implementation of onset/offset/peak/occupancy. If it drifts from dawnchorus.phenology
// the dashboard quietly stops agreeing with the CSVs and the published payload -- a
// divergence no amount of eyeballing a chart would catch.
//
// This extracts the REAL functions from the emitted page (not a copy) and runs them against
// the page's own payload at the build-time floor, where the answer is already known.
//
//   node tools/check_phen.js site/dashboard-sm.html [...]
//
// The public viewer carries no embedded payload -- it fetches one at runtime -- so for that
// page the payload is supplied separately, and the same functions are checked against it:
//
//   node tools/check_phen.js site/index.html --payload site/data/montague.json
const fs = require("fs");

function extract(src, name) {
  const i = src.indexOf("function " + name + "(");
  if (i < 0) throw new Error("no function " + name);
  let d = 0, started = false;
  for (let j = i; j < src.length; j++) {
    const c = src[j];
    if (c === "{") { d++; started = true; }
    else if (c === "}") { d--; if (started && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error("unbalanced " + name);
}

const argv = process.argv.slice(2);
const pi = argv.indexOf("--payload");
const external = pi >= 0 ? argv[pi + 1] : null;
// NB: guard on pi >= 0 -- with no --payload, `pi + 1` is 0 and would silently drop the
// first page, reporting success because nothing was checked.
const pages = pi >= 0 ? argv.filter((a, i) => i !== pi && i !== pi + 1) : argv;
if (!pages.length) { console.error("no pages given"); process.exit(2); }

let failures = 0;
for (const page of pages) {
  const src = fs.readFileSync(page, "utf8");
  const m = src.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/);
  const raw = external ? fs.readFileSync(external, "utf8") : (m && m[1]);
  let DATA;
  try { DATA = JSON.parse(raw); }
  catch (e) {
    console.error(`${page}: no usable payload` +
      (m && !external ? " (viewer page — pass --payload <json>)" : ""));
    failures++; continue;
  }
  const meta = DATA.meta;
  if (!DATA.phen) { console.log(`${page}: no phen array — skipped`); continue; }

  const ctx = {
    meta, GRID: meta.grid, NB: meta.grid.length,
    PHEN: DATA.phen, PHSP: meta.phen_species || [],
    QO: meta.onset_quantile != null ? meta.onset_quantile : 0.05,
    BUILT_SUMMARY: DATA.summary, _sumCache: {},
  };
  const body = extract(src, "quantileSorted") + "\n" + extract(src, "summaryAt") + "\nreturn summaryAt;";
  const summaryAt = new Function(...Object.keys(ctx), body)(...Object.values(ctx));

  const floor = meta.min_detections || 5;
  const got = summaryAt(floor);
  const want = DATA.summary;
  const key = r => r.date + "|" + r.name;
  const G = new Map(got.map(r => [key(r), r])), W = new Map(want.map(r => [key(r), r]));

  const only = (a, b) => [...a.keys()].filter(k => !b.has(k));
  const bad = { rows: 0, n: 0, onset: 0, offset: 0, peak: 0, occ: 0 };
  const missG = only(W, G), missW = only(G, W);
  bad.rows = missG.length + missW.length;

  const near = (a, b, tol) =>
    (a == null || b == null) ? (a == null && b == null) : Math.abs(a - b) <= tol;
  const examples = [];
  for (const [k, w] of W) {
    const g = G.get(k); if (!g) continue;
    // Python rounds summary floats to 1dp (occ 2dp) for the payload and phen times to 3dp,
    // so the only legitimate gap is half a display step plus ~0.0005 of interpolation
    // error. Anything larger is real drift between the two implementations.
    const checks = [["n", w.n, g.n, 0], ["onset", w.onset, g.onset, 0.0506],
                    ["offset", w.offset, g.offset, 0.0506], ["peak", w.peak, g.peak, 0],
                    ["occ", w.occ, g.occ, 0.0051]];
    for (const [f, a, b, tol] of checks)
      if (!near(a, b, tol)) { bad[f]++; if (examples.length < 4) examples.push(`${k} ${f}: py=${a} js=${b}`); }
  }
  const total = Object.values(bad).reduce((s, v) => s + v, 0);
  const tag = total ? "FAIL" : "ok  ";
  console.log(`${tag} ${page}  floor ${floor}: ${want.length} rows, ` +
    (total ? JSON.stringify(bad) : "every field matches dawnchorus.phenology"));
  if (missG.length) console.log(`      python-only rows: ${missG.slice(0, 3).join(", ")}`);
  if (missW.length) console.log(`      browser-only rows: ${missW.slice(0, 3).join(", ")}`);
  examples.forEach(e => console.log("      " + e));
  if (total) failures++;
}
process.exit(failures ? 1 : 0);
