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
// A const/let declaration may span lines and contain braces; scan to the `;` at depth 0.
function extractConst(src, name) {
  const m = new RegExp("(?:^|\\n)\\s*(?:const|let)\\s+" + name + "\\s*=").exec(src);
  if (!m) throw new Error("no const " + name);
  let d = 0;
  for (let j = m.index; j < src.length; j++) {
    const c = src[j];
    if ("({[".includes(c)) d++;
    else if (")}]".includes(c)) d--;
    else if (c === ";" && d === 0) return src.slice(m.index, j + 1);
  }
  throw new Error("unterminated " + name);
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

  // --- the Overview bar must equal the chapter beneath it -------------------------------
  // Two charts computing "how many species, how many calls" from the same data by different
  // routes. The Overview once summed detections straight off `summary`, which the floor does
  // not touch, so it reported 71 species while the chapter below drew 42 -- and no amount of
  // looking at either chart alone would show it. The page's lead text now promises they
  // agree, so check the promise.
  const trend = new Function("meta", "day_keys", "summary", "aggSel", "periods",
    extract(src, "periodsFor") + "\n" + extract(src, "trendRows") +
    "\nperiods = periodsFor(aggSel.value);\nreturn {rows: trendRows(), periods};");
  let mism = 0, checked = 0, moved = false;
  const yearly = f => trend(meta, DATA.day_keys, summaryAt(f), { value: "year" }, []).rows[0];
  const lo = yearly(1), hi = yearly(20);
  moved = lo && hi && (lo.species !== hi.species || lo.calls !== hi.calls);
  for (const f of [1, 2, 3, 5, 8, 12, 20]) {
    const summary = summaryAt(f);
    for (const level of ["day", "week", "month", "year"]) {
      const { rows, periods } = trend(meta, DATA.day_keys, summary, { value: level }, []);
      for (const key of periods) {
        const days = new Set(meta.days.filter(d => DATA.day_keys[d][level] === key));
        const g = {};
        summary.forEach(r => { if (days.has(r.date)) (g[r.name] || (g[r.name] = [])).push(r); });
        let sp = 0, calls = 0;
        for (const name of Object.keys(g)) {           // renderPeriod's default admission test
          const rs = g[name], n = rs.reduce((s, r) => s + r.n, 0);
          if (n > 0 && rs.some(r => r.onset != null)) { sp++; calls += n; }
        }
        const bar = rows.find(r => r.key === key);
        checked++;
        if (!bar || bar.species !== sp || bar.calls !== calls) {
          if (mism++ === 0) console.log(`     FAIL floor=${f} ${level} ${key}: bar ` +
            `${bar ? bar.species + "sp/" + bar.calls : "missing"} vs chapter ${sp}sp/${calls}`);
        }
      }
    }
  }
  if (!moved) { console.log(`     FAIL the trend does not respond to the detection floor`); failures++; }
  if (mism) { console.log(`     FAIL ${mism} of ${checked} Overview/chapter combinations disagree`); failures++; }
  else if (moved) console.log(`     ok   Overview matches the chapter across ${checked} floor x grain x period ` +
    `combinations, and follows the floor (${lo.species} species at 1 -> ${hi.species} at 20)`);

  // --- the species table must be a SUPERSET of what the charts draw ----------------------
  // Its weak-evidence filter counts total detections; the floor counts detections in one
  // morning. Below a floor of 3 those disagree, and the flat rule silently dropped species
  // every other chart on the page was still drawing. Run the real renderTable and check.
  const tableAt = floor => {
    const out = { tbl: "", note: "" };
    const document = { getElementById(id) {
      if (id === "tbl") return { set innerHTML(v) { out.tbl = v; }, querySelectorAll: () => [] };
      if (id === "tblNote") return { set textContent(v) { out.note = v; }, set innerHTML(v) { out.note = v; } };
      return null; } };
    const ctx = { meta, summary: summaryAt(floor), document, MIN_DET: floor,
      hasAudio: () => false, scopedDays: () => meta.days, tableShowAll: false,
      esc: s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])) };
    const body = extractConst(src, "median") + "\n" + extractConst(src, "fmt") + "\n" +
      extractConst(src, "TABLE_MIN_DET") + "\n" + extractConst(src, "COLDESC") + "\n" +
      extractConst(src, "colDesc") + "\n" + extract(src, "aggBySpecies") + "\n" +
      extract(src, "renderTable") + "\nrenderTable(meta.days); return out;";
    new Function(...Object.keys(ctx), "out", body)(...Object.values(ctx), out);
    return out;
  };
  let tbad = 0;
  for (const f of [1, 2, 3, 5, 10, 20]) {
    const out = tableAt(f);
    const listed = new Set([...out.tbl.matchAll(/<tr><td>(.*?)<\/td>/g)].map(m => m[1]));
    const charted = summaryAt(f).filter(r => r.onset != null).map(r => r.name);
    const lost = [...new Set(charted)].filter(n => !listed.has(n));
    if (lost.length) {
      tbad++;
      console.log(`     FAIL floor=${f}: ${lost.length} species have an onset but are not in the ` +
        `table (${lost.slice(0, 3).join(", ")}${lost.length > 3 ? ", …" : ""})`);
    }
    // help text quoting a stale floor is the same class of bug, one layer up
    const help = /<th title="([^"]*)">onset<\/th>/.exec(out.tbl);
    const q = help && /\((\d+) detections\)/.exec(help[1]);
    if (!q || +q[1] !== f) {
      tbad++;
      console.log(`     FAIL floor=${f}: the onset column's help says "${q ? q[1] : "?"}"`);
    }
  }
  if (tbad) failures++;
  else console.log(`     ok   species table lists everything the charts draw, and its column help ` +
    `tracks the floor, at floors 1-20`);
}
process.exit(failures ? 1 : 0);
