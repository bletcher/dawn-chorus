// UI-state invariants for the emitted dashboard.
//
// check_js.py proves the page's JavaScript parses; check_phen.js proves its numbers match
// Python. Neither says anything about what the page SHOWS. This runs the page's real
// state functions against a DOM stub and checks the states that differ between builds --
// above all the audio one, which is the difference between the local dashboard and the
// published site and therefore never visible in a single page's markup.
//
//   node tools/check_ui.js site/dashboard-sm.html [...]
const fs = require("fs");

function grabFn(src, name) {
  const i = src.indexOf("function " + name + "(");
  if (i < 0) throw new Error("missing function " + name);
  let d = 0, on = false;
  for (let j = i; j < src.length; j++) {
    const c = src[j];
    if (c === "{") { d++; on = true; }
    else if (c === "}") { d--; if (on && d === 0) return src.slice(i, j + 1); }
  }
  throw new Error("unbalanced " + name);
}

function node() { return { hidden: false, innerHTML: "", set onclick(v) {} }; }

// Recordings never leave the machine, so the published payload carries no `audio`. Rung 3
// must then not exist at all -- a permanently locked rung reads as a broken feature rather
// than as a feature that lives somewhere else.
function audioStates(src) {
  const body = grabFn(src, "updateAudioCard");
  const run = (audio, S) => {
    const n = { rung: node(), card: node(), gate: node(), lead: node() };
    const ctx = {
      document: {
        getElementById: id => ({ audioCard: n.card, rungGate: n.gate, audioLead: n.lead }[id] || null),
        querySelector: sel => (sel.includes('data-rung="morning"') ? n.rung : null),
      },
      rungEl: k => (k === "morning" ? n.rung : null),
      hasAudio: () => !!audio && Object.keys(audio).length > 0,
      AUDIO: audio,
      esc: s => String(s),
      narrowToMorning: () => {},
    };
    new Function(...Object.keys(ctx), body + `\nupdateAudioCard(${JSON.stringify(S)});`)(...Object.values(ctx));
    return n;
  };
  const A = { "2026-08-11": [{}], "2026-08-12": [{}, {}] };
  const out = [];
  const pub = run(null, ["2026-08-11", "2026-08-12"]);
  out.push([pub.rung.hidden === true,
    "published build (no audio): rung 3 is absent, not a dead locked rung"]);
  const wide = run(A, ["2026-08-11", "2026-08-12"]);
  out.push([wide.rung.hidden === false && wide.card.hidden === true && wide.gate.hidden === false,
    "local build, 2 mornings: rung 3 shows, player gated, gate explains and offers to narrow"]);
  out.push([/narrow|one morning/i.test(wide.gate.innerHTML),
    "the gate says WHY rather than just disabling itself"]);
  const one = run(A, ["2026-08-12"]);
  out.push([one.rung.hidden === false && one.card.hidden === false && one.gate.hidden === true,
    "local build, 1 morning: player live, gate gone"]);
  return out;
}

// Every rung must state its own scope; that sentence is what stops one control appearing
// to mean three different things.
function ladder(src) {
  const rungs = [...src.matchAll(/<section class="rung" data-rung="(\w+)"/g)].map(m => m[1]);
  const scopes = [...src.matchAll(/id="scope-(\w+)"/g)].map(m => m[1]);
  const cards = [...src.matchAll(/data-card="([\w-]+)"/g)].map(m => m[1]);
  const seasonIdx = src.indexOf('data-rung="season"');
  const selIdx = src.indexOf('data-rung="selection"');
  const seasonBody = src.slice(seasonIdx, selIdx);
  return [
    [rungs.join(",") === "season,selection,morning", `rungs are season → selection → morning (got ${rungs})`],
    [rungs.every(r => scopes.includes(r)), "every rung has a scope sentence element"],
    [cards.length === 9, `all nine cards survive the restructure (got ${cards.length})`],
    [/data-card="season"/.test(seasonBody),
      "the seasonal chart sits on the rung that ignores the scope"],
    [/data-card="trend"/.test(seasonBody), "the record-wide trend sits on the season rung"],
    [!src.includes('class="chapter"'), "the old chapter layout is gone"],
    [src.includes('id="crumb"'), "the breadcrumb is present"],
  ];
}

// The context strip must show the RECORD, not the selection. If it re-bucketed itself with
// the aggregation it would be showing the selection twice and the record never -- and the
// brush that replaces the period slider needs these bars to be one morning each.
function strip(src, DATA) {
  if (!DATA || !DATA.phen) return [[true, "strip checks skipped (no payload)"]];
  const meta = DATA.meta;
  const pctx = { meta, GRID: meta.grid, NB: meta.grid.length, PHEN: DATA.phen,
    PHSP: meta.phen_species, QO: meta.onset_quantile, BUILT_SUMMARY: DATA.summary, _sumCache: {} };
  const summaryAt = new Function(...Object.keys(pctx),
    grabFn(src, "quantileSorted") + "\n" + grabFn(src, "summaryAt") + "\nreturn summaryAt;")(...Object.values(pctx));

  const draw = (level, S, metric, floor) => {
    const host = { clientWidth: 600, clientHeight: 46, innerHTML: "" };
    const hint = { textContent: "" };
    const ctx = { meta, day_keys: DATA.day_keys, summary: summaryAt(floor),
      aggSel: { value: level }, periods: [], TREND_METRIC: metric, STRIP_ROWS: null,
      css: () => "#000", esc: s => String(s),
      document: { getElementById: id => (id === "strip" ? host : id === "stripHint" ? hint : null) } };
    const body = grabFn(src, "periodsFor") + "\n" + grabFn(src, "trendRows") + "\n" +
      grabFn(src, "stripRows") + "\n" + grabFn(src, "renderStrip") +
      `\nperiods = periodsFor(aggSel.value);\nrenderStrip(${JSON.stringify(S)});` +
      `\nreturn {html: host.innerHTML, hint: hint.textContent};`;
    return new Function(...Object.keys(ctx), "host", "hint", body)(...Object.values(ctx), host, hint);
  };

  const days = meta.days, out = [];
  const counts = ["day", "week", "month", "year"].map(l =>
    (draw(l, days, "calls", 5).html.match(/class="bar/g) || []).length);
  out.push([counts.every(c => c === days.length),
    `strip draws one bar per morning at every aggregation (got ${counts.join("/")}, expected ${days.length})`]);

  const step = 592 / days.length;
  const S = days.slice(Math.min(2, days.length - 1), Math.min(5, days.length));
  const html = draw("day", S, "calls", 5).html;
  const m = /<rect class="sel" x="([\d.]+)" y="0" width="([\d.]+)"/.exec(html);
  const i0 = days.indexOf(S[0]);
  out.push([!!m && Math.abs(+m[1] - i0 * step) < 1 && Math.abs(+m[2] - S.length * step) < 1,
    "the selection rectangle spans exactly the scoped mornings"]);
  out.push([(html.match(/class="bar in"/g) || []).length === S.length,
    "exactly the scoped mornings are lit"]);

  const geo = f => (draw("day", days, "species", f).html.match(/height="([\d.]+)"/g) || []).join(",");
  out.push([geo(1) !== geo(20), "the strip follows the detection floor"]);

  // A total here would be day-grain and would not match Rung 1's at a coarser grain.
  const hint = draw("day", [days[0]], "calls", 5).hint;
  out.push([!/\d{3,}/.test(hint.replace(/\d{4}-\d{2}-\d{2}/g, "")),
    "the strip's hint carries no grain-dependent total"]);
  return out;
}

const argv = process.argv.slice(2);
const pi = argv.indexOf("--payload");
const external = pi >= 0 ? argv[pi + 1] : null;
const pages = pi >= 0 ? argv.filter((a, i) => i !== pi && i !== pi + 1) : argv;

let failures = 0;
for (const page of pages) {
  const src = fs.readFileSync(page, "utf8");
  const m = src.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/);
  let DATA = null;
  try { DATA = JSON.parse(external ? fs.readFileSync(external, "utf8") : m[1]); } catch (e) {}
  console.log(page);
  for (const [ok, what] of [...ladder(src), ...audioStates(src), ...strip(src, DATA)]) {
    console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
    if (!ok) failures++;
  }
}
process.exit(failures ? 1 : 0);
