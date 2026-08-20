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
  // Order-independent: take from the season rung to whatever rung follows it, so a
  // reordering of the ladder never silently turns these two checks into empty slices.
  const seasonIdx = src.indexOf('data-rung="season"');
  const nextIdx = src.indexOf('data-rung=', seasonIdx + 20);
  const seasonBody = src.slice(seasonIdx, nextIdx > 0 ? nextIdx : src.indexOf("<footer", seasonIdx));
  return [
    // Selection leads because the context strip is pinned above everything and is itself the
    // overview -- so the ladder below can open on what you are actually looking at.
    [rungs.join(",") === "selection,season,morning", `rungs are selection → season → morning (got ${rungs})`],
    [/data-rung="selection"[\s\S]*?data-card="timeline"[\s\S]*?data-card="period"/.test(src),
      "Who sings when leads the selection rung"],
    [rungs.every(r => scopes.includes(r)), "every rung has a scope sentence element"],
    [cards.length === 9, `all nine cards survive the restructure (got ${cards.length})`],
    [/data-card="season"/.test(seasonBody),
      "the seasonal chart sits on the rung that ignores the scope"],
    [/data-card="trend"/.test(seasonBody), "the record-wide trend sits on the season rung"],
    [!src.includes('class="chapter"'), "the old chapter layout is gone"],
    [src.includes('id="crumb"'), "the breadcrumb is present"],
    // The roll-up chart used to follow the Snap control, which meant that at Snap = Morning
    // it drew the context strip's series a second time one screen below it. Its grain is now
    // its own and starts at Week, so the two can never show the same thing.
    [!/<select id="trendGrain">[\s\S]*?value="day"[\s\S]*?<\/select>/.test(src),
      "the roll-up cannot be set to Morning, so it can never duplicate the strip"],
    [/let TREND_GRAIN/.test(src) && /renderTrend[\s\S]{0,400}?TREND_GRAIN/.test(src),
      "the roll-up reads its own grain, not the selection's"],
    // Method text sits behind a button; the one line that stays on screen is the computed
    // finding, because that one changes with the data. A card that puts its prose back on
    // the page would undo the point of the button.
    // Count card OPENINGS, not every `data-card=` in the file -- the JS selectors match too.
    [(src.match(/class="infobtn"/g) || []).length ===
     (src.match(/<section class="card" data-card="/g) || []).length,
      "every chart card has an info button"],
    [!/<p class="lead">/.test(src.slice(src.indexOf('class="ladder"'), src.indexOf("<footer"))),
      "no card puts its method text back on the page"],
    [!src.includes('class="cardq"'), "the chapter questions, which restated the titles, are gone"],
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
    const host = { clientWidth: 600, clientHeight: 46, innerHTML: "",
                   setAttribute() {}, classList: { add() {}, remove() {} } };
    const hint = { textContent: "" };
    const days = meta.days;
    const ctx = { meta, day_keys: DATA.day_keys, summary: summaryAt(floor),
      aggSel: { value: level }, periods: [], TREND_METRIC: metric, STRIP_ROWS: null,
      SEL: { a: days.indexOf(S[0]), b: days.indexOf(S[S.length - 1]) },
      SNAP: level === "day" ? "day" : level,
      grain: () => level,
      css: () => "#000", esc: s => String(s),
      document: { getElementById: id => (id === "strip" ? host : id === "stripHint" ? hint : null) } };
    const body = grabFn(src, "periodsFor") + "\n" + grabFn(src, "trendRows") + "\n" +
      grabFn(src, "stripRows") + "\n" + grabFn(src, "selLabel") + "\n" + grabFn(src, "renderStrip") +
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

// The equivalence that let the period slider be retired rather than kept alongside.
//
// The old control could only name one whole bucket: pick a grain, pick an index, get
// `meta.days.filter(d => day_keys[d][grain] === key)`. The brush must reproduce that day set
// EXACTLY for every grain and every period -- otherwise "Snap = Week behaves as it always
// did" is a claim rather than a fact, and a scoping regression would show up as slightly
// wrong numbers everywhere at once rather than as an error.
function equivalence(src, DATA) {
  if (!DATA) return [[true, "equivalence skipped (no payload)"]];
  const meta = DATA.meta, day_keys = DATA.day_keys, days = meta.days;
  const ctx = { meta, day_keys, SNAP: "day", SEL: { a: 0, b: 0 }, periods: [] };
  const body = grabFn(src, "periodsFor") + "\n" + grabFn(src, "periodBounds") + "\n" +
    grabFn(src, "snapRange") + "\n" + grabFn(src, "setSel") + "\n" +
    grabFn(src, "scopedDays") + "\n" + grabFn(src, "daysOfPeriod") + "\n" +
    `
    const nDays = () => meta.days.length;
    const grain = () => SNAP === "free" ? "day" : SNAP;
    const results = [];
    for (const level of ["day","week","month","year"]) {
      SNAP = level;
      periods = periodsFor(level);
      for (const key of periods) {
        const want = meta.days.filter(d => day_keys[d][level] === key);
        // land the brush anywhere inside the period, as a drag would
        const mid = meta.days.indexOf(want[Math.floor(want.length/2)]);
        SEL = {a:0, b:0};
        setSel(mid, mid);
        const got = scopedDays();
        results.push([level, key, want.join("|") === got.join("|"), want.length, got.length]);
      }
    }
    return results;
    `;
  const rows = new Function(...Object.keys(ctx), body)(...Object.values(ctx));
  const bad = rows.filter(r => !r[2]);
  const out = [[bad.length === 0,
    `brush reproduces the old period slider exactly: ${rows.length} (grain × period) ` +
    `combinations, ${bad.length} mismatches`]];
  bad.slice(0, 3).forEach(r => out.push([false, `   ${r[0]} ${r[1]}: wanted ${r[3]} mornings, got ${r[4]}`]));
  return out;
}

// And the capabilities the old control could NOT express, plus the states a brush gets
// wrong when it is built carelessly: empty selections, wrapping, dragging off the end.
function brush(src, DATA) {
  if (!DATA) return [[true, "brush checks skipped (no payload)"]];
  const meta = DATA.meta, day_keys = DATA.day_keys, days = meta.days;
  // A station on its first morning is a real, valid page -- ranges and stepping simply have
  // nowhere to go. Skipping is correct here; failing would make a fresh install look broken.
  if (days.length < 8) return [[true, `brush range checks skipped (${days.length} mornings on record)`]];
  const api = snap => {
    const ctx = { meta, day_keys, SNAP: snap, SEL: { a: 0, b: 0 }, periods: [] };
    const body = grabFn(src, "periodsFor") + "\n" + grabFn(src, "periodBounds") + "\n" +
      grabFn(src, "snapRange") + "\n" + grabFn(src, "setSel") + "\n" +
      grabFn(src, "scopedDays") + "\n" + grabFn(src, "daysOfPeriod") + "\n" +
      grabFn(src, "stepSelection") + "\n" +
      `const nDays = () => meta.days.length;
       const grain = () => SNAP === "free" ? "day" : SNAP;
       periods = periodsFor(grain());
       return { sel: () => ({...SEL}), set: (a,b,o) => setSel(a,b,o),
                days: () => scopedDays(), step: d => stepSelection(d) };`;
    return new Function(...Object.keys(ctx), body)(...Object.values(ctx));
  };
  const out = [];
  let b = api("free");
  b.set(2, 4, { snap: false });
  out.push([b.days().join() === days.slice(2, 5).join(),
    "Free snap selects an arbitrary range — what the bucket picker could not express"]);
  b.step(1);
  out.push([b.sel().a === 5 && b.sel().b === 7, "stepping a free range moves it by its own width"]);
  b.set(-9, 999, { snap: false });
  out.push([b.sel().a === 0 && b.sel().b === days.length - 1,
    "dragging past either end clamps to the record rather than scrolling into empty time"]);
  b.set(6, 2, { snap: false });
  out.push([b.sel().a === 2 && b.sel().b === 6,
    "a right-to-left drag is normalised, never left empty"]);
  b.set(4, 4, { snap: false });
  out.push([b.days().length === 1, "one morning is the minimum: an empty selection is unrepresentable"]);
  b = api("week");
  b.set(0, 0);
  const w1 = b.days();
  b.step(1);
  const w2 = b.days();
  out.push([w1.join() !== w2.join() && w2.every(d => day_keys[d].week === day_keys[w2[0]].week),
    "with Week snap the steppers land on exactly one week, as the old ‹ › did"]);
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
  for (const [ok, what] of [...ladder(src), ...audioStates(src), ...strip(src, DATA),
                            ...equivalence(src, DATA), ...brush(src, DATA)]) {
    console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
    if (!ok) failures++;
  }
}
process.exit(failures ? 1 : 0);
