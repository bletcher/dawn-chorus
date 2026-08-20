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

let failures = 0;
for (const page of process.argv.slice(2)) {
  const src = fs.readFileSync(page, "utf8");
  console.log(page);
  for (const [ok, what] of [...ladder(src), ...audioStates(src)]) {
    console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`);
    if (!ok) failures++;
  }
}
process.exit(failures ? 1 : 0);
