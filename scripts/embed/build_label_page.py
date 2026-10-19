"""Build a self-contained local labeling page (label.html) for bounding-box + move labels.

Inputs:
  * <frames_dir>/frames.json  : [{"frame", "time_s", "file", "reason"}, ...]  (file relative to page)
  * move lists: data/dustloop/manifest.json if present, otherwise the Dustloop cargo dumps
    (<name>_full.json from api.php?action=cargoquery&tables=MoveData_GGST) passed via --cargo.

The page has no external dependencies, loads images by relative path (open via file://),
autosaves to localStorage and exports labels.json.

Usage:
  .venv/Scripts/python.exe scripts/embed/build_label_page.py output/embed_test \
      --frames label_frames/frames.json --cargo output/embed_test/Anji_Mito_full.json \
      output/embed_test/Faust_full.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHARACTERS = ["Anji Mito", "Faust"]
GENERIC = ["idle/walk/dash", "blocking", "hitstun", "knockdown/wakeup", "Psych Burst",
           "other"]
TYPE_ORDER = {"normal": 0, "command_normal": 1, "special": 2, "super": 3, "throw": 4, "system": 5, "other": 6}


def moves_from_manifest(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    if isinstance(data, dict) and isinstance(data.get("moves"), list):
        # data/dustloop/manifest.json format: {"moves": [{"character","name","input","category"}]}
        seen = set()
        for m in data["moves"]:
            ch, inp = m["character"], m.get("input") or m.get("name")
            if (ch, inp) in seen:
                continue
            seen.add((ch, inp))
            name = m.get("name") or ""
            out.setdefault(ch, []).append({"input": inp, "name": "" if name == inp else name,
                                           "type": m.get("category") or m.get("cargo_type") or ""})
        return out
    chars = data.get("characters", data) if isinstance(data, dict) else {}
    for ch, entry in chars.items():
        name = ch.replace("_", " ")
        moves = entry.get("moves", entry) if isinstance(entry, dict) else entry
        lst = []
        for m in moves:
            inp = m.get("input") or m.get("id") or m.get("name")
            lst.append({"input": inp, "name": m.get("name") or "", "type": m.get("type") or ""})
        out[name] = lst
    return out


def moves_from_cargo(paths: list[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        rows = json.loads(p.read_text(encoding="utf-8"))["cargoquery"]
        for r in rows:
            t = r["title"]
            out.setdefault(t["chara"], []).append(
                {"input": t["input"], "name": t.get("name") or "", "type": t.get("type") or ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="directory that will contain label.html")
    ap.add_argument("--frames", default="label_frames/frames.json",
                    help="frames.json path relative to out_dir")
    ap.add_argument("--cargo", nargs="*", default=[])
    ap.add_argument("--manifest", default=str(ROOT / "data" / "dustloop" / "manifest.json"))
    ap.add_argument("--title", default="Set 17 move labels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    frames_rel = Path(args.frames)
    frames = json.loads((out_dir / frames_rel).read_text(encoding="utf-8"))
    img_prefix = frames_rel.parent.as_posix()
    for f in frames:
        f["src"] = f"{img_prefix}/{f['file']}" if img_prefix not in ("", ".") else f["file"]

    manifest = Path(args.manifest)
    if manifest.exists():
        moves = moves_from_manifest(manifest)
        source = f"manifest {manifest}"
    else:
        moves = moves_from_cargo([Path(p) for p in args.cargo])
        source = "Dustloop cargo API (MoveData_GGST)"
    for ch in moves:
        moves[ch].sort(key=lambda m: (TYPE_ORDER.get(m["type"], 9), m["input"]))
    for ch in CHARACTERS:
        if ch not in moves:
            raise SystemExit(f"no moves for {ch}")

    payload = {"frames": frames, "moves": moves, "generic": GENERIC,
               "characters": CHARACTERS, "source": source, "title": args.title}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__TITLE__", args.title)
    (out_dir / "label.html").write_text(html, encoding="utf-8")
    print(f"wrote {out_dir / 'label.html'}: {len(frames)} frames, "
          + ", ".join(f"{k}: {len(v)} moves" for k, v in moves.items()) + f" ({source})")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#16181d;--panel:#1f232b;--fg:#e6e6e6;--mut:#9aa3b2;--acc:#4ea1ff;--anji:#ff6b6b;--faust:#5be37d;--warn:#ffcc4d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,Segoe UI,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:8px 12px;background:var(--panel);flex-wrap:wrap}
header b{font-size:15px}button{background:#2c323d;color:var(--fg);border:1px solid #3b4250;border-radius:4px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}button.primary{background:var(--acc);color:#000;border-color:var(--acc)}
main{display:flex;gap:12px;padding:12px;align-items:flex-start}
#stage{position:relative;flex:0 1 auto;min-width:0;user-select:none;line-height:0}
#stage img{display:block;max-width:100%;max-height:calc(100vh - 90px);border-radius:4px}
#overlay{position:absolute;inset:0;cursor:crosshair}
.box{position:absolute;border:2px solid var(--acc);background:rgba(78,161,255,.08)}
.box.sel{border-width:3px;box-shadow:0 0 0 1px #000}
.box .tag{position:absolute;left:-2px;top:-20px;font-size:12px;padding:1px 4px;background:var(--acc);color:#000;white-space:nowrap}
aside{width:360px;flex:0 0 360px;background:var(--panel);border-radius:4px;padding:10px}
aside h3{margin:4px 0 8px;font-size:14px}.mut{color:var(--mut)}
.bitem{border:1px solid #3b4250;border-radius:4px;padding:6px;margin-bottom:8px}.bitem.sel{border-color:var(--acc)}
.bitem select,.bitem input{width:100%;margin-top:4px;background:#12141a;color:var(--fg);border:1px solid #3b4250;border-radius:3px;padding:4px}
.row{display:flex;gap:6px;align-items:center}.row>*{flex:1}
#strip{display:flex;gap:3px;flex-wrap:wrap;padding:0 12px 12px}
#strip span{width:22px;height:14px;border-radius:2px;background:#333;cursor:pointer;font-size:9px;text-align:center;line-height:14px;color:#000}
#strip span.done{background:var(--faust)}#strip span.skip{background:var(--warn)}#strip span.cur{outline:2px solid var(--acc)}
kbd{background:#2c323d;border:1px solid #444;border-radius:3px;padding:0 4px;font-size:12px}
.keys{font-size:12px;color:var(--mut);margin-top:10px;line-height:1.8}
.skipbanner{position:absolute;top:8px;left:8px;background:var(--warn);color:#000;padding:4px 8px;font-weight:bold;border-radius:3px}
@media (max-width:900px){main{flex-direction:column}aside{width:100%;flex:auto}}
</style></head><body>
<header><b>__TITLE__</b><span id="pos" class="mut"></span><span id="meta" class="mut"></span>
<span style="flex:1"></span>
<button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
<button id="skipbtn">Skip frame (S)</button>
<button id="exportbtn" class="primary">Export labels.json</button>
<label class="mut"><input type="file" id="importf" accept=".json" style="display:none"><span style="cursor:pointer;text-decoration:underline">Import</span></label>
</header>
<main>
<div id="stage"><img id="img" alt="frame"><div id="overlay"></div></div>
<aside>
<h3>Boxes <span class="mut" id="nb"></span></h3>
<div id="blist"></div>
<div class="mut" id="hint">Drag on the image to draw a box around a character.</div>
<div class="keys">
<kbd>drag</kbd> new box &nbsp; <kbd>click</kbd> select box<br>
<kbd>1</kbd> Anji &nbsp; <kbd>2</kbd> Faust (selected box)<br>
<kbd>I</kbd> idle/walk/dash &nbsp; <kbd>B</kbd> blocking &nbsp; <kbd>H</kbd> hitstun &nbsp; <kbd>K</kbd> knockdown &nbsp; <kbd>O</kbd> other<br>
<kbd>/</kbd> or <kbd>M</kbd> type to search moves (Enter picks first match)<br>
<kbd>Tab</kbd> cycle selected box &nbsp; <kbd>Del</kbd> delete box<br>
<kbd>&rarr;</kbd>/<kbd>N</kbd>/<kbd>Space</kbd> next &nbsp; <kbd>&larr;</kbd>/<kbd>P</kbd> prev &nbsp; <kbd>S</kbd> skip frame<br>
<kbd>Ctrl+E</kbd> export &nbsp; <kbd>U</kbd> next unlabeled
</div>
<div class="mut" style="margin-top:8px;font-size:11px" id="src"></div>
</aside>
</main>
<div id="strip"></div>
<script>
const D = /*__DATA__*/null;
const KEY = 'replanal_labels_v1::' + D.title;
let state = {};   // frame -> {boxes:[...], skipped:bool}
try { state = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { state = {}; }
let idx = 0, sel = -1;
try { idx = Math.min(D.frames.length - 1, +localStorage.getItem(KEY + '::idx') || 0); } catch (e) {}
const $ = id => document.getElementById(id);
const img = $('img'), ov = $('overlay');
const COLORS = {'Anji Mito': 'var(--anji)', 'Faust': 'var(--faust)'};
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); localStorage.setItem(KEY + '::idx', idx); } catch (e) {} }
function cur() { const f = D.frames[idx]; return state[f.frame] || (state[f.frame] = {boxes: [], skipped: false}); }
function moveOptions(ch) {
  const o = ['<option value="">— move —</option>', '<optgroup label="state">'];
  D.generic.forEach(g => o.push(`<option>${g}</option>`)); o.push('</optgroup>');
  const chars = ch ? [ch] : D.characters;
  chars.forEach(c => {
    let grp = null;
    (D.moves[c] || []).forEach(m => {
      if (m.type !== grp) { if (grp !== null) o.push('</optgroup>'); grp = m.type; o.push(`<optgroup label="${c} ${grp}">`); }
      const v = m.input; o.push(`<option value="${esc(v)}">${esc(v)}${m.name ? ' — ' + esc(m.name) : ''}</option>`);
    });
    if (grp !== null) o.push('</optgroup>');
  });
  return o.join('');
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function scale() { return img.clientWidth / (img.naturalWidth || 1920); }
function render() {
  const f = D.frames[idx];
  if (img.getAttribute('src') !== f.src) img.src = f.src;
  $('pos').textContent = `${idx + 1} / ${D.frames.length}`;
  $('meta').textContent = `frame ${f.frame} · t=${(+f.time_s).toFixed(2)}s · ${f.reason || ''}`;
  const c = cur(), s = scale();
  ov.innerHTML = c.skipped ? '<div class="skipbanner">SKIPPED</div>' : '';
  c.boxes.forEach((b, i) => {
    const d = document.createElement('div'); d.className = 'box' + (i === sel ? ' sel' : '');
    const col = COLORS[b.character] || 'var(--acc)';
    d.style.cssText = `left:${b.x * s}px;top:${b.y * s}px;width:${b.w * s}px;height:${b.h * s}px;border-color:${col}`;
    d.innerHTML = `<span class="tag" style="background:${col}">${i + 1}: ${esc(b.character || '?')} · ${esc(b.move || '?')}</span>`;
    d.addEventListener('mousedown', e => { e.stopPropagation(); sel = i; render(); });
    ov.appendChild(d);
  });
  const bl = $('blist'); bl.innerHTML = '';
  c.boxes.forEach((b, i) => {
    const el = document.createElement('div'); el.className = 'bitem' + (i === sel ? ' sel' : '');
    el.innerHTML = `<div class="row"><b style="flex:0 0 auto">#${i + 1}</b><span class="mut">${b.w}×${b.h} @ ${b.x},${b.y}</span><button style="flex:0 0 auto" data-del>✕</button></div>
      <select data-ch><option value="">— character —</option>${D.characters.map(ch => `<option ${ch === b.character ? 'selected' : ''}>${ch}</option>`).join('')}</select>
      <input data-q placeholder="search moves (e.g. 5H, fuujin)">
      <select data-mv>${moveOptions(b.character)}</select>`;
    const mv = el.querySelector('[data-mv]'); mv.value = b.move || '';
    el.addEventListener('mousedown', () => { if (sel !== i) { sel = i; render(); } });
    el.querySelector('[data-del]').onclick = () => { c.boxes.splice(i, 1); sel = -1; save(); render(); };
    el.querySelector('[data-ch]').onchange = e => { b.character = e.target.value; save(); render(); };
    mv.onchange = e => { b.move = e.target.value; save(); render(); };
    const q = el.querySelector('[data-q]');
    q.oninput = () => filterMoves(mv, q.value, b.character);
    q.onkeydown = e => {
      e.stopPropagation();
      if (e.key === 'Enter') { const first = [...mv.options].find(o => o.value && !o.hidden); if (first) { b.move = first.value; save(); render(); } }
      if (e.key === 'Escape') { q.blur(); }
    };
    bl.appendChild(el);
  });
  $('nb').textContent = c.boxes.length ? `(${c.boxes.length})` : '';
  $('hint').style.display = c.boxes.length ? 'none' : '';
  const st = $('strip'); st.innerHTML = '';
  D.frames.forEach((fr, i) => {
    const s2 = state[fr.frame]; const sp = document.createElement('span');
    const done = s2 && s2.boxes.length && s2.boxes.every(b => b.character && b.move);
    sp.className = (s2 && s2.skipped ? 'skip' : done ? 'done' : '') + (i === idx ? ' cur' : '');
    sp.title = `#${i + 1} frame ${fr.frame}`; sp.textContent = i + 1;
    sp.onclick = () => go(i); st.appendChild(sp);
  });
}
function filterMoves(sel_, q, ch) {
  q = q.trim().toLowerCase();
  [...sel_.options].forEach(o => { o.hidden = q && !(o.textContent.toLowerCase().includes(q)); });
}
function go(i) { idx = Math.max(0, Math.min(D.frames.length - 1, i)); sel = cur().boxes.length ? 0 : -1; save(); render(); }
// drawing
let drag = null;
ov.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const r = ov.getBoundingClientRect(); drag = {x0: e.clientX - r.left, y0: e.clientY - r.top, el: document.createElement('div')};
  drag.el.className = 'box'; ov.appendChild(drag.el);
});
window.addEventListener('mousemove', e => {
  if (!drag) return; const r = ov.getBoundingClientRect();
  const x = Math.max(0, Math.min(r.width, e.clientX - r.left)), y = Math.max(0, Math.min(r.height, e.clientY - r.top));
  drag.x1 = x; drag.y1 = y;
  Object.assign(drag.el.style, {left: Math.min(x, drag.x0) + 'px', top: Math.min(y, drag.y0) + 'px', width: Math.abs(x - drag.x0) + 'px', height: Math.abs(y - drag.y0) + 'px'});
});
window.addEventListener('mouseup', () => {
  if (!drag) return; const d = drag; drag = null;
  if (d.x1 === undefined || Math.abs(d.x1 - d.x0) < 6 || Math.abs(d.y1 - d.y0) < 6) { render(); return; }
  const s = scale(), c = cur();
  const b = {x: Math.round(Math.min(d.x0, d.x1) / s), y: Math.round(Math.min(d.y0, d.y1) / s), w: Math.round(Math.abs(d.x1 - d.x0) / s), h: Math.round(Math.abs(d.y1 - d.y0) / s), character: '', move: ''};
  // default character: the one not yet boxed on this frame
  const used = c.boxes.map(x => x.character); b.character = D.characters.find(ch => !used.includes(ch)) || '';
  c.boxes.push(b); c.skipped = false; sel = c.boxes.length - 1; save(); render();
});
window.addEventListener('resize', render);
img.addEventListener('load', render);
img.addEventListener('error', () => { $('meta').textContent = 'IMAGE FAILED TO LOAD: ' + D.frames[idx].src + ' (keep label.html next to the label_frames folder)'; });
function setSel(field, val) { const c = cur(); if (sel < 0 || !c.boxes[sel]) return; c.boxes[sel][field] = val; save(); render(); }
document.addEventListener('keydown', e => {
  const t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA') return;
  if (t === 'SELECT') { if (e.key === 'Escape') e.target.blur(); return; }
  const k = e.key, c = cur();
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'e') { e.preventDefault(); exportLabels(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (k) {
    case 'ArrowRight': case 'n': case 'N': case ' ': e.preventDefault(); go(idx + 1); break;
    case 'ArrowLeft': case 'p': case 'P': e.preventDefault(); go(idx - 1); break;
    case 's': case 'S': c.skipped = !c.skipped; save(); if (c.skipped) go(idx + 1); else render(); break;
    case '1': setSel('character', D.characters[0]); break;
    case '2': setSel('character', D.characters[1]); break;
    case 'i': case 'I': setSel('move', 'idle/walk/dash'); break;
    case 'b': case 'B': setSel('move', 'blocking'); break;
    case 'h': case 'H': setSel('move', 'hitstun'); break;
    case 'k': case 'K': setSel('move', 'knockdown/wakeup'); break;
    case 'o': case 'O': setSel('move', 'other'); break;
    case 'Tab': e.preventDefault(); if (c.boxes.length) { sel = (sel + 1) % c.boxes.length; render(); } break;
    case 'Delete': case 'Backspace': if (sel >= 0) { c.boxes.splice(sel, 1); sel = c.boxes.length ? 0 : -1; save(); render(); } break;
    case '/': case 'm': case 'M': { e.preventDefault(); const q = document.querySelectorAll('.bitem [data-q]')[sel]; if (q) q.focus(); break; }
    case 'u': case 'U': { for (let j = 1; j <= D.frames.length; j++) { const i2 = (idx + j) % D.frames.length; const s2 = state[D.frames[i2].frame]; if (!s2 || (!s2.skipped && !s2.boxes.length)) { go(i2); break; } } break; }
  }
});
function exportLabels() {
  const out = {source_frames: 'label_frames/frames.json', exported_at: new Date().toISOString(), frames: D.frames.map(f => {
    const s = state[f.frame] || {boxes: [], skipped: false};
    return {frame: f.frame, time_s: f.time_s, file: f.file, reason: f.reason, skipped: !!s.skipped, boxes: s.boxes.map(b => ({x: b.x, y: b.y, w: b.w, h: b.h, character: b.character, move: b.move}))};
  })};
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'labels.json'; document.body.appendChild(a); a.click(); a.remove();
}
$('importf').onchange = e => {
  const f = e.target.files[0]; if (!f) return; const r = new FileReader();
  r.onload = () => { try { const d = JSON.parse(r.result); (d.frames || []).forEach(fr => { state[fr.frame] = {boxes: fr.boxes || [], skipped: !!fr.skipped}; }); save(); render(); } catch (err) { alert('bad file: ' + err); } };
  r.readAsText(f);
};
$('prev').onclick = () => go(idx - 1); $('next').onclick = () => go(idx + 1);
$('skipbtn').onclick = () => { const c = cur(); c.skipped = !c.skipped; save(); render(); };
$('exportbtn').onclick = exportLabels;
$('src').textContent = 'Move list: ' + D.source + ' · labels autosave in this browser (localStorage)';
go(idx);
</script></body></html>
"""

if __name__ == "__main__":
    main()
