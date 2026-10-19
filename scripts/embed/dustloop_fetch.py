"""Build a Dustloop (GGST) move-image reference set for one or more characters.

Data source: Dustloop's Cargo table MoveData_GGST (the structured data behind the character
pages' move cards), queried via Special:CargoExport, plus the MediaWiki imageinfo API to resolve
each File: page to its full-size original. Downloads are sequential with a delay and a
descriptive User-Agent.

Output:
  data/dustloop/<character_slug>/<image_slug>.png
  data/dustloop/manifest.json   (one record per unique image file, sprite + hitbox variants)

Usage:
  python scripts/embed/dustloop_fetch.py "Anji Mito" "Faust" [--out data/dustloop] [--delay 0.75]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WIKI = "https://www.dustloop.com"
INDEX = f"{WIKI}/wiki/index.php"
API = f"{WIKI}/wiki/api.php"
UA = ("replanal-research/0.1 (Spaghetti Lab replay-analysis reference set; "
      "sequential, rate-limited, personal non-commercial use)")
LICENSE = {
    "source": "Dustloop Wiki (https://www.dustloop.com)",
    "content_license": "CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/) "
                       "unless otherwise noted, per Dustloop_Wiki:Copyrights",
    "game_assets": "Guilty Gear -Strive- game content and materials are trademarks and copyrights of "
                   "Arc System Works and its licensors. All rights reserved.",
    "use": "Local, non-commercial research reference only. Do not redistribute.",
}

PLAIN_NORMALS = {"5P", "5K", "c.S", "f.S", "5H", "5D", "5[D]", "2P", "2K", "2S", "2H", "2D",
                 "j.P", "j.K", "j.S", "j.H", "j.D"}


def _get(url: str, params: dict | None = None, delay: float = 0.75) -> bytes:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(delay)
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1} for {url[:120]}: {e}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _norm_title(name: str) -> str:
    return name.strip().replace("_", " ")


def _slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def categorize(row: dict) -> str:
    t, inp, name = str(row.get("type") or ""), str(row.get("input") or ""), str(row.get("name") or "")
    if "Throw" in name:
        return "throw"
    if t == "normal":
        return "normal" if inp in PLAIN_NORMALS else "command_normal"
    if t in ("special", "super"):
        return t
    if inp.startswith("236P ") or inp == "Afro Explosion":
        return "item"  # Faust What Could This Be? items
    return "system"  # Wild Assault, Finish Blow, Counter Blitz, movement, ...


def cargo_moves(chara: str, delay: float) -> list[dict]:
    raw = _get(INDEX, {
        "title": "Special:CargoExport", "tables": "MoveData_GGST",
        "fields": "chara,name,input,type,images,hitboxes",
        "where": f'chara="{chara}"', "format": "json", "limit": "500",
    }, delay)
    return json.loads(raw)


def imageinfo(titles: list[str], delay: float) -> dict[str, dict]:
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = json.loads(_get(API, {
            "action": "query", "prop": "imageinfo", "iiprop": "url|size|mime|sha1",
            "titles": "|".join("File:" + t for t in chunk), "format": "json",
        }, delay))
        for p in d["query"]["pages"].values():
            t = p["title"].removeprefix("File:")
            out[t] = p["imageinfo"][0] if "imageinfo" in p else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="+")
    ap.add_argument("--out", default="data/dustloop")
    ap.add_argument("--delay", type=float, default=0.75)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}  # file title -> record
    moves_all, no_image, missing_files = [], [], []
    for chara in args.characters:
        cslug = _slug(chara)
        page_url = f"{WIKI}/w/GGST/{chara.replace(' ', '_')}"
        rows = cargo_moves(chara, args.delay)
        print(f"{chara}: {len(rows)} Cargo move rows")
        for r in rows:
            cat = categorize(r)
            r["input"] = str(r["input"])
            mv = {"character": chara, "name": str(r.get("name") or r["input"]), "input": r["input"],
                  "category": cat, "cargo_type": r.get("type"), "page_url": page_url}
            imgs = [_norm_title(x) for x in (r.get("images") or []) if x and x.strip()]
            hbs = [_norm_title(x) for x in (r.get("hitboxes") or []) if x and x.strip()]
            mv["sprite_files"], mv["hitbox_files"] = imgs, hbs
            moves_all.append(mv)
            if not imgs and not hbs:
                no_image.append(f"{chara} {mv['name']} ({mv['input']})")
            for variant, files in (("sprite", imgs), ("hitbox", hbs)):
                for idx, f in enumerate(files):
                    rec = records.setdefault(f, {
                        "file_title": f, "character": chara, "character_slug": cslug,
                        "variant": variant, "frame_index": idx, "moves": [],
                    })
                    ref = {k: mv[k] for k in ("name", "input", "category")}
                    if ref not in rec["moves"]:
                        rec["moves"].append(ref)

    info = imageinfo(sorted(records), args.delay)
    for f, rec in records.items():
        ii = info.get(f)
        if not ii:
            missing_files.append(f)
            rec["missing"] = True
            continue
        # primary move: prefer real moves over system entries (Finish Blow reuses Kou's image)
        # and base versions over Charged/Enhanced variants
        prim = min(rec["moves"], key=lambda m: (m["category"] == "system",
                   any(w in m["name"] for w in ("Charged", "Enhanced", "Finish Blow")), len(m["name"])))
        stem = _slug(Path(f).stem)
        stem = re.sub(rf"^ggst_{rec['character_slug']}_", "", stem)
        local = out / rec["character_slug"] / f"{stem}.png"
        rec.update({
            "id": f"{rec['character_slug']}/{stem}",
            "move_name": prim["name"], "input": prim["input"], "category": prim["category"],
            "has_hitbox_overlay": rec["variant"] == "hitbox",
            "image_url": ii["url"], "file_page_url": ii["descriptionurl"],
            "page_url": f"{WIKI}/w/GGST/{rec['character'].replace(' ', '_')}",
            "width": ii.get("width"), "height": ii.get("height"), "mime": ii.get("mime"),
            "wiki_sha1": ii.get("sha1"), "local_path": local.as_posix(),
        })

    ok = 0
    for rec in records.values():
        if rec.get("missing"):
            continue
        local = Path(rec["local_path"])
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and hashlib.sha1(local.read_bytes()).hexdigest() == rec["wiki_sha1"]:
            ok += 1
            continue
        try:
            data = _get(rec["image_url"], delay=args.delay)
        except Exception as e:  # noqa: BLE001
            rec["download_error"] = str(e)
            print("  FAIL", rec["image_url"], e)
            continue
        if not data.startswith(b"\x89PNG"):
            rec["download_error"] = f"not a PNG (starts {data[:8]!r})"
            print("  FAIL", rec["image_url"], rec["download_error"])
            continue
        local.write_bytes(data)
        ok += 1
        print(f"  [{ok}/{len(records)}] {local}")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": LICENSE,
        "source_query": "Special:CargoExport MoveData_GGST + api.php imageinfo",
        "characters": args.characters,
        "n_images": sum(1 for r in records.values() if not r.get("missing") and "download_error" not in r),
        "moves_without_images": no_image,
        "missing_files": missing_files,
        "records": [r for r in records.values() if not r.get("missing")],
        "moves": moves_all,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"downloaded/verified {ok}, missing on wiki {len(missing_files)}, moves w/o image {len(no_image)}")


if __name__ == "__main__":
    main()
