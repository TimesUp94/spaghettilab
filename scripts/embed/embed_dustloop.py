"""Embed the Dustloop reference set (data/dustloop/manifest.json) with EmbeddingGemma-2 and run a
nearest-neighbour sanity check.

Writes output/embed_test/dustloop_embeddings.npz with:
  ids            (N,)  image ids (manifest record "id", e.g. "faust/2d")
  character      (N,)  character per image
  variant        (N,)  "sprite" | "hitbox"
  move_name      (N,)  primary move name per image
  emb_280        (N,768) image embeddings, vision budget 280
  emb_1120       (N,768) image embeddings, vision budget 1120
  text_img       (N,)  "<Character> <move name>" string per image
  emb_text_img   (N,768) SearchQuery-prefixed text embedding of text_img
  move_keys      (M,)  unique "<Character> <move name>" strings over all Cargo moves with images
  emb_text       (M,768) their text embeddings (SearchQuery prefix)

Transparent sprites are flattened onto neutral grey by gemma_embed._to_pil (the hidden RGB under
alpha differs between files and would otherwise leak into the embeddings).

Usage: python scripts/embed/embed_dustloop.py [--manifest data/dustloop/manifest.json]
                                              [--out output/embed_test/dustloop_embeddings.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_embed as ge  # noqa: E402


def nn_report(emb: np.ndarray, chars: np.ndarray, variants: np.ndarray, moves: list[set],
              label: str) -> dict:
    s = ge.cosine_sim(emb, emb)
    np.fill_diagonal(s, -np.inf)
    out = {}

    def frac(mask_rows, exclude=None):
        hits, n = 0, 0
        for i in np.where(mask_rows)[0]:
            row = s[i].copy()
            if exclude is not None:
                row[exclude(i)] = -np.inf
            row[~pool] = -np.inf
            j = int(np.argmax(row))
            if not np.isfinite(row[j]):
                continue
            hits += chars[j] == chars[i]
            n += 1
        return (100.0 * hits / n if n else float("nan")), n

    all_rows = np.ones(len(chars), bool)
    pool = all_rows
    out["all"] = frac(all_rows)
    # exclude neighbours depicting the same move (hitbox twin, multi-frame of same move)
    same_move = lambda i: np.array([bool(moves[i] & m) for m in moves])  # noqa: E731
    out["all_excl_same_move"] = frac(all_rows, same_move)
    pool = variants == "sprite"
    out["sprite_only"] = frac(pool)
    out["sprite_only_excl_same_move"] = frac(pool, same_move)
    pool = variants == "hitbox"
    out["hitbox_only_excl_same_move"] = frac(pool, same_move)
    # per character, all images, excluding same move
    pool = all_rows
    for c in np.unique(chars):
        out[f"{c}_excl_same_move"] = frac(chars == c, same_move)
    # mean intra vs inter character cosine
    s2 = ge.cosine_sim(emb, emb)
    same = chars[:, None] == chars[None, :]
    off = ~np.eye(len(chars), dtype=bool)
    out["mean_cos_same_char"] = float(s2[same & off].mean())
    out["mean_cos_cross_char"] = float(s2[~same].mean())
    print(f"--- NN same-character rate [{label}]")
    for k, v in out.items():
        print(f"  {k:32s} {v[0]:6.1f}%  (n={v[1]})" if isinstance(v, tuple) else f"  {k:32s} {v:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dustloop/manifest.json")
    ap.add_argument("--out", default="output/embed_test/dustloop_embeddings.npz")
    args = ap.parse_args()
    m = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    recs = [r for r in m["records"] if "download_error" not in r]
    ids = np.array([r["id"] for r in recs])
    chars = np.array([r["character"] for r in recs])
    variants = np.array([r["variant"] for r in recs])
    move_name = np.array([r["move_name"] for r in recs])
    moves = [{(r["character"], mv["name"]) for mv in r["moves"]} for r in recs]
    paths = [r["local_path"] for r in recs]

    model = ge.load_model(("text", "vision"))
    embs = {}
    for budget, bs in ((280, 16), (1120, 4)):
        t0 = time.time()
        embs[budget] = ge.embed_images(model, paths, token_budget=budget, batch_size=bs)
        print(f"budget {budget}: {embs[budget].shape} in {time.time() - t0:.1f}s, "
              f"nan={np.isnan(embs[budget]).any()}")

    text_img = np.array([f"{r['character']} {r['move_name']}" for r in recs])
    move_keys = np.array(sorted({f"{mv['character']} {mv['name']}" for mv in m["moves"]
                                 if mv["sprite_files"] or mv["hitbox_files"]}))
    emb_text_img = ge.embed_texts(model, list(text_img))
    emb_text = ge.embed_texts(model, list(move_keys))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids, character=chars, variant=variants, move_name=move_name,
             emb_280=embs[280], emb_1120=embs[1120], text_img=text_img, emb_text_img=emb_text_img,
             move_keys=move_keys, emb_text=emb_text)
    print("saved", out)

    report = {}
    for budget in (280, 1120):
        report[f"img_{budget}"] = nn_report(embs[budget], chars, variants, moves, f"budget {budget}")

    # bonus: text->image move retrieval. For each image, rank the M move strings; is its own
    # move (any of its moves) top-1 / top-5? And does the top-1 at least name the right character?
    key_char = np.array([k.split(" ", 1)[0] for k in move_keys])
    for budget in (280, 1120):
        s = ge.cosine_sim(embs[budget], emb_text)
        t1 = t5 = cc = 0
        for i, r in enumerate(recs):
            own = {f"{r['character']} {mv['name']}" for mv in r["moves"]}
            order = np.argsort(-s[i])
            t1 += move_keys[order[0]] in own
            t5 += bool(own & set(move_keys[order[:5]]))
            cc += key_char[order[0]] == r["character"].split(" ", 1)[0]
        n = len(recs)
        report[f"text_retrieval_{budget}"] = {"top1": t1 / n, "top5": t5 / n, "char_top1": cc / n,
                                              "n_moves": len(move_keys)}
        print(f"--- text->image [{budget}] own-move top1 {100 * t1 / n:.1f}%  top5 {100 * t5 / n:.1f}%"
              f"  right-character top1 {100 * cc / n:.1f}%  (chance top1 ~{100 / len(move_keys):.1f}%)")
    rp = out.with_name("dustloop_sanity.json")
    rp.write_text(json.dumps(report, indent=2, default=lambda x: x), encoding="utf-8")
    print("saved", rp)


if __name__ == "__main__":
    main()
