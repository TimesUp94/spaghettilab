"""Evaluate EmbeddingGemma-2 on the hand-labeled set-17 crops.

Reads output/embed_test/labels.json (boxes around Anji/Faust with a move or
state label) and reports, with leave-one-frame-out (LOFO) so a crop is never
matched against its own frame:

  1. character  - Anji vs Faust          (crop kNN, crop->Dustloop, zero-shot text)
  2. state      - attack / hitstun / blocking / movement / knockdown (crop kNN)
  3. move       - exact move of attack crops
                  a) kNN against the user's other labeled crops (moves with >=2 examples)
                  b) rank of the true move among that character's Dustloop moves

Labels "idle/walk/dash" and "other" (the user's jumps/airdashes) count as movement.
Writes output/embed_test/eval/report.json and a few example sheets.

Usage: .venv/Scripts/python.exe scripts/embed/eval_labels.py [--budgets 280 1120] [--pad 0.15]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_embed as ge  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "output" / "embed_test"
OUT = TEST / "eval"

STATE_OF = {
    "hitstun": "hitstun",
    "blocking": "blocking",
    "idle/walk/dash": "movement",
    "other": "movement",
    "knockdown/wakeup": "knockdown",
}


def state_of(move: str) -> str:
    return STATE_OF.get(move, "attack")


def load_boxes(pad: float):
    labels = json.loads((TEST / "labels.json").read_text(encoding="utf-8"))
    boxes = []
    for fr in labels["frames"]:
        if fr.get("skipped"):
            continue
        img = Image.open(TEST / "label_frames" / fr["file"]).convert("RGB")
        W, H = img.size
        for b in fr.get("boxes", []):
            if not b.get("move") or not b.get("character"):
                continue
            px, py = b["w"] * pad, b["h"] * pad
            x0, y0 = max(0, b["x"] - px), max(0, b["y"] - py)
            x1, y1 = min(W, b["x"] + b["w"] + px), min(H, b["y"] + b["h"] + py)
            boxes.append({
                "frame": fr["frame"], "file": fr["file"], "character": b["character"],
                "move": b["move"], "state": state_of(b["move"]),
                "crop": img.crop((int(x0), int(y0), int(x1), int(y1))),
            })
    return boxes


def lofo_knn(emb, labels, frames, k, mask=None):
    """Leave-one-frame-out kNN (majority vote, ties -> nearest). Returns predictions."""
    S = emb @ emb.T
    idx_all = np.arange(len(labels))
    preds = []
    for i in idx_all:
        if mask is not None and not mask[i]:
            preds.append(None)
            continue
        cand = [j for j in idx_all if frames[j] != frames[i] and (mask is None or mask[j])]
        if not cand:
            preds.append(None)
            continue
        cand = sorted(cand, key=lambda j: -S[i, j])[:k]
        votes = collections.Counter(labels[j] for j in cand)
        best = max(votes.values())
        preds.append(next(labels[j] for j in cand if votes[labels[j]] == best))
    return preds


def acc(preds, truth, mask=None):
    pairs = [(p, t) for i, (p, t) in enumerate(zip(preds, truth))
             if p is not None and (mask is None or mask[i])]
    return (sum(p == t for p, t in pairs) / len(pairs), len(pairs)) if pairs else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[280, 1120])
    ap.add_argument("--pad", type=float, default=0.15)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    boxes = load_boxes(args.pad)
    chars = [b["character"] for b in boxes]
    states = [b["state"] for b in boxes]
    moves = [b["move"] for b in boxes]
    frames = [b["frame"] for b in boxes]
    n = len(boxes)
    print(f"{n} boxes from {len(set(frames))} frames")

    man = json.loads((ROOT / "data" / "dustloop" / "manifest.json").read_text(encoding="utf-8"))
    dl = np.load(TEST / "dustloop_embeddings.npz", allow_pickle=True)
    dl_ids = list(dl["ids"])
    rec_by_id = {r["id"]: r for r in man["records"]}
    # every (character, input) a Dustloop image depicts
    dl_moves = [[(rec_by_id[i]["character"], m["input"]) for m in rec_by_id[i]["moves"]] for i in dl_ids]
    dl_char = [rec_by_id[i]["character"] for i in dl_ids]

    model = ge.load_model(modalities=("text", "vision"))
    text_emb = ge.embed_texts(model, ["Anji Mito from Guilty Gear Strive", "Faust from Guilty Gear Strive"])
    report = {"n_boxes": n, "n_frames": len(set(frames)),
              "class_counts": {"character": collections.Counter(chars),
                               "state": collections.Counter(states)},
              "budgets": {}}

    attack = np.array([s == "attack" for s in states])
    move_counts = collections.Counter((c, m) for c, m, a in zip(chars, moves, attack) if a)
    repeat_move = np.array([a and move_counts[(c, m)] >= 2 for c, m, a in zip(chars, moves, attack)])

    for budget in args.budgets:
        emb = ge.normalize(ge.embed_images(model, [b["crop"] for b in boxes], token_budget=budget))
        R = {}

        # 1. character
        majority = collections.Counter(chars).most_common(1)[0][1] / n
        R["character"] = {"majority_baseline": majority}
        for k in (1, 5):
            R["character"][f"crop_knn{k}"] = acc(lofo_knn(emb, chars, frames, k), chars)[0]
        zs = emb @ text_emb.T
        R["character"]["zero_shot_text"] = acc(
            [("Anji Mito", "Faust")[int(np.argmax(r))] for r in zs], chars)[0]
        dl_emb = ge.normalize(dl[f"emb_{budget}"]) if f"emb_{budget}" in dl.files else None
        if dl_emb is not None:
            nn = (emb @ dl_emb.T).argmax(axis=1)
            R["character"]["crop_to_dustloop_nn"] = acc([dl_char[j] for j in nn], chars)[0]

        # 2. state
        maj_state = collections.Counter(states).most_common(1)[0]
        R["state"] = {"majority_baseline": maj_state[1] / n, "majority_class": maj_state[0]}
        for k in (1, 5):
            p = lofo_knn(emb, states, frames, k)
            R["state"][f"crop_knn{k}"] = acc(p, states)[0]
            if k == 5:
                conf = collections.Counter(zip(states, p))
                R["state"]["confusion_knn5"] = {f"{t}->{q}": v for (t, q), v in sorted(conf.items())}
        # attack vs not (binary), useful as a "something is happening" detector
        bin_lab = ["attack" if a else "non-attack" for a in attack]
        R["state"]["attack_vs_not_knn5"] = acc(lofo_knn(emb, bin_lab, frames, 5), bin_lab)[0]
        R["state"]["attack_vs_not_majority"] = max(attack.mean(), 1 - attack.mean())

        # 3a. move vs user's own crops (same character only, moves seen >=2 times)
        cm = [f"{c}|{m}" for c, m in zip(chars, moves)]
        p1 = lofo_knn(emb, cm, frames, 1, mask=attack)
        a_rep, n_rep = acc(p1, cm, mask=repeat_move)
        n_classes = len([1 for v in move_counts.values() if v >= 2])
        R["move_vs_own_crops"] = {
            "evaluable_crops": n_rep, "classes_with_2plus": n_classes,
            "knn1_acc": a_rep,
            "chance_approx": 1 / max(1, len(move_counts)),
            "note": "candidates are all other attack crops (any move, either character)",
        }

        # 3b. move vs Dustloop, character given: rank of true move
        if dl_emb is not None:
            ranks = []
            examples = []
            for i in np.where(attack)[0]:
                c, m = chars[i], moves[i]
                s = emb[i] @ dl_emb.T
                best = {}
                for j, mv_list in enumerate(dl_moves):
                    for (mc, mi) in mv_list:
                        if mc == c:
                            best[mi] = max(best.get(mi, -1), float(s[j]))
                if m not in best:
                    continue
                order = sorted(best, key=lambda x: -best[x])
                ranks.append((order.index(m) + 1, len(order)))
                examples.append({"frame": frames[i], "character": c, "true": m,
                                 "top5": order[:5], "rank": order.index(m) + 1,
                                 "top1_score": best[order[0]], "true_score": best[m]})
            r = np.array([x[0] for x in ranks]); sizes = np.array([x[1] for x in ranks])
            R["move_vs_dustloop"] = {
                "evaluated": len(ranks),
                "top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
                "median_rank": float(np.median(r)), "median_n_moves": float(np.median(sizes)),
                "chance_top1": float(np.mean(1 / sizes)), "chance_top5": float(np.mean(np.minimum(5, sizes) / sizes)),
            }
            (OUT / f"dustloop_examples_{budget}.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")

        report["budgets"][budget] = R
        print(json.dumps({budget: R}, indent=1, default=str))

    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    # contact sheet of a few crops for sanity
    thumbs = [b["crop"].resize((160, 160)) for b in boxes[:24]]
    sheet = Image.new("RGB", (160 * 8, 160 * 3), "grey")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 8) * 160, (i // 8) * 160))
    sheet.save(OUT / "crops_sample.jpg")


if __name__ == "__main__":
    main()
