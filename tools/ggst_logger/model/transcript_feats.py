"""Whole-match embeddings for the move transcript (ROADMAP point 2): every replay of the v2 dataset
(the 1,090 replays of transcript_events.parquet) at 15 fps through frozen V-JEPA 2 ViT-L, in three
square views, plus both players' positions and the CV pipeline's HUD readings on the same timeline.

Views (256 x 256, one 16-frame clip = 1.07 s through the encoder, mean over space per tubelet):
- scene: the whole frame squashed to 256 x 256 (the openers' scene stream, measured in point 1);
- p1, p2: a 640 x 640 square (1080p pixels, as video_crops.S) on each player, centred a quarter of
  the way towards the opponent and 20 px below the player's point, mirrored so the player faces
  right (towards the opponent). The positions come from the player network (video_players_*.pt,
  default the test-fold net so test replays see no net trained on them), decoded per 16-frame
  window with the replay's running character vote, smoothed over +-2 frames.

Timeline (all in 60 fps video frames, counted from the first decoded frame like the pipeline's
frame_number and the events' start_frame):
- frame j of the 15 fps grid is video frame 4j;
- tubelet i (V-JEPA 2 merges 2 frames) covers grid frames 2i, 2i+1, i.e. video frames 8i .. 8i+7;
  the stored tub_frame is 8i;
- hud row j is video frame 2j (the pipeline samples every 2nd frame): video_openers.parquet_inputs'
  23 features (per side: HP, read flag, pending, conf, dHP, dpending, tension, dtension, combo
  counter on this side, counter-side flag, 0; then timer present), forward-filled where the
  pipeline has no row (hud_ok = False there). Tubelet i covers hud rows 4i .. 4i+3.

Per replay -> <data>/ggst_dataset/transcript/feats/<stem>.npz (written atomically; existing files
are skipped, so the run resumes):
  scene, p1, p2   [Ntub, 1024] fp16       tub_frame [Ntub] int32
  frame [N15] int32, pos [N15, 2, 2] float32 (1920x1080 px, P1/P2, x/y), flip [N15, 2] bool
  chars_logp [2, C] float32 (summed over frames; names in meta.json)
  hud [N30, 23] fp16, hud_ok [N30] bool, n_frames, fps
Settings and the character list: transcript/feats/meta.json. Timing: transcript/feats_times.jsonl.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats.py smoke --n 3      # throughput, estimates
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats.py extract           # every replay (resumable)
    .venv/Scripts/python.exe tools/ggst_logger/model/transcript_feats.py check             # files, sizes, alignment
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUT, device, low_priority  # noqa: E402

TR = DATA / "transcript"
FEATS = TR / "feats"
RUNS = DATA / "runs"
STEP = 4                    # video frames per sampled frame (60 -> 15 fps)
CLIP = 16                   # sampled frames per V-JEPA 2 clip
CHUNK = 64                  # sampled frames per loader item (a multiple of CLIP)
DEC_W, DEC_H = 960, 540     # decode size (half of 1080p): the player squares are 320 px here
FW, FH = 1920, 1080
S = 640                     # player square side in 1080p px (video_crops.S)
TILE = 256
VIEWS = ("scene", "p1", "p2")
HF = "facebook/vjepa2-vitl-fpc64-256"
DIM = 1024
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_LOCAL_FFMPEG = DATA.parent / "ggst_mod" / "dl" / "ffmpeg-9.0.2-essentials_build" / "bin" / "ffmpeg.exe"
FFMPEG = os.environ.get("GGST_FFMPEG") or (str(_LOCAL_FFMPEG) if _LOCAL_FFMPEG.exists() else "ffmpeg")
HUD_NAMES = [f"p{s}_{n}" for s in (1, 2) for n in ("hp", "hp_read", "pending", "conf", "dhp", "dpending", "tension",
                                                   "dtension", "combo_mine", "combo_side", "zero")] + ["timer_present"]


def replays():
    """(stem, video) of the v2 dataset: the replays with dense labels."""
    import pyarrow.parquet as pq
    stems = sorted(set(pq.read_table(TR / "transcript_events.parquet", columns=["replay"]).column("replay").to_pylist()))
    vid = {r["replay"]: r["video"] for r in pq.read_table(DATA / "openers.parquet", columns=["replay", "video"]).to_pylist()}
    return [(s, vid[s]) for s in stems]


# ---- decode (loader workers) ---------------------------------------------------------------

def video_format(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return out


class Chunks(torch.utils.data.IterableDataset):
    """Every STEP-th frame of each replay at DEC_W x DEC_H RGB, in items of CHUNK frames; the
    replay's last item has last=True (and may be short or empty). Workers split the replays."""

    def __init__(self, tasks, threads, limit):
        self.tasks, self.threads, self.limit = tasks, threads, limit

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        mine = self.tasks if wi is None else self.tasks[wi.id::wi.num_workers]
        for stem, video in mine:
            yield from self.replay(stem, video)

    def replay(self, stem, video):
        t0 = time.time()
        limit = self.limit
        fps, w, h = video_format(video)
        canon = "fps=60," if abs(fps - 60) > 0.5 else ""
        cmd = [FFMPEG, "-hide_banner", "-v", "error", "-threads", str(self.threads), "-i", str(video), "-vf",
               f"{canon}select='not(mod(n\\,{STEP}))',scale={DEC_W}:{DEC_H}:flags=area",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=DEC_W * DEC_H * 3 * 8)
        n = DEC_W * DEC_H * 3
        buf, k, ci = [], 0, 0
        while True:
            if limit and k >= limit:
                break
            b = p.stdout.read(n)
            if len(b) < n:
                break
            buf.append(np.frombuffer(b, np.uint8).reshape(DEC_H, DEC_W, 3))
            k += 1
            if len(buf) == CHUNK:
                yield dict(stem=stem, ci=ci, frames=np.stack(buf), last=False, err="", fps=fps, dec_s=time.time() - t0)
                buf, ci = [], ci + 1
        p.stdout.close()
        if limit and k >= limit:
            p.kill()
        err = p.stderr.read().decode(errors="replace")[-500:]
        rc = p.wait()
        frames = np.stack(buf) if buf else np.zeros((0, DEC_H, DEC_W, 3), np.uint8)
        yield dict(stem=stem, ci=ci, frames=frames, last=True, err="" if (rc == 0 or limit) else f"ffmpeg rc {rc}: {err}",
                   fps=fps, dec_s=time.time() - t0)


# ---- GPU side -----------------------------------------------------------------------------------

class Encoder:
    def __init__(self, players_net, vjepa=True):
        import torch
        import video_players as vp
        from transformers import VJEPA2Model
        self.torch, self.vp, self.dev = torch, vp, device()
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.m = VJEPA2Model.from_pretrained(HF, dtype=self.dt, attn_implementation="sdpa").to(self.dev).eval() if vjepa else None
        ck = torch.load(players_net, map_location=self.dev)
        self.chars = ck["chars"]
        self.pn = vp.make_net(len(ck["chars"])).to(self.dev).eval()
        self.pn.load_state_dict(ck["state"])
        self.mean = torch.tensor(IMNET[0], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(IMNET[1], device=self.dev).view(1, 3, 1, 1)

    def players(self, x, vote):
        """x [n, 3, DEC_H, DEC_W] float 0..1 -> positions [n, 2, 2] (1080p px), new vote [2, C].
        The player net's own input: the 256x144 tile (area) upsampled to 512x288, ImageNet-normalised;
        P1/P2 decided per 16-frame window with the replay's running character vote."""
        F = self.torch.nn.functional
        small = F.interpolate(x, size=(144, 256), mode="area")
        inp = (F.interpolate(small, size=(self.vp.IN_H, self.vp.IN_W), mode="bilinear", align_corners=False) - self.mean) / self.std
        pos = []
        for j in range(0, len(x), CLIP):
            with self.torch.autocast("cuda", dtype=self.torch.float16, enabled=self.dev == "cuda"):
                out = self.pn(inp[j:j + CLIP].float())
            gl = self.torch.log_softmax(out["gchar"].float(), -1)
            vote = gl.sum(0) if vote is None else vote + gl.sum(0)
            chars = vote.argmax(-1)[None].expand(len(gl), 2)
            p, _ = self.vp.decode(out, chars, window=True)
            pos.append(p)
        return self.torch.cat(pos) * self.torch.tensor([FW, FH], device=self.dev), vote

    def squares(self, x, pos):
        """Player squares [2, n, 3, TILE, TILE] (mirrored so each player faces right) and flips [n, 2]."""
        from torchvision.ops import roi_align
        n = len(x)
        tiles, flips = [], []
        for s in (0, 1):
            xs, ys, xo = pos[:, s, 0], pos[:, s, 1], pos[:, 1 - s, 0]
            cx = (xs + 0.25 * (xo - xs)) / 2           # 1080p -> decode px
            cy = (ys + 20) / 2
            h = S / 4
            boxes = self.torch.stack([self.torch.arange(n, device=self.dev, dtype=x.dtype), cx - h, cy - h, cx + h, cy + h], 1)
            t = roi_align(x, boxes.to(x.dtype), output_size=TILE, spatial_scale=1.0, sampling_ratio=2, aligned=True)
            flip = xo < xs
            t = self.torch.where(flip[:, None, None, None], t.flip(-1), t)
            tiles.append(t)
            flips.append(flip)
        return tiles, self.torch.stack(flips, 1)

    def vjepa(self, views):
        """views: list of [n, 3, TILE, TILE] 0..1 -> list of [ceil(n/2), DIM] fp16 (16-frame clips, the
        last padded with its final frame)."""
        torch = self.torch
        n = len(views[0])
        pad = (-n) % CLIP
        clips = []
        for v in views:
            v = (v - self.mean) / self.std
            if pad:
                v = torch.cat([v, v[-1:].expand(pad, -1, -1, -1)])
            clips.append(v.view(-1, CLIP, 3, TILE, TILE))
        x = torch.cat(clips).to(self.dt)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            h = self.m(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        h = h.view(len(x), CLIP // 2, -1, DIM).float().mean(2).reshape(len(views), -1, DIM)
        return [z[:(n + 1) // 2].half().cpu().numpy() for z in h]

    def chunk(self, frames, st):
        """One loader item -> appends to the replay state st."""
        torch = self.torch
        if not len(frames):
            return
        with torch.no_grad():
            x = frames.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            pos, st["vote"] = self.players(x, st["vote"])
            # median over +-2 frames, with the previous chunk's last 2 frames on the left
            prev = st["pos_tail"]
            ext = torch.cat([prev, pos]) if prev is not None else pos
            k = len(ext) - len(pos)
            sm = torch.stack([ext[max(0, i - 2):i + 3].median(0).values for i in range(k, len(ext))])
            st["pos_tail"] = pos[-2:]
            scene = torch.nn.functional.interpolate(x, size=(TILE, TILE), mode="bilinear", antialias=True, align_corners=False)
            (p1, p2), flip = self.squares(x, sm)
            del x
            f = self.vjepa([scene, p1, p2])
        for name, z in zip(VIEWS, f):
            st[name].append(z)
        st["pos"].append(sm.cpu().numpy().astype(np.float32))
        st["flip"].append(flip.cpu().numpy())


def hud(stem, n_frames):
    """[N30, 23] fp16 HUD features on rows of video frame 2j, forward-filled, and the read mask."""
    import video_openers as vo
    X, fr = vo.parquet_inputs(RUNS / stem / "frames" / f"{stem}_frames.parquet")
    n30 = (n_frames + 1) // 2
    out = np.zeros((n30, X.shape[1]), np.float32)
    ok = np.zeros(n30, bool)
    idx = np.asarray(fr) // 2
    keep = (idx >= 0) & (idx < n30)
    out[idx[keep]] = X[keep]
    ok[idx[keep]] = True
    last = np.maximum.accumulate(np.where(ok, np.arange(n30), -1))
    out = np.where((last >= 0)[:, None], out[np.maximum(last, 0)], 0)
    return out.astype(np.float16), ok


def write(stem, st, enc):
    n15 = sum(len(p) for p in st["pos"])
    n_frames = n15 * STEP              # sampled frames cover video frames 0 .. 4*n15-1
    feats = {v: np.concatenate(st[v]) if st[v] else np.zeros((0, DIM), np.float16) for v in VIEWS}
    ntub = len(feats["scene"])
    h, ok = hud(stem, n_frames)
    tmp = FEATS / f"{stem}.tmp.npz"
    np.savez(tmp, **feats, tub_frame=(np.arange(ntub) * 2 * STEP).astype(np.int32),
             frame=(np.arange(n15) * STEP).astype(np.int32),
             pos=np.concatenate(st["pos"]) if st["pos"] else np.zeros((0, 2, 2), np.float32),
             flip=np.concatenate(st["flip"]) if st["flip"] else np.zeros((0, 2), bool),
             chars_logp=(st["vote"].cpu().numpy() if st["vote"] is not None else np.zeros((2, len(enc.chars)))).astype(np.float32),
             hud=h, hud_ok=ok, n_frames=np.int32(n_frames), fps=np.float32(st["fps"]))
    os.replace(tmp, FEATS / f"{stem}.npz")
    return ntub, n15, (FEATS / f"{stem}.npz").stat().st_size


def run(tasks, players_net, workers=3, threads=6, limit=0, write_files=True):
    import torch
    FEATS.mkdir(parents=True, exist_ok=True)
    enc = Encoder(players_net)
    meta = dict(backbone=HF, views=list(VIEWS), step=STEP, fps=60 / STEP, clip=CLIP, tile=TILE, decode=[DEC_W, DEC_H],
                player_square_px=S, player_centre="player + 0.25 * (opponent - player) in x, player y + 20; mirrored to face right",
                scene="whole frame squashed to 256x256", players_net=Path(players_net).name, chars=enc.chars,
                tub_frame="video frame 8i (covers 8i..8i+7)", frame="video frame 4j", hud_row="video frame 2j", hud_names=HUD_NAMES)
    if write_files:
        (FEATS / "meta.json").write_text(json.dumps(meta, indent=1))
    dl = torch.utils.data.DataLoader(Chunks(tasks, threads, limit), batch_size=None, num_workers=workers,
                                     prefetch_factor=2 if workers else None, pin_memory=True)
    if device() == "cuda":
        torch.cuda.reset_peak_memory_stats()
    state = {}
    t0 = time.time()
    done, frames_total, gpu_s, stats = 0, 0, 0.0, []
    tlog = open(TR / "feats_times.jsonl", "a") if write_files else None
    for item in dl:
        s = item["stem"]
        st = state.setdefault(s, dict(scene=[], p1=[], p2=[], pos=[], flip=[], vote=None, pos_tail=None, t0=time.time(),
                                      gpu=0.0, fps=float(item["fps"])))
        t1 = time.time()
        enc.chunk(item["frames"], st)
        if device() == "cuda":
            torch.cuda.synchronize()
        st["gpu"] += time.time() - t1
        if not item["last"]:
            continue
        del state[s]
        n15 = sum(len(p) for p in st["pos"])
        if item["err"] or n15 == 0:
            print(f"  SKIP {s}: {item['err'] or 'no frames decoded'}", flush=True)
            stats.append(dict(stem=s, skipped=item["err"] or "no frames"))
            continue
        ntub, n15, size = write(s, st, enc) if write_files else (sum(len(z) for z in st["scene"]), n15, 0)
        done += 1
        frames_total += n15
        gpu_s += st["gpu"]
        rec = dict(stem=s, n15=n15, ntub=ntub, bytes=size, gpu_s=round(st["gpu"], 1), dec_s=round(float(item["dec_s"]), 1),
                   wall_s=round(time.time() - st["t0"], 1))
        stats.append(rec)
        if tlog:
            tlog.write(json.dumps(rec) + "\n")
            tlog.flush()
        el = time.time() - t0
        if done % 10 == 1 or done == len(tasks):
            left = len(tasks) - done
            print(f"  {done}/{len(tasks)} {s}: {n15} frames, {size / 2 ** 20:.1f} MB | total {frames_total} frames in {el:.0f}s "
                  f"= {frames_total / el:.0f} fr/s wall ({3 * frames_total / max(gpu_s, 1e-6):.0f} view-fr/s GPU), "
                  f"eta {left * el / done / 3600:.1f} h", flush=True)
    if tlog:
        tlog.close()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device() == "cuda" else 0
    return dict(replays=done, frames=frames_total, seconds=round(time.time() - t0, 1), gpu_seconds=round(gpu_s, 1),
                peak_gpu_gb=round(peak, 2), stats=stats)


def smoke(n, players_net, limit, workers, threads):
    """A few replays (not written): decode and GPU throughput, MB per replay, estimates for all."""
    import pyarrow.parquet as pq
    tasks = replays()
    pick = [tasks[i] for i in np.linspace(0, len(tasks) - 1, n).astype(int)]
    r = run(pick, players_net, workers=workers, threads=threads, limit=limit, write_files=False)
    nf = {}
    for s, _ in tasks:
        t = pq.read_metadata(RUNS / s / "frames" / f"{s}_frames.parquet")
        nf[s] = 2 * t.num_rows
    total15 = sum(v for v in nf.values()) / STEP
    fps_wall = r["frames"] / r["seconds"]
    fps_gpu = r["frames"] / max(r["gpu_seconds"], 1e-6)
    bytes_per_tub = 3 * DIM * 2
    est = dict(sampled_frames_total=int(total15), wall_fr_s=round(fps_wall, 1), gpu_fr_s_per_3views=round(fps_gpu, 1),
               gpu_view_fr_s=round(3 * fps_gpu, 1), est_hours_wall=round(total15 / fps_wall / 3600, 2),
               est_hours_gpu=round(total15 / fps_gpu / 3600, 2),
               est_gb=round((total15 / 2 * bytes_per_tub + total15 * 2 * (23 * 2 + 1) + total15 * 18) / 2 ** 30, 2))
    out = dict(n=n, limit=limit, clip=CLIP, workers=workers, threads=threads, peak_gpu_gb=r["peak_gpu_gb"], run_seconds=r["seconds"],
               frames=r["frames"], gpu_seconds=r["gpu_seconds"], per_replay=r["stats"], **est)
    print(json.dumps(out, indent=1), flush=True)
    TR.mkdir(parents=True, exist_ok=True)
    (TR / f"transcript_feats_smoke_{n}_{limit}_clip{CLIP}.json").write_text(json.dumps(out, indent=1))


def dump(stem, start_s, n, players_net):
    """Visual check: n sampled frames from start_s seconds of a replay -> transcript/debug/<stem>_<s>.jpg
    (rows: scene, P1 square, P2 square; the player points drawn on the scene)."""
    import cv2
    vid = dict(replays())[stem]
    enc = Encoder(players_net, vjepa=False)
    ch = Chunks([(stem, vid)], 4, 0)
    frames = []
    for it in ch.replay(stem, vid):
        frames.append(it["frames"])
        if sum(len(f) for f in frames) * STEP >= (start_s * 60 + n * STEP * 8):
            break
    allf = np.concatenate(frames)
    j0 = int(start_s * 60 / STEP)
    sel = allf[j0:j0 + n * 4:4]
    with torch.no_grad():
        x = torch.from_numpy(sel).to(enc.dev).permute(0, 3, 1, 2).float().div(255)
        pos, vote = enc.players(x, None)
        (p1, p2), flip = enc.squares(x, pos)
        scene = torch.nn.functional.interpolate(x, size=(TILE, TILE), mode="bilinear", antialias=True, align_corners=False)
    def im(t):
        return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    sc, a, b = im(scene), im(p1), im(p2)
    for k in range(len(sc)):
        for s_, c in ((0, (255, 0, 0)), (1, (0, 0, 255))):
            px, py = pos[k, s_].cpu().numpy() / [FW / TILE, FH / TILE]
            cv2.circle(sc[k], (int(px), int(py)), 4, c, -1)
    grid = np.vstack([np.hstack(list(r)) for r in (sc, a, b)])
    (TR / "debug").mkdir(parents=True, exist_ok=True)
    out = TR / "debug" / f"{stem}_{start_s}.jpg"
    cv2.imwrite(str(out), grid[:, :, ::-1])
    print(out, "chars", [enc.chars[i] for i in vote.argmax(-1).tolist()], "flip", flip.cpu().numpy().tolist())


def check():
    """Every replay has a file; shapes agree with each other and with the frames parquet; sizes."""
    import pyarrow.parquet as pq
    tasks = replays()
    miss, bad, size, ntub = [], [], 0, 0
    for s, _ in tasks:
        f = FEATS / f"{s}.npz"
        if not f.exists():
            miss.append(s)
            continue
        size += f.stat().st_size
        z = np.load(f)
        n15, nt = len(z["frame"]), len(z["scene"])
        ntub += nt
        rows = pq.read_metadata(RUNS / s / "frames" / f"{s}_frames.parquet").num_rows
        ok = (nt == (n15 + 1) // 2 and len(z["p1"]) == nt == len(z["p2"]) and len(z["pos"]) == n15
              and abs(int(z["n_frames"]) - 2 * rows) <= STEP + 2 and np.isfinite(z["scene"].astype(np.float32)).all())
        if not ok:
            bad.append(dict(stem=s, n15=n15, ntub=nt, n_frames=int(z["n_frames"]), parquet_rows=rows))
    out = dict(replays=len(tasks), present=len(tasks) - len(miss), missing=miss, bad=bad, gb=round(size / 2 ** 30, 2), tubelets=ntub)
    print(json.dumps(out, indent=1))
    (TR / "transcript_feats_check.json").write_text(json.dumps(out, indent=1))


def main():
    global CLIP
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "extract", "check", "dump"])
    ap.add_argument("--stem")
    ap.add_argument("--start", type=int, default=60, help="dump: seconds")
    ap.add_argument("--n", type=int, default=3, help="smoke: replays")
    ap.add_argument("--limit", type=int, default=0, help="smoke: sampled frames per replay (0 = all)")
    ap.add_argument("--players", default="test", choices=["test", "final"], help="player net (test: trained without the test fold)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6, help="ffmpeg decode threads per worker")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--clip", type=int, default=CLIP, help="sampled frames per V-JEPA 2 clip (a divisor of CHUNK)")
    a = ap.parse_args()
    assert CHUNK % a.clip == 0
    CLIP = a.clip
    low_priority()
    if a.mode == "check":
        return check()
    net = str(OUT / f"video_players_{a.players}.pt")
    if a.mode == "dump":
        return dump(a.stem, a.start, 6, net)
    if a.mode == "smoke":
        return smoke(a.n, net, a.limit, a.workers, a.threads)
    tasks = [t for t in replays() if not a.only or t[0] in a.only]
    todo = [t for t in tasks if not (FEATS / f"{t[0]}.npz").exists()]
    print(f"{len(tasks) - len(todo)} replays already extracted, {len(todo)} to go", flush=True)
    if todo:
        r = run(todo, net, workers=a.workers, threads=a.threads)
        skipped = [x for x in r.pop("stats") if "skipped" in x]
        print(json.dumps(dict(r, skipped=skipped)), flush=True)


if __name__ == "__main__":
    main()
