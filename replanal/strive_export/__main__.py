"""Export saved Guilty Gear -Strive- replays as video files.

    python -m replanal.strive_export --out DIR [--count N | --all] [--since YYYY-MM-DD|today]
                                     [--start N] [--dry-run] [--json-progress]
    python -m replanal.strive_export calibrate --lang en
    python -m replanal.strive_export probe
    python -m replanal.strive_export classify shot.png ...

Start with the game running at the title screen (or main menu), keyboard
controls. The exporter takes over the game window: don't use the keyboard
while it runs. See docs/strive-export.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import threading

SUBCOMMANDS = ("export", "calibrate", "probe", "classify")


class Emitter:
    """Progress output: one JSON object per line (``--json-progress``, for
    the app) or readable lines."""

    def __init__(self, json_mode: bool):
        self.json_mode = json_mode
        self.lock = threading.Lock()

    def __call__(self, event: str, **kw) -> None:
        with self.lock:
            if self.json_mode:
                print(json.dumps({"event": event, **kw}, default=str, ensure_ascii=False), flush=True)
                return
            if event == "screen":
                print(f"  [screen] {kw.get('detail')}", flush=True)
            elif event == "entry":
                who = f"{kw.get('p1') or '?'} vs {kw.get('p2') or '?'}"
                print(f"[{kw['index']}] {kw.get('date') or '?'} {kw.get('time') or ''}  {who}  -> {kw['file']}", flush=True)
            elif event == "recorded":
                print(f"[{kw['index']}] recorded {kw['file']} ({kw.get('duration_s')} s"
                      f"{', partial' if kw.get('partial') else ''})", flush=True)
            elif event in ("warning", "error"):
                print(f"{event.upper()}: {kw.get('message')}", file=sys.stderr, flush=True)
            elif event == "done":
                print(f"done ({kw['reason']}): {len(kw['exported'])} exported, "
                      f"{len(kw['skipped'])} skipped, {kw['listed']} listed", flush=True)
                if kw.get("manifest"):
                    print(f"manifest: {kw['manifest']}", flush=True)
            else:
                rest = " ".join(f"{k}={v}" for k, v in kw.items())
                print(f"[{event}] {rest}", flush=True)


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--lang", default="auto",
                    help="UI language of the game for the text checks (default: auto-detect "
                         "among the reference languages; 'it' ships)")
    ap.add_argument("--refs-dir", action="append", default=[],
                    help="extra reference folder (holds <lang>/<screen>.jpg); searched first")
    ap.add_argument("--ffmpeg", help="ffmpeg.exe to use (FFmpeg 8+ for window capture)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m replanal.strive_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    ex = sub.add_parser("export", help="record saved replays (default)")
    ex.add_argument("--out", required=True, help="output folder for the mp4 files and manifest.json")
    n = ex.add_mutually_exclusive_group()
    n.add_argument("--count", type=int, help="export the N newest replays (after --start)")
    n.add_argument("--all", action="store_true", help="export every saved replay")
    ex.add_argument("--since", help="only replays played on or after this date (YYYY-MM-DD, today, yesterday)")
    ex.add_argument("--start", type=int, default=0, help="skip the N newest replays first")
    ex.add_argument("--dry-run", action="store_true", help="navigate and list the replays, record nothing")
    ex.add_argument("--json-progress", action="store_true",
                    help="one JSON object per line on stdout; read 'cancel' / 'stop' lines on stdin")
    ex.add_argument("--no-ocr", action="store_true", help="don't read dates/names (files named by index)")
    ex.add_argument("--overwrite", action="store_true", help="record again replays whose file exists")
    ex.add_argument("--date-order", choices=["DMY", "MDY", "YMD"],
                    help="order of the list's dates (default: from the language profile, else DMY)")
    ex.add_argument("--fps", type=int, default=60)
    ex.add_argument("--encoder", choices=["h264_nvenc", "h264_mf", "libx264"], help="try this encoder first")
    ex.add_argument("--max-minutes", type=float, default=15, help="longest replay to record")
    _common(ex)

    cal = sub.add_parser("calibrate", help="capture reference screens for another UI language")
    cal.add_argument("--lang", required=True, help="language code to store the references under, e.g. en")
    cal.add_argument("--refs-dir", help="where to write (default: %%LOCALAPPDATA%%/SpaghettiLab/strive_refs)")
    cal.add_argument("--ffmpeg")

    pr = sub.add_parser("probe", help="show the FFmpeg/capture/encoder choice and the current screen")
    _common(pr)
    pr.add_argument("--save", help="also save the capture to this file")

    cl = sub.add_parser("classify", help="classify saved screenshots (no game needed)")
    cl.add_argument("images", nargs="+")
    cl.add_argument("--lang", default="auto")
    cl.add_argument("--refs-dir", action="append", default=[])
    return ap


def parse_args(argv):
    argv = list(argv)
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "export")
    args = build_parser().parse_args(argv)
    if args.cmd == "export":
        if args.count is None and not args.all and not args.since:
            raise SystemExit("choose how many replays: --count N, --all or --since DATE")
        if args.count is not None and args.count < 1:
            raise SystemExit("--count must be at least 1")
    return args


def _capture(args, need_encoder: bool, emit, fps: int = 60, prefer=None):
    from .capture import Capture, choose_ffmpeg, ffmpeg_candidates, pick_encoder
    choice = choose_ffmpeg(ffmpeg_candidates(getattr(args, "ffmpeg", None)))
    if choice.warning:
        emit("warning", message=choice.warning)
    encoder = pick_encoder(choice.ffmpeg, prefer=prefer) if need_encoder else None
    emit("ffmpeg", path=choice.ffmpeg.path, version=choice.ffmpeg.version, capture=choice.method,
         encoder=encoder[0] if encoder else None)
    return Capture(choice, encoder, fps=fps)


class LiveGame:
    """The real game: FFmpeg captures plus scan-code key presses."""

    def __init__(self, cap, max_seconds: int = 930):
        from . import winio
        self.cap = cap
        self.winio = winio
        self.max_seconds = max_seconds

    def grab(self):
        if self.cap.method == "ddagrab":
            try:
                self.winio.focus()   # screen capture: the window must be on top
            except Exception:
                return None
        return self.cap.grab()

    def press(self, key: str) -> None:
        from .exporter import ExportError
        try:
            self.winio.press(key, gap_ms=150, verify_focus=True)
        except self.winio.FocusError as e:
            raise ExportError(f"{e}. Not sending keys; stopping.") from e

    def start_recording(self, path: str):
        return self.cap.start_recording(path, max_seconds=self.max_seconds)


def _watch_stdin(cancel, emit):
    def run():
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "cancel":
                cancel.request(now=False)
                emit("status", message="Stopping after the current replay...")
            elif cmd == "stop":
                cancel.request(now=True)
                emit("status", message="Stopping now...")
    threading.Thread(target=run, daemon=True).start()


def _watch_ctrl_c(cancel, emit):
    def handler(_sig, _frm):
        if cancel.soft.is_set():
            cancel.request(now=True)
            emit("status", message="Stopping now (finalising the current file)...")
        else:
            cancel.request(now=False)
            emit("status", message="Stopping after the current replay (Ctrl+C again to stop now)...")
    signal.signal(signal.SIGINT, handler)


def cmd_export(args, emit) -> int:
    from . import winio
    from .exporter import Cancel, Exporter, Options
    from .rows import RowReader, parse_since
    from .screens import Classifier, RefSet

    winio.require_windows()
    if not winio.find_window():
        raise RuntimeError("Guilty Gear -Strive- is not running (no game window found). Start the "
                           "game, leave it at the title screen and try again.")
    since = parse_since(args.since) if args.since else None
    refs = RefSet(args.refs_dir)
    classifier = Classifier(refs, lang=args.lang)
    profile = refs.profile(None if args.lang == "auto" else args.lang)
    order = args.date_order or profile.get("date_order") or "DMY"
    reader = None
    if not args.no_ocr:
        reader = RowReader(date_order=order)
        if not reader.available:
            if since:
                raise RuntimeError(f"--since needs the dates in the replay list, but {reader.error}")
            emit("warning", message=f"{reader.error}. Files are named by list position.")
            reader = None
    elif since:
        raise RuntimeError("--since needs OCR; drop --no-ocr")

    cap = _capture(args, need_encoder=not args.dry_run, emit=emit, fps=args.fps, prefer=args.encoder)
    game = LiveGame(cap, max_seconds=int(args.max_minutes * 60) + 30)
    out = os.path.abspath(args.out)
    opts = Options(out_dir=out, count=None if args.all else args.count, since=since, start=args.start,
                   dry_run=args.dry_run, skip_existing=not args.overwrite,
                   max_seconds=int(args.max_minutes * 60),
                   meta={"capture": cap.method, "encoder": cap.encoder[0] if cap.encoder else None,
                         "fps": args.fps})
    cancel = Cancel()
    if args.json_progress:
        _watch_stdin(cancel, emit)
    _watch_ctrl_c(cancel, emit)
    emit("start", out=out, count=opts.count, since=since.isoformat() if since else None,
         start=args.start, dry_run=args.dry_run)
    Exporter(game, classifier, reader, opts, emit=emit, cancel=cancel).run()
    return 0


def cmd_calibrate(args, emit) -> int:
    import cv2
    from . import winio
    from .rows import RowReader
    from .screens import Classifier, RefSet, normalize, user_refs_dir, full_res

    winio.require_windows()
    if not winio.find_window():
        raise RuntimeError("Start Guilty Gear -Strive- first.")
    cap = _capture(args, need_encoder=False, emit=emit)
    base = os.path.abspath(args.refs_dir) if args.refs_dir else str(user_refs_dir())
    dest = os.path.join(base, args.lang)
    os.makedirs(dest, exist_ok=True)
    # Language-independent checks only, so a new language is recognised.
    cls = Classifier(RefSet([]), lang="auto")
    cls.auto = False
    steps = [
        ("title", "the title screen (GUILTY GEAR -STRIVE- logo, 'press a key' prompt)", "title"),
        ("main", "the main menu with the first tab (online / network) selected", "main"),
        ("collection", "the main menu with the Collection tab selected (Replay highlighted in its list)", "main"),
        ("saved_list", "Replay room > the 'Saved replays' tab (third icon), first replay highlighted", "saved_list"),
        ("replay_menu", "the menu of a saved replay (press confirm on one; 'View replay' highlighted)", "replay_menu"),
        ("replay_end", "the menu at the end of a replay ('Back to the replay list' highlighted)", "replay_end"),
        ("server", "optional: the 'communicating with the server' dialog (Enter to skip)", "server"),
    ]
    print(f"Calibrating '{args.lang}' into {dest}. The tool takes screenshots only; it never "
          "presses keys. For each screen, open it in the game, then press Enter here.\n")
    saved_img = None
    for name, desc, expect in steps:
        while True:
            ans = input(f"Open {desc}, then press Enter (s = skip): ").strip().lower()
            if ans == "s":
                break
            img = cap.grab()
            if img is None:
                print("  capture failed, try again")
                continue
            s = cls.classify(img)
            ok = s.name == expect and (name != "collection" or s.tab == 6) and (name != "main" or s.tab == 0)
            if not ok:
                print(f"  this looks like: {s} (expected {expect})")
                if input("  save it anyway? [y/N] ").strip().lower() != "y":
                    continue
            cv2.imwrite(os.path.join(dest, f"{name}.jpg"), normalize(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  saved {name}.jpg")
            if name == "saved_list":
                saved_img = (img, s)
            break
    order = None
    if saved_img is not None and saved_img[1].row_y is not None:
        rr = RowReader()
        if rr.available:
            info = rr.read_row(full_res(saved_img[0]), saved_img[1].row_y)
            print(f"  first row reads: {info.raw!r}")
    ans = input("Date order of the replay list's 'match date' (DMY, MDY or YMD) [DMY]: ").strip().upper()
    order = ans if ans in ("DMY", "MDY", "YMD") else "DMY"
    with open(os.path.join(dest, "profile.json"), "w", encoding="utf-8") as f:
        json.dump({"date_order": order, "calibrated": dt.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"done. Use it with: python -m replanal.strive_export --lang {args.lang} ...")
    return 0


def cmd_probe(args, emit) -> int:
    import cv2
    from . import winio
    from .capture import pick_encoder
    from .screens import Classifier, RefSet
    winio.require_windows()
    hwnd = winio.find_window()
    emit("window", found=bool(hwnd), rect=winio.client_rect(hwnd) if hwnd else None)
    cap = _capture(args, need_encoder=True, emit=emit)
    img = cap.grab()
    if img is None:
        emit("error", message="capture came back empty")
        return 1
    if args.save:
        cv2.imwrite(args.save, img)
    s = Classifier(RefSet(args.refs_dir), lang=args.lang).classify(img)
    emit("screen", screen=s.name, detail=str(s), lang=s.lang, scores=s.scores, size=list(img.shape[:2]))
    return 0


def cmd_classify(args, emit) -> int:
    import cv2
    from .screens import Classifier, RefSet
    cls = Classifier(RefSet(args.refs_dir), lang=args.lang)
    for p in args.images:
        img = cv2.imread(p)
        s = cls.classify(img)
        print(f"{p}: {s}  lang={s.lang} scores={s.scores} {s.detail}")
    return 0


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    emit = Emitter(getattr(args, "json_progress", False))
    try:
        return {"export": cmd_export, "calibrate": cmd_calibrate, "probe": cmd_probe,
                "classify": cmd_classify}[args.cmd](args, emit)
    except Exception as e:
        from .winio import NotWindows
        emit("error", message=str(e))
        if not isinstance(e, (RuntimeError, NotWindows)) or os.environ.get("STRIVE_EXPORT_DEBUG"):
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
