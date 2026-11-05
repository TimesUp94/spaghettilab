"""Language layer (ROADMAP point 5), local part: match facts, LLM prompts, grounding check.

Never sends anything anywhere: the only provider that runs is "none" (writes the prompt). The Claude
provider is a stub that refuses to run until the user decides that match data may leave the machine
(and then still needs GGST_LANG_ALLOW_NETWORK=1 and enabled=True). No GPU, no models: it reads
predictions that already exist.

1. facts: the structured output of one match -> a compact, faithful "match facts" document (json) and a
   short markdown rendering. Inputs, all from the video at inference:
   - the opener chain's openers.json (recognize.py schema);
   - optionally the transcript predictions (video_transcript.py preds rows: side, start_frame (60 fps),
     category, char_move, normal, outcome, p_cat, p_move, score);
   - optionally the CV pipeline's frames parquet (HUD: HP, timer, round hearts), for rounds, winners,
     damage per opener and combo ends.
   Every fact carries video timestamps (m:ss.s); uncertain ones (low model probability, a round winner
   without a KO, hearts disagreeing) are marked. Only what the data supports is computed.
2. prompts: system prompt + facts (it / en), asking for a short summary and 3-5 coaching notes, each
   citing timestamps, forbidding claims that are not in the facts.
3. check: an LLM answer against the facts: every cited timestamp, move and character must exist in the
   facts; no frame data; percentages must be in the facts; every note must cite a timestamp.

    .venv/Scripts/python.exe tools/ggst_logger/model/lang_summary.py facts <dir with openers.json> [--lang it en]
    .venv/Scripts/python.exe tools/ggst_logger/model/lang_summary.py check <answer.md> <facts.json>
    .venv/Scripts/python.exe tools/ggst_logger/model/lang_summary.py run [<dir> ...]   # every DATA/lang/<stem>/
    .venv/Scripts/python.exe tools/ggst_logger/model/lang_summary.py verify [<dir> ...] # facts vs the game log (test only)

A <dir> holds openers.json and optionally transcript_preds.json and frames.parquet (lang_testdata.py makes
them for test-fold replays); outputs go next to them: facts.json, facts.md, prompt_<lang>.json / .txt.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "ggst-match-facts/1"
FPS60 = 60.0
MOVES_JSON = Path(__file__).resolve().parents[1] / "dataset" / "moves.json"

# uncertainty thresholds (fixed by hand, not tuned on any fold)
P_MOVE_MIN = 0.5        # opener / transcript move probability below this: "uncertain"
P_CAT_MIN = 0.6         # opener category probability
P_HIT_KIND_MIN = 0.6    # counter / punish / neither from the banner reader
TOL_S = 1.5             # a cited timestamp matches a fact within this many seconds

ATTACK_CATS = ("normal", "special", "super", "throw", "projectile")


# ---- small helpers ------------------------------------------------------------------------------

def ts(t: float) -> str:
    """Video time as m:ss.s (the citation format)."""
    t = max(0.0, float(t))
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def parse_ts(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def pct(x: float) -> int:
    return int(round(100 * x))


def other(side: str) -> str:
    return "P2" if side == "P1" else "P1"


_MOVES_CACHE: dict = {}


def catalog() -> dict:
    """Characters and per-character notation -> English move name, from dataset/moves.json."""
    if not _MOVES_CACHE:
        d = json.loads(MOVES_JSON.read_text(encoding="utf-8"))
        names = {}
        for ch, acts in d["moves"].items():
            nm = names.setdefault(ch, {})
            for v in acts.values():
                if v.get("notation") and v.get("name"):
                    nm.setdefault(v["notation"], v["name"])
        common = {v[0]: v[1] for v in d["common"].values() if v[0]}
        chars = sorted(set(d["characters"]["codes"].values()))
        _MOVES_CACHE.update(names=names, common=common, chars=chars)
    return _MOVES_CACHE


def move_name(char: str, move: str) -> str | None:
    c = catalog()
    return c["names"].get(char, {}).get(move) or c["common"].get(move)


# ---- HUD: rounds, damage, combo ends -------------------------------------------------------------

@dataclass
class Hud:
    t: list[float]
    hp: dict[str, list[float | None]]
    timer: list[int | None]
    hearts: dict[str, list[int | None]]

    @staticmethod
    def from_columns(c: dict) -> "Hud":
        t = [ms / 1000.0 for ms in c["timestamp_ms"]]
        return Hud(t=t, hp={"P1": list(c["p1_health_pct"]), "P2": list(c["p2_health_pct"])},
                   timer=list(c.get("timer_value") or [None] * len(t)),
                   hearts={"P1": list(c.get("p1_rounds_won") or [None] * len(t)),
                           "P2": list(c.get("p2_rounds_won") or [None] * len(t))})

    @staticmethod
    def load(path: Path) -> "Hud":
        import pyarrow.parquet as pq
        return Hud.from_columns(pq.read_table(path).to_pydict())

    def hp_at(self, side: str, t: float, before=True) -> float | None:
        """Last HP reading at or before t (before=True), else first at or after t."""
        import bisect
        i = bisect.bisect_right(self.t, t) - 1 if before else bisect.bisect_left(self.t, t)
        step = -1 if before else 1
        while 0 <= i < len(self.t):
            v = self.hp[side][i]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
            i += step
        return None


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


KO_HP = 0.001
FULL_HP = 0.97
CLOSE_KO_PCT = 3        # a KO won with <= 3 % of the bar left: a near double KO, the HUD may have it backwards


def _stays_zero(h: Hud, side: str, i: int, span: float = 1.0) -> bool:
    """A KO'd bar stays empty; a misread 0 comes back within a second."""
    t0 = h.t[i]
    for j in range(i, len(h.t)):
        if h.t[j] - t0 > span:
            break
        v = h.hp[side][j]
        if _ok(v) and v > KO_HP:
            return False
    return True


def detect_rounds(h: Hud) -> list[dict]:
    """Rounds from the cleaned HUD HP (the cleaner writes exactly 0 only for a KO'd side): a round
    starts when the clock reads 97-99 (or both bars are full with no clock reading yet), ends at the
    first KO, or (timeout / missed KO) where the bars refill without one: then the higher bar wins
    and the round is marked uncertain unless the clock had reached 0. Hearts are a cross-check only."""
    n = len(h.t)
    rounds, cur = [], None
    last_timer = None
    for i in range(n):
        a, b, tm = h.hp["P1"][i], h.hp["P2"][i], h.timer[i]
        if tm is not None:
            last_timer = (tm, h.t[i])
        if not (_ok(a) and _ok(b)):
            continue
        full = a >= FULL_HP and b >= FULL_HP
        clock_up = tm is None or tm >= 95
        if cur is None:
            # the clock shows 99 when a round starts; the bars can be misread for the first second
            if (full and clock_up) or (tm is not None and tm >= 97):
                cur = dict(start_s=h.t[i], last_i=i, min_tm=tm if tm is not None else 99)
            continue
        if tm is not None:
            cur["min_tm"] = min(cur["min_tm"], tm)
        ko = [s for s, v in (("P1", a), ("P2", b)) if v <= KO_HP and _stays_zero(h, s, i)]
        if ko and h.t[i] - cur["start_s"] > 2.0:
            if len(ko) == 2:
                rounds.append(dict(cur, end_s=h.t[i], end="double KO", winner=None, final={"P1": a, "P2": b}))
            else:
                rounds.append(dict(cur, end_s=h.t[i], end="KO", winner=other(ko[0]), final={"P1": a, "P2": b}))
            cur = None
            continue
        prev = cur["last_i"]
        pa, pb = h.hp["P1"][prev], h.hp["P2"][prev]
        if full and min(pa, pb) < 0.9 and clock_up and h.t[i] - cur["start_s"] > 10:
            # the bars refilled without a KO: timeout (clock at 0) or a KO the HUD missed
            end = "timeout" if cur["min_tm"] <= 1 else "unclear"
            w = "P1" if pa > pb else "P2" if pb > pa else None
            rounds.append(dict(cur, end_s=h.t[prev], end=end, winner=w, final={"P1": pa, "P2": pb}))
            cur = dict(start_s=h.t[i], last_i=i, min_tm=tm if tm is not None else 99)
            continue
        cur["last_i"] = i
    if cur is not None and h.t[cur["last_i"]] - cur["start_s"] > 10:
        prev = cur["last_i"]
        pa, pb = h.hp["P1"][prev], h.hp["P2"][prev]
        end = "timeout" if cur["min_tm"] <= 1 else "unclear (video ends)"
        w = "P1" if pa > pb else "P2" if pb > pa else None
        rounds.append(dict(cur, end_s=h.t[prev], end=end, winner=w, final={"P1": pa, "P2": pb}))
    out = []
    for k, r in enumerate(rounds):
        # hearts just after the round (next round's first 15 s, or the 6 s after the KO for the last)
        t0 = r["end_s"] + 1.0
        t1 = (rounds[k + 1]["start_s"] + 15) if k + 1 < len(rounds) else r["end_s"] + 6
        votes = collections.Counter((h.hearts["P1"][i], h.hearts["P2"][i]) for i in range(n)
                                    if t0 <= h.t[i] <= t1 and h.hearts["P1"][i] is not None and h.hearts["P2"][i] is not None)
        hearts = list(votes.most_common(1)[0][0]) if votes else None
        out.append(dict(n=k + 1, start_s=round(r["start_s"], 2), end_s=round(r["end_s"], 2), end=r["end"], winner=r["winner"],
                        final_hp={s: pct(v) for s, v in r["final"].items()}, hearts_after=hearts))
    # hearts agreement: the winner's count goes up by one from the previous round's
    prev = [0, 0]
    for r in out:
        hv = r["hearts_after"]
        if hv is None or r["winner"] is None:
            r["hearts_agree"] = None
        else:
            # agree: the winner's heart count went up; disagree: the loser's did; else unknown (the
            # hearts after the last round are often not shown, and the reader misses lost hearts)
            j = 0 if r["winner"] == "P1" else 1
            if hv[1 - j] > prev[1 - j]:
                r["hearts_agree"] = False
            elif hv[j] > prev[j]:
                r["hearts_agree"] = True
            else:
                r["hearts_agree"] = None
            prev = [max(a, b) for a, b in zip(hv, prev)]
        margin = abs(r["final_hp"]["P1"] - r["final_hp"]["P2"])
        r["uncertain"] = (r["winner"] is None or r["end"] not in ("KO", "timeout") or r["hearts_agree"] is False
                          or (r["end"] == "timeout" and margin < 9)
                          or (r["end"] == "KO" and r["final_hp"][r["winner"]] <= CLOSE_KO_PCT))
    # first to 2: a round after a side has 2 wins, or a third win, means a misread round somewhere
    wins = collections.Counter()
    for r in out:
        if max(wins.values(), default=0) >= 2:
            r["uncertain"] = True
            for q in out:
                q["uncertain"] = q["uncertain"] or q["end"] != "KO" or q["final_hp"].get(q["winner"] or "P1", 0) <= 15
        if r["winner"]:
            wins[r["winner"]] += 1
    return out


def round_of(rounds: list[dict], t: float) -> int | None:
    for r in rounds:
        if r["start_s"] - 1.0 <= t <= r["end_s"] + 0.5:
            return r["n"]
    return None


def combo_outcome(h: Hud, side: str, t: float, t_stop: float) -> tuple[float, float, bool]:
    """Damage (fraction of the bar) the defender lost from just before t until t_stop, the time of the
    last drop (combo end) and whether the defender reached 0 (KO)."""
    d = other(side)
    before = h.hp_at(d, t - 0.1)
    if before is None:
        return 0.0, t, False
    last, end_t, ko = before, t, False
    for i in range(len(h.t)):
        if h.t[i] < t - 0.1 or h.t[i] > t_stop:
            continue
        v = h.hp[d][i]
        if not _ok(v):
            continue
        if v < last - 1e-4:
            last, end_t = v, h.t[i]
        if v <= KO_HP:
            ko = True
            break
    return max(0.0, before - last), end_t, ko


# ---- facts -----------------------------------------------------------------------------------------

class Index:
    """What the facts document states, for the grounding checker: times (with what happened there),
    moves per character, percentages and counts."""

    def __init__(self):
        self.times: list[dict] = []
        self.moves: dict[str, set] = collections.defaultdict(set)
        self.pcts: set = set()
        self.counts: set = set()

    def time(self, t: float, **what) -> str:
        s = ts(t)
        self.times.append(dict(t=s, t_s=round(t, 2), **what))
        return s

    def move(self, char: str, move: str):
        if char and move:
            self.moves[char].add(move)
            nm = move_name(char, move)
            if nm:
                self.moves[char].add(nm)

    def to_json(self) -> dict:
        return dict(times=self.times, moves={k: sorted(v) for k, v in sorted(self.moves.items())},
                    percentages=sorted(self.pcts), counts=sorted(self.counts))


def _counter(items, top=None, ix: "Index | None" = None):
    """[{what, n, examples: up to 3 times}] by count. An item is (key, time) or (key, seconds, meta):
    only the examples kept are entered in the index (meta = Index.time fields + register: [(char, move)]),
    so the index holds exactly what the document shows."""
    by = collections.OrderedDict()
    for it in items:
        by.setdefault(it[0], []).append((it[1], it[2] if len(it) > 2 else None))
    rows = sorted(by.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:top]
    out = []
    for k, v in rows:
        ex = []
        for t, meta in v[:3]:
            if isinstance(t, str):
                ex.append(t)
                continue
            meta = dict(meta)
            reg = meta.pop("register", [])
            ex.append(ix.time(t, **meta))
            for c, m in reg:
                ix.move(c, m)
        out.append(dict(what=k, n=len(v), examples=ex))
    return out


def match_facts(openers_doc: dict, transcript: list[dict] | None = None, hud: Hud | None = None,
                match_id: str | None = None) -> dict:
    """The facts document of one match (see the module doc). Pure function of its inputs."""
    ix = Index()
    chars = dict(openers_doc["characters"])
    caveats = ["models", "video_time", "damage", "openers", "no_frame_data"]
    rounds = detect_rounds(hud) if hud is not None else []
    for r in rounds:
        r["t_start"] = ix.time(r["start_s"], kind="round_start", round=r["n"])
        r["t_end"] = ix.time(r["end_s"], kind="round_end", round=r["n"], winner=r["winner"])
    result = None
    if rounds:
        wins = collections.Counter(r["winner"] for r in rounds if r["winner"])
        w = max(("P1", "P2"), key=lambda s: wins[s]) if wins and wins["P1"] != wins["P2"] else None
        result = dict(winner=w, score=f"{wins['P1']}-{wins['P2']}", rounds=len(rounds),
                      uncertain=any(r["uncertain"] for r in rounds) or w is None or max(wins.values()) != 2)
        ix.counts.update([wins["P1"], wins["P2"], len(rounds)])
    else:
        caveats.append("no_hud")

    # openers
    ops = sorted(openers_doc["openers"], key=lambda o: o["time_s"])
    end_t = rounds[-1]["end_s"] + 5 if rounds else (ops[-1]["time_s"] + 10 if ops else 0)
    facts_ops = []
    for k, o in enumerate(ops):
        t = o["time_s"]
        side, char = o["attacker"], o["attacker_char"]
        nxt = ops[k + 1]["time_s"] - 0.1 if k + 1 < len(ops) else end_t
        rn = round_of(rounds, t) if rounds else None
        r_end = next((r["end_s"] for r in rounds if r["n"] == rn), end_t) if rn else end_t
        why = []
        if o.get("move_p", 1) < P_MOVE_MIN:
            why.append(f"move p={o['move_p']:.2f}")
        if o["move"] == "(rare)":
            why.append("rare move, not named by the model")
        if o.get("category_p", 1) < P_CAT_MIN:
            why.append(f"category p={o['category_p']:.2f}")
        f = dict(id=f"O{k + 1}", t_s=round(t, 2), round=rn, attacker=side, char=char, category=o["category"],
                 move=o["move"], move_name=move_name(char, o["move"]),
                 hit_kind=({"none": "neither"}.get(o.get("hit_kind"), o.get("hit_kind")) if o.get("hit_kind_p", 1) >= P_HIT_KIND_MIN else "unclear"),
                 uncertain=bool(why))
        if why:
            f["why_uncertain"] = ", ".join(why)
            f["alternatives"] = [m for m, p in o.get("move_top3", [])[1:] if p >= 0.15]
            for m in f["alternatives"]:
                ix.move(char, m)
        if hud is not None:
            dmg, ce, ko = combo_outcome(hud, side, t, min(nxt, t + 12.0, r_end + 0.5))
            f.update(damage_pct=pct(dmg), combo_end_s=round(max(ce, t), 2), ko=ko)
            ix.pcts.add(f["damage_pct"])
        f["t"] = ix.time(t, kind="opener", side=side, char=char, move=o["move"], id=f["id"])
        if o["move"] != "(rare)":
            ix.move(char, o["move"])
        facts_ops.append(f)

    players = {}
    for side in ("P1", "P2"):
        mine = [f for f in facts_ops if f["attacker"] == side]
        p = dict(char=chars[side], openers=len(mine), openers_uncertain=sum(f["uncertain"] for f in mine))
        ix.counts.update([p["openers"], p["openers_uncertain"]])
        p["by_category"] = dict(collections.Counter(f["category"] for f in mine).most_common())
        p["by_hit_kind"] = dict(collections.Counter(f["hit_kind"] for f in mine).most_common())
        ix.counts.update(p["by_category"].values())
        ix.counts.update(p["by_hit_kind"].values())
        p["top_opening_moves"] = _counter([(f["move"], f["t"]) for f in mine if not f["uncertain"]], top=5)
        ix.counts.update(r["n"] for r in p["top_opening_moves"])
        if hud is not None:
            p["damage_dealt_pct"] = sum(f["damage_pct"] for f in mine)
            ix.pcts.add(p["damage_dealt_pct"])
            for r in p["top_opening_moves"]:
                d = [f["damage_pct"] for f in mine if f["move"] == r["what"] and not f["uncertain"]]
                r["damage_pct_total"], r["damage_pct_avg"] = sum(d), (round(sum(d) / len(d)) if d else 0)
                ix.pcts.update([r["damage_pct_total"], r["damage_pct_avg"]])
            best = sorted(mine, key=lambda f: -f["damage_pct"])[:3]
            p["biggest_combos"] = [dict(id=f["id"], t=f["t"], move=f["move"], damage_pct=f["damage_pct"], ko=f["ko"],
                                        hit_kind=f["hit_kind"]) for f in best if f["damage_pct"] > 0]
            if rounds:
                p["damage_by_round_pct"] = {str(r["n"]): sum(f["damage_pct"] for f in mine if f["round"] == r["n"]) for r in rounds}
                ix.pcts.update(p["damage_by_round_pct"].values())
        players[side] = p

    doc = dict(schema=SCHEMA, match_id=match_id, characters=chars,
               duration_s=round(hud.t[-1], 1) if hud is not None and hud.t else None,
               sources=dict(openers=openers_doc.get("source", "recognize.py"), transcript=transcript is not None,
                            hud=hud is not None),
               result=result, rounds=rounds, players=players, openers=facts_ops)
    segs = openers_doc.get("character_segments") or []
    if len(segs) > 1:      # a set whose players change characters between games
        doc["character_segments"] = [dict(t=ix.time(g["start_s"], kind="characters", P1=g["P1"]["char"], P2=g["P2"]["char"]), P1=g["P1"]["char"],
                                          P2=g["P2"]["char"]) for g in segs]
    if transcript is not None:
        doc["transcript"] = transcript_facts(transcript, facts_ops, chars, rounds, ix)
        caveats += ["transcript", "habits"]
    doc["caveats"] = caveats
    doc["index"] = ix.to_json()
    return doc


def _tr_events(rows: list[dict]) -> list[dict]:
    ev = []
    for r in rows:
        mv = r["char_move"].split(":", 1)[1] if r.get("char_move") else None
        ev.append(dict(t=r["start_frame"] / FPS60, side=r["side"], char=r.get("char"), cat=r["category"], move=mv,
                       outcome=r["outcome"], sure=(r.get("p_move") or 0) >= P_MOVE_MIN and mv != "(rare)"))
    return sorted(ev, key=lambda e: e["t"])


def transcript_facts(rows: list[dict], ops: list[dict], chars: dict, rounds: list[dict], ix: Index) -> dict:
    ev = _tr_events(rows)
    if rounds:   # only inside rounds (+ the KO margin)
        ev = [e for e in ev if any(r["start_s"] - 1 <= e["t"] <= r["end_s"] + 1 for r in rounds)]
    out = {}
    for side in ("P1", "P2"):
        char = chars[side]
        mine = [e for e in ev if e["side"] == side]
        att = [e for e in mine if e["cat"] in ATTACK_CATS]
        oc = collections.Counter(e["outcome"] for e in att)
        n = len(att)
        rates = {}
        if n:
            rates = dict(hit=pct((oc["hit"] + oc["counter"]) / n), counter=pct(oc["counter"] / n), blocked=pct(oc["block"] / n),
                         whiff=pct(oc["whiff"] / n), interrupted=pct(oc["interrupted"] / n))
            ix.pcts.update(rates.values())
        def tag(e):
            return e["move"] if e["sure"] else None

        used = _counter([(tag(e), e["t"], dict(kind="move", side=side, char=char, move=e["move"], outcome=e["outcome"],
                                                register=[(char, e["move"])]))
                         for e in att if tag(e)], top=6, ix=ix)
        p = dict(char=char, moves_seen=len(mine), attacks_seen=n, attack_outcomes_pct=rates,
                 most_used_attacks=used,
                 throws=dict(collections.Counter(e["outcome"] for e in mine if e["cat"] == "throw")),
                 bursts=sum(1 for e in mine if e["move"] == "Burst"),
                 roman_cancels=sum(1 for e in mine if e["move"] == "RC"))
        ix.counts.update([p["moves_seen"], n, p["bursts"], p["roman_cancels"], *p["throws"].values(), *(r["n"] for r in used)])
        # what counter-hit this player, and what they were doing
        opp = other(side)
        ch = [e for e in ev if e["side"] == opp and e["outcome"] == "counter" and e["sure"]]
        ch_items = []
        for e in ch:
            doing = [x for x in mine if e["t"] - 0.6 <= x["t"] <= e["t"] and x["sure"]]
            what = f"{e['move']} (while doing {doing[-1]['move']})" if doing else e["move"]
            reg = [(chars[opp], e["move"])] + ([(char, doing[-1]["move"])] if doing else [])
            ch_items.append((what, e["t"], dict(kind="move", side=opp, char=chars[opp], move=e["move"], outcome="counter",
                                                register=reg)))
        p["counter_hit_by"] = _counter(ch_items, top=4, ix=ix)
        ix.counts.update(r["n"] for r in p["counter_hit_by"])
        out[side] = p
    habits = _habits(ev, ops, chars, rounds, ix)
    for side in ("P1", "P2"):
        out[side]["habits"] = habits[side]
    return out


def _first(ev, side, t0, t1):
    for e in ev:
        if e["side"] == side and t0 < e["t"] <= t1:
            return e
        if e["t"] > t1:
            break
    return None


def _habits(ev: list[dict], ops: list[dict], chars: dict, rounds: list[dict], ix: Index) -> dict:
    res = {s: dict(after_own_combo=[], after_getting_comboed=[], after_blocking=[], pressure_strings=[]) for s in ("P1", "P2")}
    acc = {s: collections.defaultdict(list) for s in ("P1", "P2")}

    def note(side, kind, e):
        if e is None or not e["sure"]:
            return
        lab = f"{e['move']} ({e['outcome']})"
        acc[side][kind].append((lab, e["t"], dict(kind="move", side=e["side"], char=chars[e["side"]], move=e["move"],
                                                  outcome=e["outcome"], register=[(chars[e["side"]], e["move"])])))

    # after a combo ends (not KO): the attacker's next move within 3 s, the defender's within 2.5 s
    for f in ops:
        if f.get("ko") or "combo_end_s" not in f or f.get("damage_pct", 0) <= 0:
            continue
        ce = f["combo_end_s"]
        a, d = f["attacker"], other(f["attacker"])
        note(a, "after_own_combo", _first(ev, a, ce + 0.2, ce + 3.0))
        note(d, "after_getting_comboed", _first(ev, d, ce + 0.2, ce + 2.5))
    # blockstrings: consecutive blocked attacks of one side, < 1.0 s apart, no move of the other side between
    for a in ("P1", "P2"):
        d = other(a)
        strings, cur = [], []
        for e in ev:
            if e["side"] == d and cur:
                if e["t"] - cur[-1]["t"] > 0.05:
                    strings.append(cur)
                    cur = []
                continue
            if e["side"] != a:
                continue
            if e["outcome"] == "block" and e["cat"] in ATTACK_CATS:
                if cur and e["t"] - cur[-1]["t"] > 1.0:
                    strings.append(cur)
                    cur = []
                cur.append(e)
            elif cur:
                strings.append(cur)
                cur = []
        if cur:
            strings.append(cur)
        for s in strings:
            last = s[-1]
            note(d, "after_blocking", _first(ev, d, last["t"], last["t"] + 1.0))
            if len(s) >= 2 and all(x["sure"] for x in s):
                seq = " > ".join(x["move"] for x in s[:4])
                mv = [x["move"] for x in s[:4]]
                acc[a]["pressure_strings"].append((seq, s[0]["t"], dict(kind="string", side=a, char=chars[a], move=mv[0],
                                                                        moves=mv, register=[(chars[a], m) for m in mv])))
    for side in ("P1", "P2"):
        for kind in list(res[side]):
            res[side][kind] = _counter(acc[side][kind], top=4, ix=ix)
            ix.counts.update(r["n"] for r in res[side][kind])
            ix.counts.add(len(acc[side][kind]))
            res[side][kind + "_total"] = len(acc[side][kind])
    return res


CAVEATS = {
    "models": dict(
        en="All facts come from video models, not from the game: moves can be misread; uncertain ones are marked.",
        it="Tutti i fatti vengono da modelli video, non dal gioco: le mosse possono essere lette male; quelle incerte sono segnate."),
    "video_time": dict(en="Timestamps are video time (m:ss.s), not the round clock.",
                       it="I timestamp sono il tempo del video (m:ss.s), non il timer del round."),
    "damage": dict(
        en="Damage is the share of the defender's health bar lost until the next opener or the end of the combo, read from the HUD; totals add up these shares.",
        it="Il danno è la quota della barra della vita persa dal difensore fino all'apertura successiva o alla fine della combo, letta dall'HUD; i totali sommano queste quote."),
    "openers": dict(
        en="Combo openers: the first hit of each combo (bursts excluded). Counter / punish is read from the game's banner.",
        it="Aperture di combo: il primo colpo di ogni combo (burst esclusi). Counter / punish viene letto dal banner del gioco."),
    "no_frame_data": dict(en="There is no frame data in these facts.", it="In questi fatti non ci sono frame data."),
    "no_hud": dict(en="No HUD readings: rounds, winners and damage are unknown.",
                   it="Nessuna lettura dell'HUD: round, vincitori e danni sono sconosciuti."),
    "transcript": dict(
        en="The transcript (every move start, not only openers) comes from a video model: it finds about 9 in 10 moves that hit or are blocked but only about 6 in 10 whiffs, so whiff shares are too low; repeated moves under 8 frames apart are often merged.",
        it="La trascrizione (ogni mossa, non solo le aperture) viene da un modello video: trova circa 9 mosse su 10 che colpiscono o vengono parate ma solo circa 6 whiff su 10, quindi le quote di whiff sono troppo basse; mosse ripetute a meno di 8 frame di distanza vengono spesso unite."),
    "habits": dict(en="Habits are counted on transcript moves with a confident move label only; n is the number of situations.",
                   it="Le abitudini contano solo mosse della trascrizione con etichetta sicura; n è il numero di situazioni."),
}


# ---- markdown ----------------------------------------------------------------------------------------

L = {
    "en": dict(char_changes="Characters by part of the video", title="Match facts", n_openers="Combo openers", attacks="attacks", throws="throws", result="Result", winner="winner", rounds="Rounds", round="Round", ko="KO",
               uncertain="uncertain", openers="Combo openers (timeline)", players="Players", dmg="dmg",
               top="Top opening moves", biggest="Biggest combos", mix="Opener mix", hk="Counter / punish",
               tr="Transcript (every move)", outc="Attack outcomes", used="Most used attacks",
               chby="Counter-hit by", hab="Habits", aoc="After own combo (next move)",
               agc="After getting comboed (next move)", ab="After blocking (next move)", ps="Pressure strings (blocked)",
               cav="Caveats", none="none", hearts_no="hearts disagree", n="n", ex="e.g."),
    "it": dict(char_changes="Personaggi per parte del video", title="Fatti della partita", n_openers="Aperture di combo", attacks="attacchi", throws="prese", result="Risultato", winner="vince", rounds="Round", round="Round", ko="KO",
               uncertain="incerto", openers="Aperture di combo (cronologia)", players="Giocatori", dmg="danno",
               top="Mosse di apertura principali", biggest="Combo più dannose", mix="Tipi di apertura",
               hk="Counter / punish", tr="Trascrizione (tutte le mosse)", outc="Esito degli attacchi",
               used="Attacchi più usati", chby="Preso counter da", hab="Abitudini",
               aoc="Dopo una propria combo (mossa successiva)", agc="Dopo aver subito una combo (mossa successiva)",
               ab="Dopo aver parato (mossa successiva)", ps="Stringhe di pressione (parate)", cav="Avvertenze",
               none="nessuno", hearts_no="i cuori non concordano", n="n", ex="es."),
}


def _mv(char, m):
    nm = move_name(char, m)
    return f"{m} ({nm})" if nm and nm != m else m


def render_md(doc: dict, lang: str = "en") -> str:
    """Markdown rendering of a facts document. It states nothing that is not in the document."""
    t = L[lang]
    ch = doc["characters"]
    out = [f"# {t['title']}: P1 {ch['P1']} vs P2 {ch['P2']}" + (f" ({doc['match_id']})" if doc.get("match_id") else "")]
    if doc.get("character_segments"):
        out.append(f"\n**{t['char_changes']}:** " + "; ".join(f"[{g['t']}] P1 {g['P1']} vs P2 {g['P2']}" for g in doc["character_segments"]))
    if doc.get("result"):
        r = doc["result"]
        w = f"{r['winner']} ({ch[r['winner']]})" if r["winner"] else "?"
        out.append(f"\n**{t['result']}:** {t['winner']} {w}, {r['score']}" + (f" ({t['uncertain']})" if r["uncertain"] else ""))
    if doc["rounds"]:
        out.append(f"\n## {t['rounds']}")
        for r in doc["rounds"]:
            flag = []
            if r["uncertain"]:
                flag.append(t["uncertain"])
            if r.get("hearts_agree") is False:
                flag.append(t["hearts_no"])
            w = f"{r['winner']} ({ch[r['winner']]})" if r["winner"] else "?"
            out.append(f"- {t['round']} {r['n']} [{r['t_start']}–{r['t_end']}]: {t['winner']} {w} ({r['end']}), "
                       f"HP P1 {r['final_hp']['P1']}% / P2 {r['final_hp']['P2']}%" + (f" — {', '.join(flag)}" if flag else ""))
    out.append(f"\n## {t['openers']}")
    for f in doc["openers"]:
        bits = [f"[{f['t']}] {f['id']} {f['attacker']} {f['char']}: {_mv(f['char'], f['move'])}, {f['category']}"]
        if f["hit_kind"] != "neither":
            bits.append(f["hit_kind"])
        if "damage_pct" in f:
            bits.append(f"{t['dmg']} {f['damage_pct']}%" + (f" {t['ko']}" if f["ko"] else ""))
        if f["uncertain"]:
            alt = f" / {', '.join(f['alternatives'])}?" if f.get("alternatives") else ""
            bits.append(f"{t['uncertain']}{alt}")
        out.append("- " + ", ".join(bits))
    out.append(f"\n## {t['players']}")
    for side in ("P1", "P2"):
        p = doc["players"][side]
        out.append(f"\n### {side} {p['char']}")
        line = f"- {t['n_openers']}: {p['openers']}" + (f" ({p['openers_uncertain']} {t['uncertain']})" if p["openers_uncertain"] else "")
        if "damage_dealt_pct" in p:
            line += f"; {t['dmg']} {p['damage_dealt_pct']}%"
            if p.get("damage_by_round_pct"):
                line += " (" + ", ".join(f"R{k} {v}%" for k, v in p["damage_by_round_pct"].items()) + ")"
        out.append(line)
        out.append(f"- {t['mix']}: " + (", ".join(f"{k} {v}" for k, v in p["by_category"].items()) or t["none"]))
        out.append(f"- {t['hk']}: " + (", ".join(f"{k} {v}" for k, v in p["by_hit_kind"].items()) or t["none"]))
        if p["top_opening_moves"]:
            out.append(f"- {t['top']}: " + "; ".join(
                f"{_mv(p['char'], r['what'])} ×{r['n']}" + (f" ({t['dmg']} {r['damage_pct_total']}%)" if "damage_pct_total" in r else "")
                + f" [{', '.join(r['examples'])}]" for r in p["top_opening_moves"]))
        if p.get("biggest_combos"):
            out.append(f"- {t['biggest']}: " + "; ".join(
                f"{b['id']} [{b['t']}] {b['move']} {b['damage_pct']}%" + (f" {t['ko']}" if b["ko"] else "") for b in p["biggest_combos"]))
        tr = doc.get("transcript", {}).get(side)
        if tr:
            o = tr["attack_outcomes_pct"]
            out.append(f"- {t['tr']}: {tr['attacks_seen']} {t['attacks']}; {t['outc']}: " +
                       (", ".join(f"{k} {v}%" for k, v in o.items()) if o else t["none"]))
            if tr["most_used_attacks"]:
                out.append(f"- {t['used']}: " + "; ".join(f"{_mv(p['char'], r['what'])} ×{r['n']} [{', '.join(r['examples'])}]"
                                                         for r in tr["most_used_attacks"]))
            if tr["throws"] or tr["bursts"] or tr["roman_cancels"]:
                out.append(f"- {t['throws']} {sum(tr['throws'].values())} ({', '.join(f'{k} {v}' for k, v in tr['throws'].items()) or '-'}), "
                           f"burst {tr['bursts']}, RC {tr['roman_cancels']}")
            if tr["counter_hit_by"]:
                out.append(f"- {t['chby']}: " + "; ".join(f"{r['what']} ×{r['n']} [{', '.join(r['examples'])}]" for r in tr["counter_hit_by"]))
            h = tr["habits"]
            for key, lab in (("after_own_combo", "aoc"), ("after_getting_comboed", "agc"), ("after_blocking", "ab"), ("pressure_strings", "ps")):
                if h[key]:
                    out.append(f"- {t[lab]} ({t['n']}={h[key + '_total']}): " + "; ".join(
                        f"{r['what']} ×{r['n']} [{', '.join(r['examples'])}]" for r in h[key]))
    out.append(f"\n## {t['cav']}")
    out += [f"- {CAVEATS[c][lang]}" for c in doc["caveats"]]
    return "\n".join(out) + "\n"


# ---- prompts ---------------------------------------------------------------------------------------

SYSTEM = {
    "en": """You are a Guilty Gear -Strive- coach writing for the player who just played this match.
You get a "match facts" document produced by video analysis models. It is the only source of truth.

Rules:
- Use only what the facts state. Do not add frame data, damage values, move properties, matchup knowledge or anything else from memory.
- Name moves exactly as they appear in the facts (notation, optionally the name given in brackets). Never mention a move or a character that is not in the facts.
- Every coaching note cites at least one timestamp from the facts in square brackets, e.g. [1:23.4], so the player can jump to it in the video. Cite only timestamps that appear in the facts.
- Facts marked uncertain may be wrong: either leave them out or say they are uncertain.
- Numbers only as written in the facts. Never compute a share or a percentage yourself: write a % only if that exact figure is in the facts, otherwise give the counts ("8 of 14").
- Whiffs are under-counted by the transcript; do not draw conclusions from a low whiff share.
- Be concrete and brief. No generic advice that the facts do not support.""",
    "it": """Sei un coach di Guilty Gear -Strive- e scrivi per il giocatore che ha appena giocato questa partita.
Ricevi un documento di "fatti della partita" prodotto da modelli di analisi video. È l'unica fonte di verità.

Regole:
- Usa solo quello che dicono i fatti. Non aggiungere frame data, valori di danno, proprietà delle mosse, conoscenza del matchup o altro a memoria.
- Chiama le mosse esattamente come compaiono nei fatti (notazione, eventualmente il nome tra parentesi). Non citare mai mosse o personaggi che non sono nei fatti.
- Ogni nota di coaching cita almeno un timestamp dei fatti tra parentesi quadre, es. [1:23.4], così il giocatore può saltare a quel punto del video. Cita solo timestamp presenti nei fatti.
- I fatti segnati come incerti possono essere sbagliati: tralasciali o dì che sono incerti.
- Numeri solo come scritti nei fatti. Non calcolare mai quote o percentuali: scrivi una % solo se quella cifra esatta è nei fatti, altrimenti dai i conteggi ("8 su 14").
- La trascrizione conta meno whiff di quelli reali; non trarre conclusioni da una quota di whiff bassa.
- Sii concreto e breve. Niente consigli generici che i fatti non sostengono.""",
}

TASK = {
    "en": """Write in English, in this format:

## Summary
Two to four sentences: who won and how (rounds, the openers and combos that decided it), with timestamps.

## Coaching notes
3 to 5 numbered notes for {who}, as a plain numbered list (1. 2. 3.) with each note a single paragraph and no sub-bullets. Each note: what happened (with [timestamps]), why it matters, one concrete thing to try. If the facts do not support 3 notes, write fewer and say so.""",
    "it": """Scrivi in italiano, in questo formato:

## Riepilogo
Da due a quattro frasi: chi ha vinto e come (round, aperture e combo decisive), con i timestamp.

## Note di coaching
Da 3 a 5 note numerate per {who}, come semplice elenco numerato (1. 2. 3.) con ogni nota in un solo paragrafo e senza sotto-elenchi. Ogni nota: cosa è successo (con [timestamp]), perché conta, una cosa concreta da provare. Se i fatti non bastano per 3 note, scrivine meno e dillo.""",
}


def build_prompt(doc: dict, lang: str = "en", player: str | None = None) -> dict:
    """{system, user} for a chat LLM. player: 'P1' / 'P2' to coach one side, else both."""
    ch = doc["characters"]
    if player:
        who = f"{player} ({ch[player]})"
    else:
        who = "both players" if lang == "en" else "entrambi i giocatori"
    facts = render_md(doc, lang)
    lead = "Match facts:" if lang == "en" else "Fatti della partita:"
    return dict(lang=lang, system=SYSTEM[lang], user=f"{lead}\n\n{facts}\n---\n{TASK[lang].format(who=who)}")


REPAIR = {
    "en": "A draft of this answer failed the fact check. Rewrite it, keeping what is right and removing or correcting exactly these problems. Reply with the full corrected answer only.\n\nDraft:\n{draft}\n\nProblems:\n{problems}",
    "it": "Una bozza di questa risposta non ha superato il controllo dei fatti. Riscrivila, tenendo ciò che è giusto e togliendo o correggendo esattamente questi problemi. Rispondi solo con la risposta completa corretta.\n\nBozza:\n{draft}\n\nProblemi:\n{problems}",
}


def repair_prompt(prompt: dict, draft: str, errors: list[dict]) -> dict:
    """The same prompt plus a failed draft and what the check found: one fresh call, no conversation state."""
    problems = "\n".join(dict.fromkeys(f"- {e['kind']}: {e['text']}" for e in errors))
    extra = REPAIR[prompt["lang"]].format(draft=draft, problems=problems)
    return dict(prompt, user=f"{prompt['user']}\n\n---\n{extra}")


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer (no network): ~3.3 characters per token for this mix of
    prose, notation and numbers. The real count needs the provider's count_tokens endpoint."""
    return int(math.ceil(len(text) / 3.3))


# ---- providers -------------------------------------------------------------------------------------

class Provider:
    name = "base"

    def complete(self, prompt: dict, out_dir: Path | None = None) -> str | None:
        raise NotImplementedError


class NoneProvider(Provider):
    """Writes the prompt (json + plain text) and returns None. Nothing leaves the machine."""
    name = "none"

    def complete(self, prompt: dict, out_dir: Path | None = None) -> str | None:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            lang = prompt["lang"]
            (out_dir / f"prompt_{lang}.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=1), encoding="utf-8")
            (out_dir / f"prompt_{lang}.txt").write_text(f"[system]\n{prompt['system']}\n\n[user]\n{prompt['user']}", encoding="utf-8")
        return None


@dataclass
class ClaudeProvider(Provider):
    """Claude Opus 5.5 over the facts document (decided 2026-11-04: the facts may go to Anthropic; only the
    facts, no video, frames or player names). It refuses unless constructed with enabled=True AND
    GGST_LANG_ALLOW_NETWORK=1 is set, so nothing leaves a machine that wasn't configured for it.
    Thinking is always on with this model (effort is the control); max_tokens covers thinking + answer."""
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000
    effort: str = "medium"
    fallbacks: str | None = "default"     # server-side refusal fallback
    enabled: bool = False
    name: str = field(default="claude", init=False)

    def complete(self, prompt: dict, out_dir: Path | None = None) -> str | None:
        if not (self.enabled and os.environ.get("GGST_LANG_ALLOW_NETWORK") == "1"):
            raise RuntimeError("ClaudeProvider is disabled: no decision yet on sending match data to an LLM provider "
                               "(construct with enabled=True and set GGST_LANG_ALLOW_NETWORK=1 once decided)")
        import anthropic  # optional: pip install anthropic (the server extra has it)
        client = anthropic.Anthropic()
        # the system prompt (~400 tokens) is below the minimum cacheable prefix: no cache_control
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=prompt["system"],
                  messages=[{"role": "user", "content": prompt["user"]}],
                  output_config={"effort": self.effort})
        if self.fallbacks:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks, **kw)
        else:
            resp = client.messages.create(**kw)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("the answer hit max_tokens")
        self.last_usage = dict(model=getattr(resp, "model", self.model), input_tokens=resp.usage.input_tokens,
                               output_tokens=resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if b.type == "text")


PROVIDERS = {"none": NoneProvider, "claude": ClaudeProvider}


def provider(name: str = "none", **kw) -> Provider:
    return PROVIDERS[name](**kw)


# ---- grounding check -------------------------------------------------------------------------------

TS_RE = re.compile(r"(?<![\d:])(\d{1,2}:\d{2}(?:\.\d+)?)(?![\d:])")
# a whole move: charge ([4]6S), alternatives (214P/K), follow-ups (214P/K~H, 41236K~4~H)
NOTATION_RE = re.compile(r"(?<![\w.])((?:\[[1-9]\])?(?:j\.|c\.|f\.)?[1-9]{1,7}[PKSHD](?:/[PKSHD])*(?:~[1-9]*[PKSHD]?(?:/[PKSHD])*)*|[cfj]\.[PKSHD]|(?:j\.)?Throw|Burst|RC)(?![\w])")
FRAME_RE = re.compile(
    r"(?i)(\b\d+\s*(?:frames?|fotogramm[ia]|f)\b|[+-]\d+\s*(?:on block|on hit|in parata|a segno|frames?|f\b)|"
    r"\b(?:startup|recovery|active frames|frame advantage|frame data|(?:s?vantaggio) di frame|frame trap)\b)")
PCT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
COUNT_RE = re.compile(r"(?i)\b(\d+)\s+(?:times|volte|openers?|aperture|combos?|combo|counter[- ]?hits?|counters?|throws?|prese|lanci)\b")
NOTES_HEAD = re.compile(r"(?im)^#+\s*(coaching notes|note di coaching)\s*$")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _char_aliases() -> dict[str, str]:
    al = {}
    for c in catalog()["chars"]:
        al[c.rstrip("?")] = c
    al.update({"Chaos": "Happy Chaos", "Zato": "Zato-1", "ABA": "A.B.A", "Goldlewis": "Goldlewis",
               "Nago": "Nagoriyuki", "Gio": "Giovanna", "Robo Ky": "Robo-Ky"})
    return al


def _alternatives(part: str) -> list[str]:
    """214P/K -> 214P, 214K; anything without a / is itself."""
    m = re.fullmatch(r"(.*?)([PKSHD](?:/[PKSHD])+)", part)
    return [m.group(1) + b for b in m.group(2).split("/")] if m else [part]


def move_forms(move: str) -> set[str]:
    """Every way of writing a move in the facts that still names that move: 214P/K~H is also 214P/K, 214P, 214K,
    214P~H and 214K~H, and "5H H" is also 5H."""
    head = re.split(r"[ (]", move, maxsplit=1)[0]
    forms = {move, head}
    parts = head.split("~")
    for n in range(1, len(parts) + 1):
        prefixes = [""]
        for part in parts[:n]:
            prefixes = [(p + "~" if p else "") + a for p in prefixes for a in {part, *_alternatives(part)}]
        forms.update(prefixes)
    return forms


def check_answer(answer: str, doc: dict, tol: float = TOL_S) -> dict:
    """Grounding check of an LLM answer against the facts document it was given.
    errors: unknown timestamps, characters not in the match, moves not in the facts, frame data,
    percentages not in the facts, coaching notes without a timestamp. warnings: counts not in the
    facts, a move named next to a timestamp where the facts have a different move, note count."""
    ix = doc["index"]
    times = ix["times"]
    t_all = [x["t_s"] for x in times]
    chars = set(doc["characters"].values()) | {g[s] for g in doc.get("character_segments", []) for s in ("P1", "P2")}
    moves = {c: set(v) for c, v in ix["moves"].items()}
    all_moves = set().union(*moves.values()) if moves else set()
    all_forms = {f for x in all_moves for f in move_forms(x)}
    errors, warnings = [], []

    def near(t):
        return [x for x in times if abs(x["t_s"] - t) <= tol]

    # timestamps
    cited = []
    for m in TS_RE.finditer(answer):
        t = parse_ts(m.group(1))
        cited.append(t)
        if not any(abs(t - u) <= tol for u in t_all):
            errors.append(dict(kind="unknown_timestamp", text=m.group(1)))
    # characters
    for alias, c in _char_aliases().items():
        for m in re.finditer(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", answer):
            if alias == "May" and (m.start() == 0 or answer[m.start() - 2:m.start()] in (". ", "\n")):
                continue   # sentence-initial English "May"
            if c not in chars:
                errors.append(dict(kind="character_not_in_match", text=alias))
    # moves: notation and English names
    named = []
    for m in NOTATION_RE.finditer(answer):
        mv = m.group(1)
        named.append((m.start(), mv))
        if mv not in all_forms:
            errors.append(dict(kind="move_not_in_facts", text=mv))
    names = {}       # move name (and its short form: "S Volcanic Viper" -> "Volcanic Viper") -> full names
    for ch, nm in catalog()["names"].items():
        for no, n in nm.items():
            if len(n) >= 5 and re.search(r"[A-Za-z]{3}", n):
                for v in {n, re.sub(r"^(?:Aerial |[PKSHD] )+", "", n)}:
                    if len(v) >= 5:
                        names.setdefault(v, set()).add(n)
    in_facts = {n for c in chars for n in moves.get(c, ())}
    spans = []       # longest names first; a shorter name inside a matched one is not checked again
    for n, full in sorted(names.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"(?<!\w){re.escape(n)}(?!\w)", answer):
            if any(a0 <= m.start() and m.end() <= b0 for a0, b0 in spans):
                continue
            spans.append((m.start(), m.end()))
            if not (full & in_facts or n in in_facts):
                errors.append(dict(kind="move_not_in_facts", text=n))
            else:
                named.append((m.start(), n))
    # frame data
    for m in FRAME_RE.finditer(answer):
        errors.append(dict(kind="frame_data_claim", text=m.group(0).strip()))
    # numbers
    pcts = ix["percentages"]
    for m in PCT_RE.finditer(answer):
        v = float(m.group(1).replace(",", "."))
        if any(abs(v - p) <= 1.0 for p in pcts):
            continue
        if any(0 < a <= b and abs(v - 100.0 * a / b) <= 1.0 for b in ix["counts"] for a in ix["counts"]):
            # a share of two counts in the facts: not a figure of the facts, but it can be true
            warnings.append(dict(kind="derived_percentage", text=m.group(0)))
        else:
            errors.append(dict(kind="unsupported_percentage", text=m.group(0)))
    for m in COUNT_RE.finditer(answer):
        if int(m.group(1)) not in ix["counts"]:
            warnings.append(dict(kind="unsupported_count", text=m.group(0)))
    # move next to a timestamp: the facts should have that move there
    for s in _sentences(answer):
        st = [parse_ts(x) for x in TS_RE.findall(s)]
        sm = {m.group(1) for m in NOTATION_RE.finditer(s)} | {n for n in names if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", s)}
        if not st or not sm:
            continue
        for t in st:
            here = near(t)
            if not here:
                continue
            have = set()
            for x in here:
                for mv in [x.get("move")] + list(x.get("moves") or []):
                    if mv:
                        have.update(move_forms(mv))
                        nm = move_name(x.get("char", ""), mv)
                        if nm:
                            have.add(nm)
            if not any(any(h == v or h.startswith(v + "~") or h.startswith(v + "(") for h in have for v in {m} | names.get(m, set())) for m in sm):
                (errors if len(st) == 1 and len(sm) == 1 else warnings).append(
                    dict(kind="move_time_mismatch", text=f"{ts(t)}: {sorted(sm)} vs facts {sorted(have) or '-'}"))
    # coaching notes
    mh = NOTES_HEAD.search(answer)
    if not mh:
        warnings.append(dict(kind="no_notes_section", text=""))
    else:
        body = answer[mh.end():]
        nxt = re.search(r"(?m)^#+\s", body)
        body = body[:nxt.start()] if nxt else body
        # numbered items are the notes (their continuation lines and sub-bullets belong to them, and a
        # lead-in before "1." is no note); a plain bullet list only counts when nothing is numbered
        numbered = list(re.finditer(r"(?m)^\s*(?:\*\*)?\d+[.)](?:\*\*)?\s+", body))
        if numbered:
            ends = [m.start() for m in numbered[1:]] + [len(body)]
            notes = [body[m.end():e] for m, e in zip(numbered, ends)]
        else:
            notes = [n for n in re.split(r"(?m)^\s*[-*]\s+", body)[1:] if n.strip()]
        for k, n in enumerate(notes, 1):
            if not TS_RE.search(n):
                errors.append(dict(kind="note_without_timestamp", text=f"note {k}: {n.strip()[:60]}"))
        if not 3 <= len(notes) <= 5:
            warnings.append(dict(kind="note_count", text=str(len(notes))))
    return dict(ok=not errors, errors=errors, warnings=warnings, timestamps_cited=len(cited),
                moves_named=len(named))


# ---- verification against the game log (test-fold replays only; scoring, never an input) ----------

def verify_dir(d: Path) -> dict:
    """Facts vs the log: round winners, and opener moves (confident vs uncertain) against
    openers.parquet within the chain's ±6 frames. For reporting only."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DATA
    doc = json.loads((d / "facts.json").read_text(encoding="utf-8"))
    stem = d.name
    import csv
    truth, pa, pb = [], 1.0, 1.0      # round winners from the log: the side whose HP drops to 0 loses
    for r in csv.DictReader(open(DATA.parent / "ggst_logs" / f"{stem}.csv", newline="")):
        a, b = float(r["p1_hp"]), float(r["p2_hp"])
        ka, kb = a <= 0 < pa, b <= 0 < pb
        if ka or kb:
            truth.append(None if ka and kb else "P2" if ka else "P1")
        pa, pb = a, b
    got = [r["winner"] for r in doc["rounds"]]
    op = [r for r in pq.read_table(DATA / "openers.parquet").to_pylist() if r["replay"] == stem and not r.get("is_burst")]
    res = dict(replay=stem, rounds_log=truth, rounds_facts=got, rounds_match=truth == got)
    conf = dict(sure=[0, 0], uncertain=[0, 0])
    for f in doc["openers"]:
        fr = f["t_s"] * FPS60
        m = [r for r in op if r["attacker"] == f["attacker"] and abs(r["video_frame"] - fr) <= 6]
        k = "uncertain" if f["uncertain"] else "sure"
        conf[k][1] += 1
        conf[k][0] += bool(m) and any(r["move"] == f["move"] for r in m)
    res["opener_move_right"] = {k: dict(right=v[0], n=v[1]) for k, v in conf.items()}
    return res


# ---- CLI -------------------------------------------------------------------------------------------

def facts_for_dir(d: Path, langs=("it", "en"), prov: str = "none") -> dict:
    op = json.loads((d / "openers.json").read_text(encoding="utf-8"))
    tr = json.loads((d / "transcript_preds.json").read_text()) if (d / "transcript_preds.json").exists() else None
    hud = Hud.load(d / "frames.parquet") if (d / "frames.parquet").exists() else None
    doc = match_facts(op, tr, hud, match_id=d.name)
    (d / "facts.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = {}
    for lang in langs:
        (d / f"facts_{lang}.md").write_text(render_md(doc, lang), encoding="utf-8")
        p = build_prompt(doc, lang)
        provider(prov).complete(p, d)
        txt = p["system"] + "\n" + p["user"]
        sizes[lang] = dict(chars=len(txt), tokens_est=estimate_tokens(txt))
    return dict(dir=str(d), openers=len(doc["openers"]), rounds=len(doc["rounds"]), result=doc["result"], prompt=sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["facts", "run", "check", "verify"])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--lang", nargs="*", default=["it", "en"])
    ap.add_argument("--provider", default="none", choices=["none"], help="only 'none' runs until the provider is decided")
    a = ap.parse_args()
    if a.mode == "check":
        ans, facts = a.paths
        print(json.dumps(check_answer(ans.read_text(encoding="utf-8"), json.loads(facts.read_text(encoding="utf-8"))),
                         ensure_ascii=False, indent=1))
        return
    if a.mode in ("run", "verify") and not a.paths:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from common import DATA
        a.paths = sorted(p for p in (DATA / "lang").iterdir() if (p / "openers.json").exists())
    if a.mode == "verify":
        out = [verify_dir(d) for d in a.paths]
        print(json.dumps(out, indent=1))
        return
    summ = [facts_for_dir(d, a.lang, a.provider) for d in a.paths]
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
