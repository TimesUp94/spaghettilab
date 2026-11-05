// ── Combos (hit counts) ─────────────────────────────────────────────────────
//
// The `combos` table is written by the CV pipeline (replanal/storage.py write_combos: one row per combo
// the game's hit counter showed, with its highest count), and travels in .spag / .spagz files since
// schema v2. A replay analysed before the table existed has no rows, and a file from before v2 has no
// table until it is opened: both simply show no hit counts.
//
// Matching: a damage event and a combo are the same thing when the same player was hit at about the same
// time and the bar lost the same stretch of HP. Time alone isn't enough: the counter only appears from the
// second hit on and a damage event starts at the first, so a combo can start well after "its" event, and a
// poke just before a combo has a nearby time too. The HP stretch is what tells those apart.
//
// One event can hold two combos (the counter restarted while the red segment stayed up: [3, 2]), and one
// combo can span several events (the HP drops in steps): only the first shows the count, the others are
// marked `combo_continued`.
//
// The same rule is in FGCScoreboard app/components/spaghetti-lab/comboHits.js, with the same test cases.

use super::DamageEvent;
use rusqlite::Connection;
use serde::Serialize;

/// The counter can appear this long after a damage event ended (a long first hit, then the second).
const LEAD_MS: f64 = 3000.0;
/// A damage event can start this long after the counter's last frame (the bar drains late).
const TAIL_MS: f64 = 500.0;
/// The shared HP stretch must cover at least this share of the smaller of the two drops.
const MIN_OVERLAP: f64 = 0.5;

#[derive(Debug, Serialize, Clone)]
pub struct Combo {
    pub attacker_side: i64,
    pub frame_start: i64,
    pub frame_end: i64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub max_hits: i64,
    pub hp_before: Option<f64>,
    pub hp_after: Option<f64>,
    pub damage_pct: Option<f64>,
}

const COMBOS_DDL: &str = "(
    combo_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    replay_id     TEXT,
    attacker_side INTEGER,
    frame_start   INTEGER,
    frame_end     INTEGER,
    start_ms      REAL,
    end_ms        REAL,
    max_hits      INTEGER,
    hp_before     REAL,
    hp_after      REAL,
    damage_pct    REAL
)";

const COMBOS_COLUMNS: &str = "replay_id, attacker_side, frame_start, frame_end, start_ms, end_ms, max_hits, hp_before, hp_after, damage_pct";

/// Schema v2's table, on a database that may not have it yet.
pub fn ensure_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(&format!("CREATE TABLE IF NOT EXISTS combos {};", COMBOS_DDL)).map_err(|e| e.to_string())
}

fn has_table(conn: &Connection) -> bool {
    conn.query_row("SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = 'combos'", [], |_| Ok(())).is_ok()
}

/// Create `dst.combos` and copy this replay's rows: the copy of both exports (`export_spag`,
/// `copy_replay_tables_to_dst`). The source isn't touched: without a table there, the file gets an empty one.
pub fn copy_to_dst(src: &Connection, replay_id: &str) -> Result<(), String> {
    src.execute_batch(&format!("CREATE TABLE IF NOT EXISTS dst.combos {};", COMBOS_DDL)).map_err(|e| e.to_string())?;
    if !has_table(src) {
        return Ok(());
    }
    src.execute(
        &format!("INSERT INTO dst.combos ({c}) SELECT {c} FROM main.combos WHERE replay_id = ?", c = COMBOS_COLUMNS),
        [replay_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// A replay's combos by time; none when the database has no such table (analysed before it existed).
pub fn load(conn: &Connection, replay_id: &str) -> Vec<Combo> {
    if !has_table(conn) {
        return Vec::new();
    }
    let read = || -> rusqlite::Result<Vec<Combo>> {
        let mut stmt = conn.prepare(
            "SELECT attacker_side, frame_start, frame_end, start_ms, end_ms, max_hits, hp_before, hp_after, damage_pct
             FROM combos WHERE replay_id = ? AND max_hits IS NOT NULL ORDER BY start_ms",
        )?;
        let rows = stmt.query_map([replay_id], |row| {
            Ok(Combo {
                attacker_side: row.get(0)?,
                frame_start: row.get(1)?,
                frame_end: row.get(2)?,
                start_ms: row.get(3)?,
                end_ms: row.get(4)?,
                max_hits: row.get(5)?,
                hp_before: row.get(6)?,
                hp_after: row.get(7)?,
                damage_pct: row.get(8)?,
            })
        })?;
        rows.collect()
    };
    read().unwrap_or_default()
}

/// The moment an event's HP drop ended, on the same clock as the combos' times.
fn event_end_ms(e: &DamageEvent) -> f64 {
    let ms_per_frame = if e.frame_start > 0 { e.timestamp_ms / e.frame_start as f64 } else { 1000.0 / 30.0 };
    e.frame_end as f64 * ms_per_frame
}

fn same_damage(e: &DamageEvent, c: &Combo) -> bool {
    if e.target_side != 3 - c.attacker_side {
        return false;
    }
    if event_end_ms(e) < c.start_ms - LEAD_MS || e.timestamp_ms > c.end_ms + TAIL_MS {
        return false;
    }
    let (Some(before), Some(after)) = (c.hp_before, c.hp_after) else { return false };
    let shared = e.pre_health_pct.min(before) - e.post_health_pct.max(after);
    let smaller = (e.pre_health_pct - e.post_health_pct).min(before - after);
    shared > 0.0 && shared >= MIN_OVERLAP * smaller
}

/// Fill `combo_hits` and `combo_continued` of events (sorted by time, HP on the scale the combos were
/// written with, so before `rescale_events`).
pub fn attach_hits(events: &mut [DamageEvent], combos: &[Combo]) {
    let mut shown = vec![false; combos.len()];
    for e in events.iter_mut() {
        let matched: Vec<usize> = (0..combos.len()).filter(|&i| same_damage(e, &combos[i])).collect();
        let new: Vec<usize> = matched.iter().copied().filter(|&i| !shown[i]).collect();
        e.combo_continued = new.is_empty() && !matched.is_empty();
        e.combo_hits = if e.combo_continued {
            matched.iter().map(|&i| combos[i].max_hits).max().into_iter().collect()
        } else {
            new.iter().map(|&i| combos[i].max_hits).collect()
        };
        for i in new {
            shown[i] = true;
        }
    }
}

/// A replay's combos, for the Lab tab's opener rows.
#[tauri::command]
pub fn get_combos(db_path: String, replay_id: String) -> Result<Vec<Combo>, String> {
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    Ok(load(&conn, &replay_id))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn event(t_s: f64, dur_frames: i64, target: i64, pre: f64, post: f64) -> DamageEvent {
        let frame_start = (t_s * 30.0).round() as i64;
        DamageEvent {
            event_id: 0,
            replay_id: "r".into(),
            timestamp_ms: t_s * 1000.0,
            frame_start,
            frame_end: frame_start + dur_frames,
            target_side: target,
            damage_pct: pre - post,
            pre_health_pct: pre,
            post_health_pct: post,
            combo_hits: Vec::new(),
            combo_continued: false,
        }
    }

    fn combo(start_s: f64, end_s: f64, attacker: i64, hits: i64, before: f64, after: f64) -> Combo {
        Combo {
            attacker_side: attacker,
            frame_start: (start_s * 30.0).round() as i64,
            frame_end: (end_s * 30.0).round() as i64,
            start_ms: start_s * 1000.0,
            end_ms: end_s * 1000.0,
            max_hits: hits,
            hp_before: Some(before),
            hp_after: Some(after),
            damage_pct: Some(before - after),
        }
    }

    fn hits(e: &DamageEvent) -> (Vec<i64>, bool) {
        (e.combo_hits.clone(), e.combo_continued)
    }

    // These cases are the ones in FGCScoreboard's comboHits.test.mjs: change both.

    #[test]
    fn combo_starting_after_its_event_matches_by_defender_and_hp() {
        // 01_WRI_Slycoops_vs_Leftover at 40 s: the counter appears 1.4 s after the first hit
        let mut ev = vec![event(40.4, 133, 2, 0.452, 0.172)];
        attach_hits(&mut ev, &[combo(41.867, 45.4, 1, 9, 0.452, 0.172)]);
        assert_eq!(hits(&ev[0]), (vec![9], false));
    }

    #[test]
    fn the_wrong_defender_gets_nothing() {
        let mut ev = vec![event(40.4, 133, 1, 0.452, 0.172)];
        attach_hits(&mut ev, &[combo(41.867, 45.4, 1, 9, 0.452, 0.172)]);
        assert_eq!(hits(&ev[0]), (vec![], false));
    }

    #[test]
    fn a_poke_just_before_a_combo_gets_nothing() {
        let mut ev = vec![event(37.9, 5, 2, 0.627, 0.453), event(40.4, 133, 2, 0.452, 0.172)];
        attach_hits(&mut ev, &[combo(41.867, 45.4, 1, 9, 0.452, 0.172)]);
        assert_eq!(hits(&ev[0]), (vec![], false));
        assert_eq!(hits(&ev[1]), (vec![9], false));
    }

    #[test]
    fn a_far_away_event_on_the_same_hp_gets_nothing() {
        let mut ev = vec![event(10.0, 10, 2, 0.8, 0.5)];
        attach_hits(&mut ev, &[combo(30.0, 32.0, 1, 4, 0.8, 0.5)]);
        assert_eq!(hits(&ev[0]), (vec![], false));
    }

    #[test]
    fn a_combo_over_two_events_shows_once() {
        let mut ev = vec![event(78.467, 85, 2, 1.0, 0.705), event(82.033, 106, 2, 0.705, 0.572)];
        attach_hits(&mut ev, &[combo(79.333, 88.467, 1, 12, 1.0, 0.572)]);
        assert_eq!(hits(&ev[0]), (vec![12], false));
        assert_eq!(hits(&ev[1]), (vec![12], true));
    }

    #[test]
    fn two_combos_in_one_event_are_both_counted() {
        // 01_WRI_Slycoops_vs_Leftover at 141.6 s: the counter shows 3, restarts, shows 2 (checked on the video)
        let mut ev = vec![event(141.567, 92, 1, 1.0, 0.606)];
        attach_hits(&mut ev, &[combo(142.233, 143.367, 2, 3, 1.0, 0.79), combo(144.7, 145.333, 2, 2, 0.79, 0.606)]);
        assert_eq!(hits(&ev[0]), (vec![3, 2], false));
    }

    #[test]
    fn a_combo_shared_by_two_events_is_not_counted_twice() {
        let mut ev = vec![event(50.0, 60, 2, 0.55, 0.39), event(54.0, 60, 2, 0.39, 0.19)];
        attach_hits(&mut ev, &[combo(50.5, 52.0, 1, 6, 0.55, 0.39), combo(52.5, 54.5, 1, 3, 0.39, 0.19), combo(55.0, 58.0, 1, 6, 0.29, 0.10)]);
        assert_eq!(hits(&ev[0]), (vec![6], false));
        assert_eq!(hits(&ev[1]), (vec![3, 6], false));
    }

    #[test]
    fn no_combos_and_unknown_hp_show_nothing() {
        let mut ev = vec![event(40.4, 133, 2, 0.452, 0.172)];
        attach_hits(&mut ev, &[]);
        assert_eq!(hits(&ev[0]), (vec![], false));
        let mut c = combo(41.867, 45.4, 1, 9, 0.452, 0.172);
        c.hp_before = None;
        attach_hits(&mut ev, &[c]);
        assert_eq!(hits(&ev[0]), (vec![], false));
    }

    #[test]
    fn a_database_without_the_table_has_no_combos() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(load(&conn, "r").is_empty());
        ensure_table(&conn).unwrap();
        assert!(load(&conn, "r").is_empty());
    }
}
