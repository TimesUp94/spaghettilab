"""The character settings screen only takes its allow-listed keys. No game needed."""
import sys
import types
from pathlib import Path

import pytest

DRIVE = Path(__file__).resolve().parent.parent / "tools" / "ggst_logger" / "drive"


@pytest.fixture()
def cs(monkeypatch):
    fake_ggst = types.SimpleNamespace(keys=lambda *a, **k: calls.append(a), shot=lambda *a: None)
    calls = []
    fake_ts = types.SimpleNamespace(REFS="", capture=lambda: None)
    monkeypatch.setitem(sys.modules, "ggst", fake_ggst)
    monkeypatch.setitem(sys.modules, "train_sweep", fake_ts)
    monkeypatch.syspath_prepend(str(DRIVE))
    monkeypatch.delitem(sys.modules, "char_settings", raising=False)
    import char_settings
    char_settings._calls = calls
    return char_settings


@pytest.mark.parametrize("key", ["u", "i", "j", "k", "l", "p", "esc", "bksp", "space", "enter", "8"])
def test_other_keys_refused_before_anything_is_sent(cs, monkeypatch, key):
    monkeypatch.setattr(cs, "on_screen", lambda: True)
    with pytest.raises(SystemExit):
        cs.press(key)
    assert cs._calls == []


def test_allowed_key_needs_the_recognised_screen(cs, monkeypatch):
    monkeypatch.setattr(cs, "on_screen", lambda: False)
    with pytest.raises(SystemExit):
        cs.press("d")
    assert cs._calls == []


def test_allowed_key_on_the_screen_is_sent(cs, monkeypatch):
    monkeypatch.setattr(cs, "on_screen", lambda: True)
    cs.press("s", after=0)
    assert len(cs._calls) == 1


def test_value_is_never_stepped_on_another_row(cs, monkeypatch):
    monkeypatch.setattr(cs, "on_screen", lambda: True)
    monkeypatch.setattr(cs, "go_to", lambda label: label)
    monkeypatch.setattr(cs, "highlighted", lambda: ("Security Level", "Normale"))
    with pytest.raises(SystemExit):
        cs.set_value("Blood Rage", "Normale")
    assert cs._calls == []


def test_back_key_is_o_and_nothing_else_leaves(cs):
    assert cs.key_allowed("o")
    for key in ("enter", "8", "u"):
        assert not cs.key_allowed(key)


def test_reset_key_only_after_the_training_hud_check():
    src = (DRIVE / "train_sweep.py").read_text(encoding="utf-8")
    body = src[src.index("def reset_position"):src.index("def walk_back")]
    assert body.index("on_training()") < body.index("RESET_KEY")
    battle = next(line for line in src.splitlines() if line.startswith("BATTLE_KEYS ="))
    assert '"8"' not in battle


def test_paths_screen_lists(monkeypatch, cs):
    monkeypatch.setitem(sys.modules, "nav", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imwrite=None))
    monkeypatch.delitem(sys.modules, "settings_runs", raising=False)
    sys.modules["train_sweep"].LOGS = ""
    import settings_runs as sr
    assert sr.key_allowed("hud", "8") and sr.key_allowed("hud", "enter")
    assert not sr.key_allowed("hud", "u") and not sr.key_allowed("hud", "o")
    assert not sr.key_allowed("pause", "8") and not sr.key_allowed("pause", "enter")
    assert not sr.key_allowed("pause", "d") and not sr.key_allowed("train_settings", "a")
    assert not sr.key_allowed("nowhere", "w")
