"""nav.to_main, used by the replay batches: no key on a screen it doesn't recognise."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ggst_logger" / "drive"))
pytest.importorskip("cv2")
nav = pytest.importorskip("nav")


class Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def game(monkeypatch):
    clock, sent = Clock(), []
    monkeypatch.setattr(nav, "time", clock)
    monkeypatch.setattr(nav.ggst, "keys", lambda k, *a: sent.append(k))
    return sent, lambda shows: monkeypatch.setattr(nav, "score", lambda name, *a: 0 if name == shows(clock.t) else 99)


def test_unknown_screen_gets_no_key(game):
    sent, shows = game
    shows(lambda t: "nothing we know")
    with pytest.raises(SystemExit, match="no key sent"):
        nav.to_main()
    assert sent == []


def test_title_gets_the_one_allowed_key(game):
    sent, shows = game
    shows(lambda t: "main" if sent else "title")
    nav.to_main()
    assert sent == ["u"]


def test_waits_through_an_attract_video(game):
    sent, shows = game
    shows(lambda t: "attract" if t < 60 else ("main" if sent else "title"))
    nav.to_main()
    assert sent == ["u"]
