"""scripts/check_published_binary.py: the release check that the local models path isn't in a build."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_published_binary as c  # noqa: E402


def test_a_clean_binary_passes(tmp_path):
    f = tmp_path / "app.exe"
    f.write_bytes(b"\x00run_lab_analysis_remote\x00")
    assert c.main([str(f)]) == 0
    assert c.main([str(f), "--expect-local"]) == 1


def test_any_marker_fails(tmp_path):
    f = tmp_path / "app.exe"
    for m in c.MARKERS:
        f.write_bytes(b"\x00" + m + b"\x00")
        assert c.main([str(f)]) == 1


def test_the_local_build_has_them_all(tmp_path):
    f = tmp_path / "app.exe"
    f.write_bytes(b"\x00".join(c.MARKERS))
    assert c.main([str(f), "--expect-local"]) == 0


def test_a_missing_file_is_an_error(tmp_path):
    assert c.main([str(tmp_path / "nope.exe")]) == 2


def test_the_markers_are_in_the_local_module_only():
    src = Path(__file__).resolve().parents[1] / "app" / "src-tauri" / "src"
    for rs in src.glob("*.rs"):
        text = rs.read_bytes()
        code = b"\n".join(line for line in text.splitlines() if not line.lstrip().startswith(b"//"))
        for m in c.MARKERS:
            if rs.name == "lab_local.rs":
                assert m in code, m
            else:
                assert m not in code, (rs.name, m)
