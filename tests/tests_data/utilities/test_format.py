from lightning.data.utilities.format import _human_readable_bytes


def test_human_readable_bytes():
    assert _human_readable_bytes(0) == "0.0 B"
    assert _human_readable_bytes(1) == "1.0 B"
    assert _human_readable_bytes(999) == "999.0 B"
    assert _human_readable_bytes(int(1e3)) == "1.0 KB"
    assert _human_readable_bytes(int(1e3 + 1e2)) == "1.1 KB"
    assert _human_readable_bytes(int(1e6)) == "1.0 MB"
    assert _human_readable_bytes(int(1e6 + 2e5)) == "1.2 MB"
    assert _human_readable_bytes(int(1e9)) == "1.0 GB"
    assert _human_readable_bytes(int(1e9 + 3e8)) == "1.3 GB"
    assert _human_readable_bytes(int(1e12)) == "1.0 TB"
    assert _human_readable_bytes(int(1e12 + 4e11)) == "1.4 TB"
    assert _human_readable_bytes(int(1e15)) == "1.0 PB"
    assert _human_readable_bytes(int(1e15 + 5e14)) == "1.5 PB"
    assert _human_readable_bytes(int(1e18)) == "1.0 EB"
    assert _human_readable_bytes(int(1e18 + 6e17)) == "1.6 EB"
    assert _human_readable_bytes(int(1e21)) == "1.0 ZB"
    assert _human_readable_bytes(int(1e21 + 7e20)) == "1.7 ZB"
    assert _human_readable_bytes(int(1e24)) == "1.0 YB"
    assert _human_readable_bytes(int(1e24 + 8e23)) == "1.8 YB"
    assert _human_readable_bytes(int(1e27)) == "1000.0 YB"
