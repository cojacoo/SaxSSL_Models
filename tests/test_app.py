"""End-to-end: the app reads the example OPUS files and predicts every sample.

st.file_uploader cannot be driven by AppTest, so it is patched to return the
files in beispiele/ (2 samples x 4 replicates).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _File:
    def __init__(self, p: Path):
        self.name, self._b = p.name, p.read_bytes()

    def getvalue(self):
        return self._b


def test_upload_table(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    files = [_File(p) for p in sorted((ROOT / "beispiele").glob("*.0"))]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: files)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(f"{len(files)} Dateien → 2 Proben" in s.value for s in at.success)

    wide = at.dataframe[0].value
    assert list(wide.index) == ["BDF-LAAB", "BDF-LAAC"]
    assert any("OSSL" in c for c in wide.columns) and any("SaxSSL" in c for c in wide.columns)

    at.sidebar.checkbox[0].check().run()               # peat models on
    assert not at.exception
    assert any("Torf" in c for c in at.dataframe[0].value.columns)
