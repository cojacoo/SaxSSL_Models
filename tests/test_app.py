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

    tab = at.dataframe[0].value                        # rows = properties, cols = samples
    assert list(tab.columns) == ["Eigenschaft", "Quelle", "BDF-LAAB", "BDF-LAAC"]
    assert list(tab["Eigenschaft"][:3]) == ["Ähnlichkeit zur Bibliothek"] * 3
    assert list(tab["Quelle"][:3]) == ["SaxSSL", "SaxTox", "OSSL/KSSL"]   # default sources
    assert tab.iloc[0, 2][0] in "🟢🟠🔴"                # similarity as traffic light
    assert tab["Eigenschaft"][3:].str[0].isin(list("🟢🟡🟠")).all()   # quality dots, no D
    assert set(tab["Quelle"]) == {"SaxSSL", "SaxTox", "OSSL/KSSL"}
    assert not tab.astype(str).apply(lambda c: c.str.contains("annah")).any().any()

    at.sidebar.select_slider[0].set_value("A").run()  # only class A
    assert set(at.dataframe[0].value["Eigenschaft"][3:].str[0]) == {"🟢"}
    at.sidebar.select_slider[0].set_value("D").run()  # everything incl. D
    assert "🔴" in set(at.dataframe[0].value["Eigenschaft"].str[0])

    at.checkbox(key="src_Torf (pmird)").check().run()   # peat models on
    at.checkbox(key="src_SaxTox").uncheck().run()       # SaxTox off
    assert not at.exception
    q = set(at.dataframe[0].value["Quelle"])
    assert "Torf (pmird)" in q and "SaxTox" not in q
