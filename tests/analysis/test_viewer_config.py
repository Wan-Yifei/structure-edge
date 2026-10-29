"""config/viewer_qt.toml drives the Qt viewer's startup state.

The point of the file is that changing what a chart opens with no longer
means editing Python. A test that only checked the loader would not show
that: the loader could be perfect and the widgets could still ignore it.
So the last two classes build real windows against a written config and
read the widgets back.
"""
from __future__ import annotations

import pytest

from analysis import viewer_config as vcfg


@pytest.fixture(autouse=True)
def _restore_real_config():
    """Every test here repoints the module; put it back afterwards."""
    real = vcfg.path()
    yield
    vcfg.reload(real)


def _write(tmp_path, text):
    p = tmp_path / "viewer_qt.toml"
    p.write_text(text, encoding="utf-8")
    vcfg.reload(p)
    return p


class TestTheShippedFileIsValid:
    def test_it_loads_without_complaint(self):
        vcfg.reload(vcfg.path())
        assert vcfg.problems() == []

    def test_every_key_in_it_is_known(self):
        """A key the schema does not know would be silently dead."""
        import tomllib
        with open(vcfg.path(), "rb") as f:
            raw = tomllib.load(f)
        for section, body in raw.items():
            assert section in vcfg.DEFAULTS, f"unknown section {section}"
            for key in body:
                assert key in vcfg.DEFAULTS[section], \
                    f"[{section}] {key} is not in DEFAULTS"


class TestFallbacks:
    def test_a_missing_file_changes_nothing(self, tmp_path):
        vcfg.reload(tmp_path / "nope.toml")
        assert vcfg.problems() == []
        for section, body in vcfg.DEFAULTS.items():
            for key, want in body.items():
                assert vcfg.get(section, key) == want

    def test_an_unreadable_file_is_reported_not_raised(self, tmp_path):
        _write(tmp_path, "this is not = = toml")
        assert any("unreadable" in p for p in vcfg.problems())
        assert vcfg.flag("ob") is vcfg.DEFAULTS["indicators"]["ob"]

    def test_a_silent_key_keeps_its_default(self, tmp_path):
        _write(tmp_path, "[indicators]\nob = false\n")
        assert vcfg.flag("ob") is False
        assert vcfg.flag("absorb") is vcfg.DEFAULTS["indicators"]["absorb"]


class TestTyposAreReported:
    def test_unknown_key(self, tmp_path):
        _write(tmp_path, "[indicators]\nabsorbb = true\n")
        assert any("unknown key" in p and "absorbb" in p
                   for p in vcfg.problems())

    def test_unknown_section(self, tmp_path):
        _write(tmp_path, "[indicatorz]\nob = true\n")
        assert any("unknown section" in p for p in vcfg.problems())

    def test_wrong_type_keeps_the_default(self, tmp_path):
        _write(tmp_path, '[indicators]\nob = "yes"\n')
        assert any("wrong type" in p for p in vcfg.problems())
        assert vcfg.flag("ob") is vcfg.DEFAULTS["indicators"]["ob"]

    def test_a_bool_is_not_accepted_as_a_number(self, tmp_path):
        _write(tmp_path, "[numbers]\nob_max_count = true\n")
        assert any("wrong type" in p for p in vcfg.problems())
        assert vcfg.num("ob_max_count") == \
            vcfg.DEFAULTS["numbers"]["ob_max_count"]

    def test_an_int_is_accepted_where_a_float_is_wanted(self, tmp_path):
        _write(tmp_path, "[liq_hm]\ngamma = 4\n")
        assert vcfg.problems() == []
        assert vcfg.liq("gamma") == pytest.approx(4.0)

    def test_a_misspelt_session_is_dropped_and_reported(self, tmp_path):
        _write(tmp_path, '[session]\nsegments = ["regular", "overnight"]\n')
        assert any("unknown segments" in p for p in vcfg.problems())
        assert vcfg.sessions() == ["regular"]

    def test_a_bad_range_falls_back(self, tmp_path):
        _write(tmp_path, '[session]\nrange = "5d"\n')
        assert any("range" in p for p in vcfg.problems())
        assert vcfg.get("session", "range") == \
            vcfg.DEFAULTS["session"]["range"]


@pytest.fixture(scope="function")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class TestTheViewerObeysIt:
    def test_toggles_spinboxes_sessions_and_range(self, tmp_path, qapp):
        """Everything flipped away from the shipped file, then read back."""
        _write(tmp_path, """
[indicators]
bos_choch = true
ob = false
absorb = false
heatmap = false
tick_vp = true

[session]
segments = ["pre", "night"]
range = "7d"

[numbers]
ob_max_count = 9
profile_bins = 77
tick_imb_ratio = 6.5
""")
        from analysis.trade_viewer_qt import TradeViewerQt
        v = TradeViewerQt()
        for _ in range(10):
            qapp.processEvents()
        try:
            assert v._ind_checks["bos_choch"].isChecked()
            assert not v._ind_checks["ob"].isChecked()
            assert not v._ind_checks["absorb"].isChecked()
            assert not v._ind_checks["heatmap"].isChecked()
            assert v._ind_checks["tick_vp"].isChecked()

            assert not v._ind_checks["sess_regular"].isChecked()
            assert v._ind_checks["sess_pre"].isChecked()
            assert not v._ind_checks["sess_post"].isChecked()
            assert v._ind_checks["sess_night"].isChecked()

            assert v._ind_checks["range_7d"].isChecked()
            assert not v._ind_checks["range_1d"].isChecked()

            assert v._ob_max_count.value() == 9
            assert v._profile_bins_spin.value() == 77
            assert v._tick_imb_ratio.value() == pytest.approx(6.5)
        finally:
            v.close()
            qapp.processEvents()

    def test_a_broken_config_still_opens_the_window(self, tmp_path, qapp):
        """And says so. The reporting itself used to be the crash.

        The complaints were logged from inside _build_toolbar, but _log
        writes to the status bar, which __init__ builds afterwards. With a
        valid file problems() is empty and the loop never ran, so the first
        person to typo a key would have got an AttributeError instead of a
        message.
        """
        _write(tmp_path, '[indicators]\nabsorbb = true\nob = "yes"\n'
                         '[session]\nrange = "9d"\n')
        from analysis.trade_viewer_qt import TradeViewerQt
        v = TradeViewerQt()
        for _ in range(10):
            qapp.processEvents()
        try:
            assert v._ind_checks["ob"].isChecked(), "fallback not applied"
            assert v._ind_checks["range_1d"].isChecked()
            assert len(vcfg.problems()) == 3
        finally:
            v.close()
            qapp.processEvents()

    def test_an_unreadable_config_still_opens_the_window(self, tmp_path,
                                                         qapp):
        _write(tmp_path, "= = not toml")
        from analysis.trade_viewer_qt import TradeViewerQt
        v = TradeViewerQt()
        for _ in range(10):
            qapp.processEvents()
        try:
            assert v._ind_checks["absorb"].isChecked()
        finally:
            v.close()
            qapp.processEvents()

    def test_the_quota_spinbox_follows_its_checkbox(self, tmp_path, qapp):
        """A default-on OB must not open with its quota box greyed out."""
        _write(tmp_path, "[indicators]\nob = true\nfvg = true\n")
        from analysis.trade_viewer_qt import TradeViewerQt
        v = TradeViewerQt()
        for _ in range(10):
            qapp.processEvents()
        try:
            assert v._ob_max_count.isEnabled()
            assert v._fvg_min_pct.isEnabled()
        finally:
            v.close()
            qapp.processEvents()


class TestTheHeatmapObeysIt:
    def test_its_controls_come_from_the_file(self, tmp_path, qapp):
        _write(tmp_path, """
[liq_hm]
gamma = 1.4
wall_n = 7
wall = false
spoof = true
simb_ratio = 9.5
n_price = 64
""")
        from analysis.liq_hm_window import LiqHmWindow
        w = LiqHmWindow()
        for _ in range(10):
            qapp.processEvents()
        try:
            assert w._gamma_spin.value() == pytest.approx(1.4)
            assert w._wall_n_spin.value() == 7
            assert not w._wall_cb.isChecked()
            assert w._spoof_cb.isChecked()
            assert w._simb_ratio_spin.value() == pytest.approx(9.5)
            assert w._n_price_spin.value() == 64
        finally:
            w.close()
            qapp.processEvents()
