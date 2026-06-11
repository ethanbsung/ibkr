"""
Unit tests for the carry (Strategy 10) and combined carry+trend (Strategy 11)
forecast engines — ibkr_fut/carry_signals.py and ibkr_fut/carry_trend_signals.py.

These are DATA-INDEPENDENT: every fixture is a synthetic InstrumentSpec / multiple-
prices frame with a known carry sign and magnitude, so the tests are deterministic and
do not touch the PST CSV warehouse.  The focus is the behaviour that must be bulletproof
before carry goes into the live executor:

  • the carry sign / risk-adjustment / FDM math,
  • the STALE-CARRY FAIL-SAFE (a frozen term structure must force the forecast to NaN
    past CARRY_FFILL_LIMIT, never trade months-old carry),
  • the combined 60/40 blend, the single FDM keyed by TOTAL rule count, and the
    PER-TIMESTAMP fallback (a NaN carry leg must NOT wipe a valid trend forecast).

Run:  source venv/bin/activate && python3 -m pytest ibkr_fut/test_carry_combined.py -q
"""

import numpy as np
import pandas as pd
import pytest

from ibkr_fut.backtest_ewmac import InstrumentSpec
from ibkr_fut.foundations import (
    blended_vol,
    pct_returns_backadjusted,
    sigma_p_from_pct,
)
from ibkr_fut import carry_signals as cs
from ibkr_fut import carry_trend_signals as ct


# ── Synthetic fixtures ───────────────────────────────────────────────────────────────

# Contract codes 3 months apart → expiry gap of exactly 0.25 years, so an annualised
# raw carry of A points is produced by a constant PRICE-CARRY spread of A*0.25.
_GAP_YEARS = 0.25
_PRICE_CONTRACT = 20200600.0   # YYYYMM00 → 2020-06
_CARRY_CONTRACT = 20200900.0   # YYYYMM00 → 2020-09  (further → positive carry when PRICE>CARRY)


def make_spec(name="TEST", n=400, spread=0.0, mult=1.0, rolls=4, seed=0,
              start_price=100.0, sigma=0.01):
    """A synthetic InstrumentSpec: a fixed-seed geometric random walk.  spread=0 makes
    every trading rule cost-eligible; a large spread makes none eligible."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, n)
    raw = start_price * np.exp(np.cumsum(rets))
    raw_price = pd.Series(raw, index=idx)
    return InstrumentSpec(
        name=name, prices=raw_price.copy(), raw_price=raw_price,
        fx=pd.Series(1.0, index=idx), mult=mult, spread=spread, rolls=rolls,
        commission=0.0,
    )


def make_mp(raw_price, ann_carry=2.0, n_rows=None):
    """multiple-prices frame with a CONSTANT annualised raw carry == ann_carry points.
    n_rows truncates the term structure (simulates a frozen/stale roll calendar: rows
    beyond n_rows are simply absent from mp)."""
    idx = raw_price.index if n_rows is None else raw_price.index[:n_rows]
    price = raw_price.reindex(idx)
    carry = price - ann_carry * _GAP_YEARS         # PRICE - CARRY = ann_carry * gap_years
    return pd.DataFrame({
        "PRICE": price,
        "CARRY": carry,
        "PRICE_CONTRACT": pd.Series(_PRICE_CONTRACT, index=idx),
        "CARRY_CONTRACT": pd.Series(_CARRY_CONTRACT, index=idx),
    })


def sigma_series(spec):
    """Replicate the vol the instrument-signal builders compute, on raw_price.index."""
    ret = pct_returns_backadjusted(spec.prices, spec.raw_price)
    sigma = blended_vol(ret)
    return sigma.reindex(spec.prices.index, method="ffill")


# ════════════════════════════════════════════════════════════════════════════════════
#  CARRY  (Strategy 10)
# ════════════════════════════════════════════════════════════════════════════════════

class TestContractMonths:
    def test_yyyymm00_to_absolute_month(self):
        c = pd.Series([20200600.0, 20200900.0, 20211200.0])
        m = cs._contract_months(c)
        assert m.tolist() == [2020 * 12 + 6, 2020 * 12 + 9, 2021 * 12 + 12]


class TestAnnualisedRawCarry:
    def _mp(self, price, carry, pc, cc):
        idx = pd.bdate_range("2020-01-01", periods=len(price))
        return pd.DataFrame({
            "PRICE": pd.Series(price, index=idx),
            "CARRY": pd.Series(carry, index=idx),
            "PRICE_CONTRACT": pd.Series(float(pc), index=idx),
            "CARRY_CONTRACT": pd.Series(float(cc), index=idx),
        })

    def test_carry_further_backwardation_is_positive(self):
        # CARRY further out (2020-09 vs 2020-06), PRICE>CARRY (backwardation) → positive.
        mp = self._mp([100.0], [99.0], _PRICE_CONTRACT, _CARRY_CONTRACT)
        arc = cs.annualised_raw_carry(mp)
        # gap = +0.25yr, raw = +1.0 → 1.0/0.25 = 4.0
        assert arc.iloc[0] == pytest.approx(4.0)

    def test_carry_nearer_sign_consistent(self):
        # CARRY nearer (2020-03 vs 2020-06): both numerator and gap flip → same sign rule.
        mp = self._mp([100.0], [101.0], _PRICE_CONTRACT, 20200300.0)
        arc = cs.annualised_raw_carry(mp)
        # raw = -1.0, gap = (3-6)/12 = -0.25 → (-1)/(-0.25) = +4.0  (still positive carry)
        assert arc.iloc[0] == pytest.approx(4.0)

    def test_coincident_contracts_is_nan(self):
        mp = self._mp([100.0], [99.0], _PRICE_CONTRACT, _PRICE_CONTRACT)
        arc = cs.annualised_raw_carry(mp)
        assert np.isnan(arc.iloc[0])


class TestCarryEligibleSpans:
    def test_zero_cost_all_spans_eligible(self):
        assert cs.carry_eligible_spans(0.0, rolls_per_year=4) == cs.CARRY_SPANS

    def test_huge_cost_none_eligible(self):
        assert cs.carry_eligible_spans(1.0, rolls_per_year=4) == []


class TestCarrySpanForecasts:
    def test_empty_spans_returns_empty_dict(self):
        spec = make_spec()
        out = cs.carry_span_forecasts(make_mp(spec.raw_price), spec.raw_price,
                                      sigma_series(spec), active_spans=[])
        assert out == {}

    def test_positive_carry_positive_forecast(self):
        spec = make_spec()
        sig = sigma_series(spec)
        out = cs.carry_span_forecasts(make_mp(spec.raw_price, ann_carry=2.0),
                                      spec.raw_price, sig, active_spans=[20])
        fc = out[20].dropna()
        assert (fc.tail(50) > 0).all()

    def test_negative_carry_negative_forecast(self):
        spec = make_spec()
        sig = sigma_series(spec)
        out = cs.carry_span_forecasts(make_mp(spec.raw_price, ann_carry=-2.0),
                                      spec.raw_price, sig, active_spans=[20])
        fc = out[20].dropna()
        assert (fc.tail(50) < 0).all()

    def test_forecast_capped_at_20(self):
        spec = make_spec()
        sig = sigma_series(spec)
        out = cs.carry_span_forecasts(make_mp(spec.raw_price, ann_carry=80.0),
                                      spec.raw_price, sig, active_spans=[5])
        fc = out[5].dropna()
        assert fc.max() <= cs.FORECAST_CAP + 1e-9
        assert fc.iloc[-1] == pytest.approx(cs.FORECAST_CAP)   # saturates positive

    def test_stale_carry_forces_nan_past_ffill_limit(self):
        """THE critical fail-safe: a term structure that stops updating must produce NaN
        beyond CARRY_FFILL_LIMIT, not silently hold its last carry forever."""
        spec = make_spec(n=120)
        sig = sigma_series(spec)
        n_fresh = 80
        mp = make_mp(spec.raw_price, ann_carry=3.0, n_rows=n_fresh)   # carry stops at idx 79
        out = cs.carry_span_forecasts(mp, spec.raw_price, sig, active_spans=[20])
        fc = out[20]
        last_fresh = spec.raw_price.index[n_fresh - 1]
        horizon = spec.raw_price.index[n_fresh - 1 + cs.CARRY_FFILL_LIMIT]
        # within the ffill horizon the last carry is still carried (forecast defined)
        assert fc.loc[last_fresh] == fc.loc[last_fresh]   # not NaN
        assert fc.loc[horizon] == fc.loc[horizon]         # exactly at the limit: still defined
        # one day past the horizon → NaN, and every row after stays NaN
        beyond = spec.raw_price.index[n_fresh - 1 + cs.CARRY_FFILL_LIMIT + 1:]
        assert fc.reindex(beyond).isna().all()

    def test_rows_before_first_carry_are_nan(self):
        spec = make_spec(n=60)
        sig = sigma_series(spec)
        # carry only present from idx 30 onward
        idx = spec.raw_price.index
        mp_full = make_mp(spec.raw_price, ann_carry=2.0)
        mp = mp_full.copy()
        mp.loc[idx[:30], ["PRICE", "CARRY"]] = np.nan
        out = cs.carry_span_forecasts(mp, spec.raw_price, sig, active_spans=[20])
        # the very first row (no carry observed yet, before any ffill source) is NaN
        assert np.isnan(out[20].iloc[0])


class TestCarryForecast:
    def test_empty_spans_returns_zeros(self):
        spec = make_spec()
        out = cs.carry_forecast(make_mp(spec.raw_price), spec.raw_price,
                                sigma_series(spec), active_spans=[])
        assert (out == 0.0).all()
        assert out.index.equals(spec.raw_price.index)

    def test_equalweight_fdm_matches_manual(self):
        """carry_forecast == equal-weight mean of per-span forecasts × FDM, capped."""
        spec = make_spec()
        sig = sigma_series(spec)
        spans = [5, 20, 60, 120]
        per = cs.carry_span_forecasts(make_mp(spec.raw_price, ann_carry=2.0),
                                      spec.raw_price, sig, active_spans=spans)
        manual = sum((1.0 / len(spans)) * per[s] for s in spans)
        manual = (manual * cs.CARRY_FDM[tuple(spans)]).clip(-cs.FORECAST_CAP, cs.FORECAST_CAP)
        got = cs.carry_forecast(make_mp(spec.raw_price, ann_carry=2.0),
                                spec.raw_price, sig, active_spans=spans)
        pd.testing.assert_series_equal(got.dropna(), manual.dropna())

    def test_stale_carry_propagates_nan(self):
        spec = make_spec(n=120)
        sig = sigma_series(spec)
        mp = make_mp(spec.raw_price, ann_carry=3.0, n_rows=80)
        out = cs.carry_forecast(mp, spec.raw_price, sig, active_spans=[5, 20, 60, 120])
        assert out.iloc[-1] != out.iloc[-1]   # NaN at the end (stale)


class TestCarryInstrumentSignals:
    def test_mp_none_returns_none(self):
        spec = make_spec()
        assert cs.carry_instrument_signals(spec, mp=None) is None

    def test_no_eligible_spans_returns_none(self):
        spec = make_spec(spread=1e6)   # cost gate kills every span
        assert cs.carry_instrument_signals(spec, make_mp(spec.raw_price)) is None

    def test_returns_expected_shape(self):
        spec = make_spec()
        sig = cs.carry_instrument_signals(spec, make_mp(spec.raw_price, ann_carry=2.0))
        assert set(sig) == {"sigma", "forecast", "active"}
        assert sig["active"] == cs.CARRY_SPANS
        assert sig["forecast"].index.equals(spec.prices.index)
        assert sig["sigma"].index.equals(spec.prices.index)
        assert sig["forecast"].abs().dropna().max() <= cs.FORECAST_CAP + 1e-9


# ════════════════════════════════════════════════════════════════════════════════════
#  COMBINED CARRY + TREND  (Strategy 11)
# ════════════════════════════════════════════════════════════════════════════════════

class TestCombinedFDM:
    def test_exact_table_lookups(self):
        assert ct.combined_fdm(1) == 1.00
        assert ct.combined_fdm(3) == 1.03
        assert ct.combined_fdm(4) == 1.23      # the documented 3→4 jump
        assert ct.combined_fdm(9) == 1.34

    def test_nonpositive_count_is_one(self):
        assert ct.combined_fdm(0) == 1.0
        assert ct.combined_fdm(-5) == 1.0

    def test_above_table_clamps_to_max(self):
        assert ct.combined_fdm(99) == ct.COMBINED_FDM[max(ct.COMBINED_FDM)]


class TestTrendCappedForecasts:
    def test_one_series_per_active_speed_capped(self):
        spec = make_spec()
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        out = ct._trend_capped_forecasts(spec.prices, sp, active_fast=[4, 16])
        assert set(out) == {4, 16}
        for s in out.values():
            assert s.abs().dropna().max() <= ct.FORECAST_CAP + 1e-9


class TestCombinedForecast:
    def _atoms(self, spec, mp, active_trend, active_carry):
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        return sig, sp

    def test_both_styles_empty_returns_zeros(self):
        spec = make_spec()
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        out = ct.combined_carry_trend_forecast(
            spec.prices, spec.raw_price, sp, sig, make_mp(spec.raw_price),
            active_trend=[], active_carry=[])
        assert (out == 0.0).all()

    def test_blend_matches_manual_default_weight_with_total_fdm(self):
        """Where BOTH legs have signal: combined == (w·mt + (1−w)·mc)·FDM(n_t+n_c), capped,
        with w = ct.TREND_WEIGHT (the module default the live system uses)."""
        spec = make_spec()
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        mp = make_mp(spec.raw_price, ann_carry=2.0)
        at, ac = [4, 16, 64], [20, 60]

        got = ct.combined_carry_trend_forecast(spec.prices, spec.raw_price, sp, sig, mp, at, ac)

        # independent recomputation from the public atoms
        tf = ct._trend_capped_forecasts(spec.prices, sp, at)
        cf = cs.carry_span_forecasts(mp, spec.raw_price, sig, ac)
        mt = pd.DataFrame(tf).mean(axis=1)
        mc = pd.DataFrame(cf).mean(axis=1)
        fdm = ct.combined_fdm(len(at) + len(ac))
        w = ct.TREND_WEIGHT
        manual = ((w * mt + (1.0 - w) * mc) * fdm).clip(-ct.FORECAST_CAP, ct.FORECAST_CAP)

        both = mt.notna() & mc.notna()
        pd.testing.assert_series_equal(got[both], manual[both], check_names=False)

    def test_custom_trend_weight_respected(self):
        spec = make_spec()
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        mp = make_mp(spec.raw_price, ann_carry=2.0)
        at, ac = [4, 16], [20]
        got = ct.combined_carry_trend_forecast(spec.prices, spec.raw_price, sp, sig, mp,
                                               at, ac, trend_weight=0.30)
        tf = ct._trend_capped_forecasts(spec.prices, sp, at)
        cf = cs.carry_span_forecasts(mp, spec.raw_price, sig, ac)
        mt = pd.DataFrame(tf).mean(axis=1)
        mc = pd.DataFrame(cf).mean(axis=1)
        fdm = ct.combined_fdm(len(at) + len(ac))
        manual = ((0.30 * mt + 0.70 * mc) * fdm).clip(-ct.FORECAST_CAP, ct.FORECAST_CAP)
        both = mt.notna() & mc.notna()
        pd.testing.assert_series_equal(got[both], manual[both], check_names=False)

    def test_stale_carry_falls_back_to_trend_only(self):
        """REGRESSION: when carry goes stale mid-history the combined forecast must fall
        back to the trend leg (own-count FDM), NOT become NaN (trend + 0.4·NaN → NaN)."""
        spec = make_spec(n=200)
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        n_fresh = 120
        mp = make_mp(spec.raw_price, ann_carry=3.0, n_rows=n_fresh)
        at, ac = [4, 16, 64], [20, 60]

        got = ct.combined_carry_trend_forecast(spec.prices, spec.raw_price, sp, sig, mp, at, ac)

        # deep in the stale region the forecast is still defined (trend-only) ...
        tail = spec.raw_price.index[-20:]
        assert got.reindex(tail).notna().all()

        # ... and equals the trend leg × FDM(n_trend), not the 60/40 blend.
        tf = ct._trend_capped_forecasts(spec.prices, sp, at)
        mt = pd.DataFrame(tf).mean(axis=1)
        expected_trend_only = (mt * ct.combined_fdm(len(at))).clip(-ct.FORECAST_CAP, ct.FORECAST_CAP)
        pd.testing.assert_series_equal(got.reindex(tail), expected_trend_only.reindex(tail),
                                       check_names=False)

    def test_carry_only_instrument_is_carry_only_forecast(self):
        """Symmetric fallback: a carry-only instrument (no eligible trend rule — e.g. the
        carry-only EU-HEALTH/PALLAD names in the live combined universe) must produce a
        pure carry forecast at the own-count FDM, never NaN from the absent trend leg."""
        spec = make_spec(n=400)
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        mp = make_mp(spec.raw_price, ann_carry=2.0)
        at, ac = [], [5, 20]   # no trend rules, carry only

        got = ct.combined_carry_trend_forecast(spec.prices, spec.raw_price, sp, sig, mp, at, ac)

        cf = cs.carry_span_forecasts(mp, spec.raw_price, sig, ac)
        mc = pd.DataFrame(cf).mean(axis=1)
        c_only = mc.notna()
        assert c_only.any()
        expected = (mc * ct.combined_fdm(len(ac))).clip(-ct.FORECAST_CAP, ct.FORECAST_CAP)
        pd.testing.assert_series_equal(got[c_only], expected[c_only], check_names=False)

    def test_output_capped_at_20(self):
        spec = make_spec()
        sig = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sig)
        mp = make_mp(spec.raw_price, ann_carry=80.0)   # carry saturates ±20
        out = ct.combined_carry_trend_forecast(spec.prices, spec.raw_price, sp, sig, mp,
                                               active_trend=[4, 16], active_carry=[5, 20])
        assert out.abs().dropna().max() <= ct.FORECAST_CAP + 1e-9


class TestCarryTrendInstrumentSignals:
    def test_mp_none_degrades_to_trend_only(self):
        spec = make_spec()
        sig = ct.carry_trend_instrument_signals(spec, mp=None)
        assert sig is not None
        assert sig["active_carry"] == []
        assert sig["active_trend"]            # trend rules present
        assert sig["forecast"].abs().dropna().max() <= ct.FORECAST_CAP + 1e-9

    def test_both_styles_empty_returns_none(self):
        spec = make_spec(spread=1e6)   # cost gate kills trend AND carry
        assert ct.carry_trend_instrument_signals(spec, make_mp(spec.raw_price)) is None

    def test_returns_expected_shape(self):
        spec = make_spec()
        sig = ct.carry_trend_instrument_signals(spec, make_mp(spec.raw_price, ann_carry=2.0))
        assert set(sig) == {"sigma", "forecast", "active_trend", "active_carry"}
        assert sig["forecast"].index.equals(spec.prices.index)
        assert sig["sigma"].index.equals(spec.prices.index)
        assert sig["active_carry"] == cs.CARRY_SPANS

    def test_combined_forecast_consistent_with_builder(self):
        """The instrument-signal wrapper feeds the same atoms to the combined builder."""
        spec = make_spec()
        mp = make_mp(spec.raw_price, ann_carry=2.0)
        sig = ct.carry_trend_instrument_signals(spec, mp)
        sigma = sigma_series(spec)
        sp = sigma_p_from_pct(spec.raw_price, sigma)
        expected = ct.combined_carry_trend_forecast(
            spec.prices, spec.raw_price, sp, sigma, mp,
            sig["active_trend"], sig["active_carry"])
        pd.testing.assert_series_equal(sig["forecast"], expected, check_names=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
