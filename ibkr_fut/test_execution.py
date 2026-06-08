"""
test_execution.py — pytest suite for algo_execution.py and the new
snapshot / execution sections of live_dynamic.py.

Run:
    python -m pytest ibkr_fut/test_execution.py -v
No real IBKR connection required.
"""

import json
import math
import os
import sys
import tempfile
from datetime import date
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_fut.algo_execution import (
    CANCEL_WAIT_TIME,
    PASSIVE_TIME_OUT,
    TOTAL_TIME_OUT,
    FillResult,
    _adverse_price,
    _adverse_size,
    _sz,
    _valid,
    algo_exec,
    pre_trade_checks,
)
from ibkr_fut.live_dynamic import (
    check_last_targets,
    load_snapshot,
    reconcile_and_execute,
    save_last_targets,
    save_snapshot,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ticker(bid, ask, bid_sz=100, ask_sz=100):
    t = MagicMock()
    t.bid, t.ask, t.bidSize, t.askSize = bid, ask, bid_sz, ask_sz
    return t


def _ib_with_ticker(bid, ask, bid_sz=100, ask_sz=100):
    ticker = _ticker(bid, ask, bid_sz, ask_sz)
    ib = MagicMock()
    ib.reqMktData.return_value = ticker
    ib.sleep.return_value = None
    return ib, ticker


def _make_trade(status="Filled", filled=1, avg_price=100.10, order_id=42, fills=None):
    trade = MagicMock()
    trade.isDone.return_value = False
    trade.orderStatus.status = status
    trade.orderStatus.filled = filled
    trade.orderStatus.avgFillPrice = avg_price
    trade.fills = fills if fills is not None else []
    trade.order.orderId = order_id
    return trade


_MOCK_SPEC = {
    "symbol": "MES",
    "exchange": "CME",
    "currency": "USD",
    "multiplier": 5.0,
    "trading_class": "",
    "price_magnifier": 1.0,
}
_MOCK_DIAG = {
    "ES": {
        "forecast": 5.0,
        "n_ideal": 1.8,
        "raw_price": 5200.0,
        "mult": 50.0,
        "sigma": 0.15,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Pure helpers: _valid, _sz
# ══════════════════════════════════════════════════════════════════════════════

def test_valid_positive_float():
    assert _valid(5.0)

def test_valid_zero():
    assert not _valid(0)

def test_valid_negative():
    assert not _valid(-1.0)

def test_valid_nan():
    assert not _valid(float("nan"))

def test_valid_none():
    assert not _valid(None)

def test_valid_string():
    assert not _valid("abc")

def test_sz_normal():
    assert _sz(50.0) == 50.0

def test_sz_none():
    assert _sz(None) == 0.0

def test_sz_nan():
    assert _sz(float("nan")) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — Ticker helpers: _adverse_price, _adverse_size
# ══════════════════════════════════════════════════════════════════════════════

def test_adverse_price_buy_ask_risen():
    t = _ticker(99.0, 101.0)
    assert _adverse_price("BUY", 100.0, t)

def test_adverse_price_buy_ask_not_risen():
    t = _ticker(99.0, 99.0)
    assert not _adverse_price("BUY", 100.0, t)

def test_adverse_price_buy_ask_equal():
    t = _ticker(99.0, 100.0)
    assert not _adverse_price("BUY", 100.0, t)

def test_adverse_price_sell_bid_fallen():
    t = _ticker(99.0, 101.0)
    assert _adverse_price("SELL", 100.0, t)

def test_adverse_price_sell_bid_not_fallen():
    t = _ticker(101.0, 102.0)
    assert not _adverse_price("SELL", 100.0, t)

def test_adverse_price_invalid_ask():
    t = _ticker(99.0, float("nan"))
    assert not _adverse_price("BUY", 100.0, t)

def test_adverse_size_buy_heavy_buyers():
    # 600 > 5 * 100 → True
    t = _ticker(100.0, 100.25, bid_sz=600, ask_sz=100)
    assert _adverse_size("BUY", t)

def test_adverse_size_buy_balanced():
    t = _ticker(100.0, 100.25, bid_sz=200, ask_sz=100)
    assert not _adverse_size("BUY", t)

def test_adverse_size_sell_heavy_sellers():
    # ask_sz=600 > 5 * bid_sz=100
    t = _ticker(100.0, 100.25, bid_sz=100, ask_sz=600)
    assert _adverse_size("SELL", t)

def test_adverse_size_sell_balanced():
    t = _ticker(100.0, 100.25, bid_sz=100, ask_sz=200)
    assert not _adverse_size("SELL", t)

def test_adverse_size_zero_ask():
    # guard: ask_sz=0 must not divide by zero
    t = _ticker(100.0, 100.25, bid_sz=100, ask_sz=0)
    assert not _adverse_size("BUY", t)


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — pre_trade_checks
# ══════════════════════════════════════════════════════════════════════════════

def test_pre_trade_valid():
    ib, _ = _ib_with_ticker(100.0, 100.25)
    ok, reason, ticker = pre_trade_checks(ib, MagicMock(), 100.1, 0.2, 1)
    assert ok
    assert ticker is not None

def test_pre_trade_no_bid():
    ib, _ = _ib_with_ticker(float("nan"), 100.0)
    ok, reason, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    assert not ok
    assert "no valid bid/ask" in reason

def test_pre_trade_no_ask():
    ib, _ = _ib_with_ticker(100.0, float("nan"))
    ok, reason, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    assert not ok

def test_pre_trade_crossed_market():
    ib, _ = _ib_with_ticker(101.0, 100.0)
    ok, reason, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    assert not ok

def test_pre_trade_price_diverged():
    # threshold = 3 * 0.2 / sqrt(256) = 0.0375
    # mid = (103.76+103.77)/2 = 103.765 → divergence = 0.03765 > 0.0375 → diverged
    ib, _ = _ib_with_ticker(103.76, 103.77)
    ok, reason, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    assert not ok
    assert "diverged" in reason

def test_pre_trade_price_within_threshold():
    # mid = 100.30 → divergence = 0.003 < 0.0375 → ok
    ib, _ = _ib_with_ticker(100.25, 100.35)
    ok, _, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    assert ok

def test_pre_trade_no_pst_close():
    # divergence check skipped when pst_close_ibkr is None
    ib, _ = _ib_with_ticker(200.0, 201.0)  # huge deviation but no pst to check against
    ok, _, _ = pre_trade_checks(ib, MagicMock(), None, 0.2, 1)
    assert ok

def test_pre_trade_zero_sigma():
    # divergence check skipped when sigma == 0
    ib, _ = _ib_with_ticker(200.0, 201.0)
    ok, _, _ = pre_trade_checks(ib, MagicMock(), 100.0, 0, 1)
    assert ok

def test_pre_trade_cancel_on_no_bid():
    ib, _ = _ib_with_ticker(float("nan"), 100.0)
    pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    ib.cancelMktData.assert_called_once()

def test_pre_trade_cancel_on_diverged():
    ib, _ = _ib_with_ticker(103.76, 103.77)
    pre_trade_checks(ib, MagicMock(), 100.0, 0.2, 1)
    ib.cancelMktData.assert_called_once()

def test_pre_trade_no_cancel_on_success():
    ib, _ = _ib_with_ticker(100.0, 100.25)
    pre_trade_checks(ib, MagicMock(), 100.1, 0.2, 1)
    ib.cancelMktData.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — algo_exec
# ══════════════════════════════════════════════════════════════════════════════

def _make_ib_for_algo(trade, ticker):
    ib = MagicMock()
    ib.placeOrder.return_value = trade
    ib.sleep.return_value = None
    return ib


def _fill_on_sleep(trade, fill_at=2):
    """Return a sleep side_effect that marks trade done on the nth call."""
    call_n = [0]

    def _sleep(t):
        call_n[0] += 1
        if call_n[0] >= fill_at:
            trade.isDone.return_value = True

    return _sleep


@patch("ibkr_fut.algo_execution.time")
def test_algo_passive_fill(mock_time):
    mock_time.time.return_value = 0.0
    trade = _make_trade(status="Filled", filled=1, avg_price=100.10)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=2)

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.status == "Filled"
    assert not result.was_aggressive
    assert ib.placeOrder.call_count == 1


@patch("ibkr_fut.algo_execution.time")
def test_algo_partial_fill(mock_time):
    mock_time.time.return_value = 0.0
    trade = _make_trade(status="PartiallyFilled", filled=1, avg_price=100.10)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=2)

    result = algo_exec(ib, MagicMock(), "BUY", 2, ticker)

    assert result.filled_qty == 1
    assert result.status == "PartiallyFilled"


@patch("ibkr_fut.algo_execution.time")
def test_algo_passive_timeout_switches(mock_time):
    # First call = start_time=0, rest = 301 (past passive, under total)
    mock_time.time.side_effect = [0.0] + [301.0] * 100
    trade = _make_trade(status="Filled", filled=1, avg_price=100.25)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=3)

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.was_aggressive
    assert ib.placeOrder.call_count >= 2


@patch("ibkr_fut.algo_execution.time")
def test_algo_adverse_price_switches(mock_time):
    mock_time.time.return_value = 0.0
    trade = _make_trade(status="Filled", filled=1, avg_price=100.50)
    ticker = _ticker(100.0, 100.25)  # ref_price = initial ask = 100.25
    ib = _make_ib_for_algo(trade, ticker)

    call_n = [0]

    def _sleep(t):
        call_n[0] += 1
        if call_n[0] == 1:
            ticker.ask = 100.50  # rises above ref_price → adverse
        if call_n[0] >= 3:
            trade.isDone.return_value = True

    ib.sleep.side_effect = _sleep

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.was_aggressive
    assert ib.placeOrder.call_count >= 2


@patch("ibkr_fut.algo_execution.time")
def test_algo_adverse_size_switches(mock_time):
    mock_time.time.return_value = 0.0
    trade = _make_trade(status="Filled", filled=1, avg_price=100.25)
    ticker = _ticker(100.0, 100.25, bid_sz=100, ask_sz=100)
    ib = _make_ib_for_algo(trade, ticker)

    call_n = [0]

    def _sleep(t):
        call_n[0] += 1
        if call_n[0] == 1:
            ticker.bidSize = 600  # 600 > 5*100 → adverse size
        if call_n[0] >= 3:
            trade.isDone.return_value = True

    ib.sleep.side_effect = _sleep

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.was_aggressive
    assert ib.placeOrder.call_count >= 2


@patch("ibkr_fut.algo_execution.time")
def test_algo_total_timeout_cancels(mock_time):
    mock_time.time.side_effect = [0.0] + [601.0] * 100
    trade = _make_trade(status="Cancelled", filled=0, avg_price=0.0)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    ib.cancelOrder.assert_called_once()
    assert result.status != "Filled"


@patch("ibkr_fut.algo_execution.time")
def test_algo_cancel_wait_called(mock_time):
    mock_time.time.side_effect = [0.0] + [601.0] * 100
    trade = _make_trade(status="Cancelled", filled=0, avg_price=0.0)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)

    algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert call(CANCEL_WAIT_TIME) in ib.sleep.call_args_list


@patch("ibkr_fut.algo_execution.time")
def test_algo_invalid_offside_no_order(mock_time):
    mock_time.time.return_value = 0.0
    ticker = _ticker(float("nan"), 100.25)  # BUY offside = bid = nan
    ib = MagicMock()

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    ib.placeOrder.assert_not_called()
    assert result.status == "Unfilled"


@patch("ibkr_fut.algo_execution.time")
def test_algo_commission_nan_filtered(mock_time):
    mock_time.time.return_value = 0.0

    fill_nan = MagicMock()
    fill_nan.commissionReport.commission = float("nan")
    fill_good = MagicMock()
    fill_good.commissionReport.commission = 2.50

    trade = _make_trade(status="Filled", filled=1, avg_price=100.10,
                        fills=[fill_nan, fill_good])
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=2)

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.commission == 2.50


@patch("ibkr_fut.algo_execution.time")
def test_algo_aggressive_updates_limit(mock_time):
    # time: start=0, then 301 to trigger passive timeout, stays 301
    mock_time.time.side_effect = [0.0] + [301.0] * 100
    trade = _make_trade(status="Filled", filled=1, avg_price=100.40)
    ticker = _ticker(100.0, 100.25)  # initial ask=100.25
    ib = _make_ib_for_algo(trade, ticker)

    call_n = [0]

    def _sleep(t):
        call_n[0] += 1
        if call_n[0] == 3:
            ticker.ask = 100.40  # ask rises in aggressive phase
        if call_n[0] >= 4:
            trade.isDone.return_value = True

    ib.sleep.side_effect = _sleep

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    # initial place + aggressive place at 100.25 + aggressive update at 100.40
    assert ib.placeOrder.call_count >= 3


@patch("ibkr_fut.algo_execution.time")
def test_algo_aggressive_no_update_if_unchanged(mock_time):
    # Aggressive switch happens, ask stays the same → only 2 placeOrder calls
    mock_time.time.side_effect = [0.0] + [301.0] * 100
    trade = _make_trade(status="Filled", filled=1, avg_price=100.25)
    ticker = _ticker(100.0, 100.25)  # ask=100.25 throughout
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=4)

    algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    # 1st: initial passive place; 2nd: aggressive update (100.0→100.25);
    # subsequent loops: ask unchanged → no further placeOrder
    assert ib.placeOrder.call_count == 2


@patch("ibkr_fut.algo_execution.time")
def test_algo_fill_result_order_id(mock_time):
    mock_time.time.return_value = 0.0
    trade = _make_trade(status="Filled", filled=1, avg_price=100.10, order_id=99)
    ticker = _ticker(100.0, 100.25)
    ib = _make_ib_for_algo(trade, ticker)
    ib.sleep.side_effect = _fill_on_sleep(trade, fill_at=2)

    result = algo_exec(ib, MagicMock(), "BUY", 1, ticker)

    assert result.order_id == 99


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — Snapshot I/O: save_snapshot, load_snapshot,
#            check_last_targets, save_last_targets
# ══════════════════════════════════════════════════════════════════════════════

def test_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        today = "2026-06-08"
        targets = {"ES": 2, "MES": -1}
        diag = {"ES": {"sigma": 0.15}, "_meta": {"idm": 1.2}}
        save_snapshot(path, today, 100_000.0, targets, diag)
        snap = load_snapshot(path, today)

    assert snap["targets"] == targets
    assert snap["capital"] == 100_000.0
    assert snap["date"] == today


def test_snapshot_required_keys():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        save_snapshot(path, "2026-06-08", 50_000.0, {}, {})
        with open(path) as fh:
            data = json.load(fh)

    for key in ("date", "capital", "targets", "diag", "computed_at"):
        assert key in data


def test_snapshot_no_tmp_file_remains():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        save_snapshot(path, "2026-06-08", 50_000.0, {}, {})
        assert not os.path.exists(path + ".tmp")


def test_snapshot_serializes_numpy_int64():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        save_snapshot(path, "2026-06-08", 1.0, {"ES": np.int64(2)}, {})
        snap = load_snapshot(path, "2026-06-08")

    assert snap["targets"]["ES"] == 2
    assert isinstance(snap["targets"]["ES"], int)


def test_snapshot_serializes_numpy_float64():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        diag = {"ES": {"sigma": np.float64(0.15)}}
        save_snapshot(path, "2026-06-08", 1.0, {}, diag)
        snap = load_snapshot(path, "2026-06-08")

    assert abs(snap["diag"]["ES"]["sigma"] - 0.15) < 1e-9


def test_snapshot_serializes_date_object():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        diag = {"_meta": {"date": date(2026, 6, 8)}}
        save_snapshot(path, "2026-06-08", 1.0, {}, diag)
        snap = load_snapshot(path, "2026-06-08")

    assert snap["diag"]["_meta"]["date"] == "2026-06-08"


def test_load_snapshot_correct_date():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        save_snapshot(path, "2026-06-08", 1.0, {"ES": 1}, {})
        snap = load_snapshot(path, "2026-06-08")
    assert snap["date"] == "2026-06-08"


def test_load_snapshot_stale_date_exits():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.json")
        save_snapshot(path, "2026-06-07", 1.0, {}, {})  # yesterday
        with pytest.raises(SystemExit):
            load_snapshot(path, "2026-06-08")


def test_load_snapshot_missing_file_exits():
    with pytest.raises(SystemExit):
        load_snapshot("/nonexistent/path/snap.json", "2026-06-08")


def test_check_last_targets_no_file():
    # Should return silently without exception
    check_last_targets("/nonexistent/last_targets.json", {"ES": 1})


def test_check_last_targets_match(capsys):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "last.json")
        with open(path, "w") as fh:
            json.dump({"date": "2026-06-07", "targets": {"ES": 2}}, fh)
        check_last_targets(path, {"ES": 2})
    out = capsys.readouterr().out
    assert "RECONCILIATION" not in out


def test_check_last_targets_mismatch(capsys):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "last.json")
        with open(path, "w") as fh:
            json.dump({"date": "2026-06-07", "targets": {"ES": 2}}, fh)
        check_last_targets(path, {"ES": 1})  # actual=1 vs expected=2
    out = capsys.readouterr().out
    assert "RECONCILIATION" in out
    assert "ES" in out


def test_check_last_targets_missing_from_current(capsys):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "last.json")
        with open(path, "w") as fh:
            json.dump({"date": "2026-06-07", "targets": {"NQ": 1}}, fh)
        check_last_targets(path, {})  # NQ expected but not in current
    out = capsys.readouterr().out
    assert "NQ" in out


def test_save_last_targets_content():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "last.json")
        save_last_targets(path, {"ES": 2, "MES": -1}, "2026-06-08")
        with open(path) as fh:
            data = json.load(fh)
    assert "date" in data
    assert "targets" in data
    assert data["targets"]["ES"] == 2
    assert data["targets"]["MES"] == -1
    assert data["date"] == "2026-06-08"


# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — Guardrails in reconcile_and_execute
#
# Patch order (bottom decorator = first function param):
#   @patch(algo_exec)       → mock_exec  (5th)
#   @patch(pre_trade_checks)→ mock_ptc   (4th)
#   @patch(qualify)         → mock_qual  (3rd)
#   @patch(hold_contract_month)→ mock_hold (2nd)
#   @patch(ib_spec)         → mock_spec  (1st)
# ══════════════════════════════════════════════════════════════════════════════

def _rne_ib(pending_symbols=None):
    """Return a minimal IB mock for reconcile_and_execute."""
    ib = MagicMock()
    if pending_symbols:
        trades = []
        for sym in pending_symbols:
            t = MagicMock()
            t.contract.symbol = sym
            trades.append(t)
        ib.openTrades.return_value = trades
    else:
        ib.openTrades.return_value = []
    return ib


def _default_fill_result(**kwargs):
    defaults = dict(filled_qty=1, avg_price=5210.0, status="Filled",
                    was_aggressive=False, commission=0.0, order_id=42)
    defaults.update(kwargs)
    return FillResult(**defaults)


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_dry_run_no_pre_trade_check(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    ib = _rne_ib()
    ledger = MagicMock()

    placed, skipped = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, ledger, execute=False
    )

    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()
    assert placed == []


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_dry_run_prints_action(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    ib = _rne_ib()

    reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=False
    )

    out = capsys.readouterr().out
    assert "ACTION" in out
    assert "LMT ALGO" in out


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_pending_order_skipped(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC  # symbol = "MES"
    mock_hold.return_value = "202609"
    ib = _rne_ib(pending_symbols=["MES"])

    placed, skipped = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert "MES" in skipped
    mock_ptc.assert_not_called()


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_pre_trade_fail_skips_algo(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (False, "no valid bid/ask", None)
    ib = _rne_ib()

    placed, skipped = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    mock_exec.assert_not_called()
    assert any("pre-trade" in str(s) for s in skipped)


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_filled_logs_to_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_exec.return_value = _default_fill_result(filled_qty=1, avg_price=5210.0, status="Filled")
    ib = _rne_ib()
    ledger = MagicMock()

    reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, ledger, execute=True
    )

    ledger.log_fill.assert_called_once()
    _, kwargs = ledger.log_fill.call_args
    assert kwargs["fill_price"] == 5210.0
    assert kwargs["qty"] == 1


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_partial_logs_filled_qty(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_exec.return_value = _default_fill_result(
        filled_qty=1, avg_price=5200.0, status="PartiallyFilled"
    )
    ib = _rne_ib()
    ledger = MagicMock()

    # target=2 so delta=2, qty=2; only 1 filled
    reconcile_and_execute(
        ib, MagicMock(), {"ES": 2}, {}, _MOCK_DIAG, ledger, execute=True
    )

    _, kwargs = ledger.log_fill.call_args
    assert kwargs["qty"] == 1  # filled_qty, not intended qty
    out = capsys.readouterr().out
    assert "partial fill" in out


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_unfilled_no_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_exec.return_value = _default_fill_result(
        filled_qty=0, avg_price=0.0, status="Unfilled"
    )
    ib = _rne_ib()
    ledger = MagicMock()

    reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, ledger, execute=True
    )

    ledger.log_fill.assert_not_called()
    out = capsys.readouterr().out
    assert "Unfilled" in out


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_cancelled_no_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = "202609"
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_exec.return_value = _default_fill_result(
        filled_qty=0, avg_price=0.0, status="Cancelled"
    )
    ib = _rne_ib()
    ledger = MagicMock()

    reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, ledger, execute=True
    )

    ledger.log_fill.assert_not_called()


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_no_ib_config_skips(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = None  # no IB config → skip
    ib = _rne_ib()

    placed, skipped = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert placed == []
    assert skipped == []
    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.hold_contract_month")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_no_roll_calendar_skips(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = None  # no roll calendar → skip
    ib = _rne_ib()

    placed, skipped = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert placed == []
    assert skipped == []
    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()
