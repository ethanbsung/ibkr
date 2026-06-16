"""
test_execution.py — pytest suite for algo_execution.py and the new
snapshot / execution sections of live_dynamic.py.

Run:
    python -m pytest ibkr_fut/test_execution.py -v
No real IBKR connection required.
"""

import asyncio
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
    fetch_positions,
    get_positions_by_instr,
    load_snapshot,
    reconcile_and_execute,
    save_last_targets,
    save_snapshot,
    spread_roll_exec,
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
#   @patch(get_roll_info)       → mock_hold (2nd)
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
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_dry_run_no_pre_trade_check(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
    ib = _rne_ib()
    ledger = MagicMock()

    placed, skipped, _, _ = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, ledger, execute=False
    )

    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()
    assert placed == []


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_dry_run_prints_action(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
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
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_pending_order_skipped(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC  # symbol = "MES"
    mock_hold.return_value = ("202609", None, 9999)
    ib = _rne_ib(pending_symbols=["MES"])

    placed, skipped, _, _ = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert "MES" in skipped
    mock_ptc.assert_not_called()


@patch("ibkr_fut.live_dynamic.is_contract_okay_to_trade")
@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_pre_trade_fail_skips_algo(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_is_open
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (False, "no valid bid/ask", None)
    mock_is_open.return_value = True   # market open → don't defer
    ib = _rne_ib()

    placed, skipped, _, _ = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    mock_exec.assert_not_called()
    assert any("pre-trade" in str(s) for s in skipped)


@patch("ibkr_fut.live_dynamic.is_contract_okay_to_trade")
@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_filled_logs_to_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_is_open
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_is_open.return_value = True   # market open → don't defer
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


@patch("ibkr_fut.live_dynamic.is_contract_okay_to_trade")
@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_partial_logs_filled_qty(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_is_open, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_is_open.return_value = True   # market open → don't defer
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


@patch("ibkr_fut.live_dynamic.is_contract_okay_to_trade")
@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_unfilled_no_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_is_open, capsys
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
    mock_qual.return_value = MagicMock()
    mock_ptc.return_value = (True, "", MagicMock())
    mock_is_open.return_value = True   # market open → don't defer
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
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_cancelled_no_ledger(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = ("202609", None, 9999)
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
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_no_ib_config_skips(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = None  # no IB config → skip
    ib = _rne_ib()

    placed, skipped, _, _ = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert placed == []
    assert skipped == []
    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()


@patch("ibkr_fut.live_dynamic.algo_exec")
@patch("ibkr_fut.live_dynamic.pre_trade_checks")
@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.get_roll_info")
@patch("ibkr_fut.live_dynamic.ib_spec")
def test_rne_no_roll_calendar_skips(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec
):
    mock_spec.return_value = _MOCK_SPEC
    mock_hold.return_value = (None, None, 9999)  # no roll calendar → skip
    ib = _rne_ib()

    placed, skipped, _, _ = reconcile_and_execute(
        ib, MagicMock(), {"ES": 1}, {}, _MOCK_DIAG, MagicMock(), execute=True
    )

    assert placed == []
    assert skipped == []
    mock_ptc.assert_not_called()
    mock_exec.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — Roll-window routing and spread rolls
# ══════════════════════════════════════════════════════════════════════════════

def _qual_by_month(ib, spec, month):
    """qualify() stand-in returning a distinct contract mock per month."""
    c = MagicMock()
    c.month = month
    c.conId = int(month)
    c.multiplier = "5"
    return c


def _exec_orders(mock_exec):
    """[(month, action, qty)] for each algo_exec call."""
    return [(c.args[1].month, c.args[2], c.args[3])
            for c in mock_exec.call_args_list]


def _roll_patches(fn):
    """Common patch stack for roll-window reconcile tests.

    First-applied patch = innermost = first test parameter, so test signatures
    read: mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open,
    mock_vol (any extra @patch stacked above appends after these).
    """
    fn = patch("ibkr_fut.live_dynamic.ib_spec", return_value=_MOCK_SPEC)(fn)
    fn = patch("ibkr_fut.live_dynamic.get_roll_info")(fn)
    fn = patch("ibkr_fut.live_dynamic.qualify", side_effect=_qual_by_month)(fn)
    fn = patch("ibkr_fut.live_dynamic.pre_trade_checks",
               return_value=(True, "", MagicMock()))(fn)
    fn = patch("ibkr_fut.live_dynamic.algo_exec")(fn)
    fn = patch("ibkr_fut.live_dynamic.is_contract_okay_to_trade",
               return_value=True)(fn)
    fn = patch("ibkr_fut.live_dynamic.check_order_vol",
               return_value=(True, ""))(fn)
    return fn


@_roll_patches
def test_rne_passive_reduce_splits_across_months(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol
):
    # Long 1 expiring + 2 next, target 0: close the expiring leg (1) and take
    # the remaining 2 out of the incoming month — never short the expiring month.
    mock_hold.return_value = ("202609", "202612", 7)   # passive window
    mock_exec.return_value = _default_fill_result()
    held = {"ES": {"202609": 1, "202612": 2}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 0}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    orders = _exec_orders(mock_exec)
    assert ("202609", "SELL", 1) in orders
    assert ("202612", "SELL", 2) in orders
    assert len(orders) == 2


@_roll_patches
def test_rne_passive_reduce_short_splits_across_months(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol
):
    # Symmetric short case: -1 expiring / -2 next, target 0 → BUY 1 expiring,
    # BUY 2 incoming.
    mock_hold.return_value = ("202609", "202612", 7)
    mock_exec.return_value = _default_fill_result()
    held = {"ES": {"202609": -1, "202612": -2}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 0}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    orders = _exec_orders(mock_exec)
    assert ("202609", "BUY", 1) in orders
    assert ("202612", "BUY", 2) in orders
    assert len(orders) == 2


@_roll_patches
def test_rne_passive_add_routes_to_next_month(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol
):
    mock_hold.return_value = ("202609", "202612", 7)
    mock_exec.return_value = _default_fill_result()
    held = {"ES": {"202609": 1}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 3}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    assert _exec_orders(mock_exec) == [("202612", "BUY", 2)]


@_roll_patches
def test_rne_no_roll_window_routes_current_month(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol
):
    mock_hold.return_value = ("202609", None, 9999)
    mock_exec.return_value = _default_fill_result()
    held = {"ES": {"202609": 1}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 3}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    assert _exec_orders(mock_exec) == [("202609", "BUY", 2)]


@patch("ibkr_fut.live_dynamic.spread_roll_exec")
@_roll_patches
def test_rne_spread_roll_credits_fills_any_status(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol,
    mock_roll, capsys
):
    # A cancelled spread limit with partial fills must still be credited to the
    # month legs; the residual close is then sized off the true expiring holding.
    mock_hold.return_value = ("202609", "202612", 2)   # spread window
    mock_exec.return_value = _default_fill_result()
    mock_roll.return_value = ("Cancelled", 2, 0.0)
    held = {"ES": {"202609": 3}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 2}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    mock_roll.assert_called_once()
    assert mock_roll.call_args.args[4] == 2          # rolled qty
    assert mock_roll.call_args.kwargs["force"] is False   # d=2 > FORCE_ROLL_DAYS
    assert _exec_orders(mock_exec) == [("202609", "SELL", 1)]
    out = capsys.readouterr().out
    assert "cur=202609:+1" in out                    # fills credited to legs
    assert "nxt=202612:+2" in out


@patch("ibkr_fut.live_dynamic.spread_roll_exec")
@_roll_patches
def test_rne_spread_roll_forced_when_expiry_imminent(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol,
    mock_roll
):
    # On the last/second-last day (days_to_roll ≤ FORCE_ROLL_DAYS) the spread
    # roll is called with force=True so a failed limit escalates to market.
    mock_hold.return_value = ("202609", "202612", 1)
    mock_roll.return_value = ("Filled", 2, 0.5)
    held = {"ES": {"202609": 2}}

    reconcile_and_execute(_rne_ib(), MagicMock(), {"ES": 2}, held, _MOCK_DIAG,
                          MagicMock(), execute=True)

    mock_roll.assert_called_once()
    assert mock_roll.call_args.kwargs["force"] is True


@patch("ibkr_fut.live_dynamic.spread_roll_exec")
@_roll_patches
def test_rne_spread_roll_deferred_when_market_closed(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol,
    mock_roll
):
    mock_open.return_value = False                   # exchange closed
    mock_hold.return_value = ("202609", "202612", 2)
    held = {"ES": {"202609": 2}}

    placed, skipped, _, _ = reconcile_and_execute(
        _rne_ib(), MagicMock(), {"ES": 2}, held, _MOCK_DIAG,
        MagicMock(), execute=True)

    mock_roll.assert_not_called()
    assert any("roll (market closed)" in s for s in skipped)


@_roll_patches
def test_rne_risk_gate_blocks_rebalance_not_roll_close(
    mock_spec, mock_hold, mock_qual, mock_ptc, mock_exec, mock_open, mock_vol
):
    # Pathological target trips the vol gate: the rebalance order is skipped but
    # the risk-reducing old-month close still executes.
    mock_hold.return_value = ("202609", None, 9999)
    mock_exec.return_value = _default_fill_result()
    mock_vol.return_value = (False, "too big")
    held = {"ES": {"202603": 2}}                     # stranded old month

    placed, skipped, _, _ = reconcile_and_execute(
        _rne_ib(), MagicMock(), {"ES": 100}, held, _MOCK_DIAG,
        MagicMock(), execute=True, capital=100_000.0)

    assert _exec_orders(mock_exec) == [("202603", "SELL", 2)]
    assert any("risk" in s for s in skipped)


# ── spread_roll_exec unit tests ───────────────────────────────────────────────

def _spread_ib(ticker_bid=0.5, ticker_ask=0.6):
    ib = MagicMock()
    ib.reqMktData.return_value = _ticker(ticker_bid, ticker_ask)
    ib.sleep.return_value = None
    return ib


def _spread_trade(filled_on_limit, status="Cancelled", confirm_cancel=True,
                  ib=None):
    """Limit-trade mock: not done while working; done once cancelled (if
    confirm_cancel). Wire ib.cancelOrder to flip the done flag."""
    trade = MagicMock()
    done = {"v": False}
    trade.isDone.side_effect = lambda: done["v"]
    trade.filled.return_value = filled_on_limit
    trade.orderStatus.status = status
    trade.orderStatus.filled = filled_on_limit
    trade.orderStatus.avgFillPrice = 0.55
    if confirm_cancel and ib is not None:
        ib.cancelOrder.side_effect = lambda o: done.update(v=True)
    return trade


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_force_no_zero_qty_market_order(mock_time, mock_qual):
    # Limit fully filled before the cancel-ack: forced roll must NOT place a
    # zero-quantity market order, and reports the limit's fills.
    mock_time.time.side_effect = [0.0] + [100.0] * 50   # instantly past deadlines
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib()
    trade = _spread_trade(filled_on_limit=5, ib=ib)
    ib.placeOrder.return_value = trade

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=True)

    assert ib.placeOrder.call_count == 1   # limit only, no 0-qty market order
    assert filled == 5


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_force_escalates_remainder_to_market(mock_time, mock_qual):
    # Forced roll: 2 filled on the limit before the confirmed cancel, the
    # remaining 3 go out as a market order; total filled is 5.
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib()
    limit_trade = _spread_trade(filled_on_limit=2, ib=ib)
    mkt_trade = MagicMock()
    mkt_trade.orderStatus.status = "Filled"
    mkt_trade.orderStatus.filled = 3
    mkt_trade.orderStatus.avgFillPrice = 0.60
    ib.placeOrder.side_effect = [limit_trade, mkt_trade]

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=True)

    assert filled == 5
    assert status == "Filled"
    mkt_order = ib.placeOrder.call_args_list[1].args[1]
    assert mkt_order.totalQuantity == 3   # remainder only
    assert mkt_order.orderType == "MKT"


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_no_force_no_market_escalation(mock_time, mock_qual):
    # Early in the spread window (force=False): a timed-out limit is cancelled
    # and the remainder is left for the next cycle — never a market order.
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib()
    trade = _spread_trade(filled_on_limit=2, ib=ib)
    ib.placeOrder.return_value = trade

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=False)

    assert ib.placeOrder.call_count == 1
    assert filled == 2
    assert status == "Cancelled"


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_force_unconfirmed_cancel_no_market(mock_time, mock_qual, capsys):
    # Cancel never confirms: even a forced roll must not market-order on top of
    # a possibly-live limit. Advancing clock so the cancel-wait loop times out.
    clock = {"t": 0.0}
    def _tick():
        clock["t"] += 20.0
        return clock["t"]
    mock_time.time.side_effect = _tick
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib()
    trade = _spread_trade(filled_on_limit=2, confirm_cancel=False, ib=ib)
    ib.placeOrder.return_value = trade

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=True)

    assert ib.placeOrder.call_count == 1
    assert filled == 2
    assert "NOT escalating" in capsys.readouterr().out


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_no_quote_skips_unless_forced(mock_time, mock_qual):
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib(float("nan"), float("nan"))   # no spread quote

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=False)

    assert status == "Unfilled"
    assert filled == 0
    ib.placeOrder.assert_not_called()


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_no_quote_forced_goes_market(mock_time, mock_qual):
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib(float("nan"), float("nan"))
    trade = MagicMock()
    trade.isDone.return_value = True   # market order fills immediately
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 5
    trade.orderStatus.avgFillPrice = 0.60
    ib.placeOrder.return_value = trade

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=True)

    assert status == "Filled"
    assert filled == 5
    order = ib.placeOrder.call_args.args[1]
    assert order.orderType == "MKT"


def _filled_spread_trade(avg_price):
    trade = MagicMock()
    trade.isDone.return_value = True
    trade.orderStatus.status = "Filled"
    trade.orderStatus.filled = 5
    trade.orderStatus.avgFillPrice = avg_price
    return trade


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_long_sells_at_offside_ask(mock_time, mock_qual):
    # Calendar spreads legitimately trade at zero/negative prices, and rolling a
    # long SELLs the spread: the passive limit sits at the ask (offside), so we
    # never pay the spread.
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib(-0.6, -0.4)
    ib.placeOrder.return_value = _filled_spread_trade(-0.4)

    spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612", 5,
                     is_long=True, force=False)

    order = ib.placeOrder.call_args.args[1]
    assert order.orderType == "LMT"
    assert order.action == "SELL"
    assert order.lmtPrice == -0.4


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_short_buys_at_offside_bid(mock_time, mock_qual):
    # Rolling a short BUYs the spread: passive limit at the bid.
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib(-0.6, -0.4)
    ib.placeOrder.return_value = _filled_spread_trade(-0.6)

    spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612", 5,
                     is_long=False, force=False)

    order = ib.placeOrder.call_args.args[1]
    assert order.orderType == "LMT"
    assert order.action == "BUY"
    assert order.lmtPrice == -0.6


@patch("ibkr_fut.live_dynamic.qualify")
@patch("ibkr_fut.live_dynamic.time")
def test_spread_roll_minus_one_sentinel_is_no_quote(mock_time, mock_qual):
    # IB's -1 "no quote" sentinel must not be mistaken for a price.
    mock_time.time.side_effect = [0.0] + [100.0] * 50
    mock_qual.side_effect = [MagicMock(conId=1, currency="USD"),
                             MagicMock(conId=2, currency="USD")]
    ib = _spread_ib(-1.0, -1.0)

    status, filled, _ = spread_roll_exec(ib, _MOCK_SPEC, "202609", "202612",
                                         5, is_long=True, force=False)

    assert status == "Unfilled"
    ib.placeOrder.assert_not_called()


# ── fetch_positions: live re-request, never the stale cache (BUG-7) ───────────

def _ib_run(value=None, raises=None):
    """side_effect for ib.run: consume the wait_for coroutine it's handed (so no
    'coroutine never awaited' warning), then return `value` or raise `raises` —
    modelling how the real ib.run awaits asyncio.wait_for(reqPositionsAsync(), …)."""
    def _run(coro):
        if asyncio.iscoroutine(coro):
            coro.close()
        if raises is not None:
            raise raises
        return value
    return _run


def _ib_run_returns(value):
    return _ib_run(value=value)


def test_fetch_positions_bypasses_stale_cache():
    # ib.positions() is ib_insync's passively-maintained cache, which can freeze
    # after a reconnect. fetch_positions must re-request live via
    # ib.run(asyncio.wait_for(ib.reqPositionsAsync(), …)) and ignore the cache.
    ib = MagicMock()
    ib.run.side_effect = _ib_run_returns(["LIVE"])
    ib.positions.return_value = ["STALE_CACHE"]
    assert fetch_positions(ib) == ["LIVE"]
    assert ib.run.called


def test_fetch_positions_falls_back_to_cache_on_error():
    # A transient API failure degrades to the cache rather than crashing the daemon.
    ib = MagicMock()
    ib.run.side_effect = _ib_run(raises=RuntimeError("api hiccup"))
    ib.positions.return_value = ["FALLBACK"]
    assert fetch_positions(ib) == ["FALLBACK"]


def test_fetch_positions_falls_back_on_timeout():
    # A half-open connection makes reqPositions hang; the bounded wait_for raises
    # TimeoutError and we degrade to the cache rather than hanging the daemon.
    ib = MagicMock()
    ib.run.side_effect = _ib_run(raises=asyncio.TimeoutError())
    ib.positions.return_value = ["CACHE_ON_TIMEOUT"]
    assert fetch_positions(ib) == ["CACHE_ON_TIMEOUT"]


# ── get_positions_by_instr symbol mapping ─────────────────────────────────────

def _pos(symbol, exchange, qty, month="20260918"):
    p = MagicMock()
    p.contract.secType = "FUT"
    p.contract.symbol = symbol
    p.contract.exchange = exchange
    p.contract.primaryExchange = ""
    p.contract.lastTradeDateOrContractMonth = month
    p.position = qty
    return p


def _spec_for(symbol, exchange):
    return {"symbol": symbol, "exchange": exchange, "currency": "USD",
            "multiplier": 5.0, "trading_class": "", "price_magnifier": 1.0}


@patch("ibkr_fut.live_dynamic.ib_spec")
@patch("ibkr_fut.live_dynamic.UNIVERSE", {"ALPHA": "equity", "BETA": "bond"})
def test_positions_symbol_fallback_unique(mock_spec):
    # Exchange doesn't match the config, but only one instrument carries the
    # symbol → symbol-only fallback maps it.
    mock_spec.side_effect = lambda cfg, instr: {
        "ALPHA": _spec_for("MES", "CME"),
        "BETA":  _spec_for("ZB", "CBOT"),
    }[instr]
    ib = MagicMock()
    # get_positions_by_instr now reads positions via fetch_positions, which
    # re-requests live: ib.run(asyncio.wait_for(reqPositionsAsync(), …)) (BUG-7).
    ib.run.side_effect = _ib_run_returns([_pos("MES", "GLOBEX", 2)])
    # No ContractDetails → delivery_month falls back to the month prefix.
    ib.reqContractDetails.return_value = []

    held, unknown = get_positions_by_instr(ib, MagicMock())

    assert held == {"ALPHA": {"202609": 2}}
    assert unknown == []


@patch("ibkr_fut.live_dynamic.ib_spec")
@patch("ibkr_fut.live_dynamic.UNIVERSE", {"ALPHA": "equity", "BETA": "bond"})
def test_positions_symbol_fallback_ambiguous_goes_unknown(mock_spec):
    # Two instruments share the IB symbol on different exchanges: a position on
    # an unmatched exchange must be reported unknown, not silently claimed by
    # whichever instrument came first.
    mock_spec.side_effect = lambda cfg, instr: {
        "ALPHA": _spec_for("FUT", "EUREX"),
        "BETA":  _spec_for("FUT", "LIFFE"),
    }[instr]
    ib = MagicMock()
    ib.run.side_effect = _ib_run_returns([_pos("FUT", "SGX", 1)])

    held, unknown = get_positions_by_instr(ib, MagicMock())

    assert held == {}
    assert unknown == [("FUT", "SGX", 1)]


# ══════════════════════════════════════════════════════════════════════════════
# Group 8 — preflight_check
# ══════════════════════════════════════════════════════════════════════════════

from ibkr_fut.preflight_check import check_contracts, check_market_data


def _cd(tz="US/Central", hours="20260609:1700-20260610:1600"):
    cd = MagicMock()
    cd.timeZoneId = tz
    cd.tradingHours = hours
    return cd


@patch("ibkr_fut.preflight_check.is_contract_okay_to_trade", return_value=True)
@patch("ibkr_fut.preflight_check.qualify")
@patch("ibkr_fut.preflight_check.get_roll_info")
@patch("ibkr_fut.preflight_check.ib_spec")
def test_preflight_all_clear(mock_spec, mock_roll, mock_qual, mock_open):
    mock_spec.return_value = _MOCK_SPEC
    mock_roll.return_value = ("202609", "202612", 200)  # far from roll
    mock_qual.return_value = MagicMock(conId=1)
    ib = MagicMock()
    ib.reqContractDetails.return_value = [_cd()]

    failures, open_now = check_contracts(ib, MagicMock(), {"ES": "equity"})

    assert failures == []
    assert [i for i, _ in open_now] == ["ES"]
    mock_qual.assert_called_once()           # front month only outside roll window


@patch("ibkr_fut.preflight_check.is_contract_okay_to_trade", return_value=False)
@patch("ibkr_fut.preflight_check.qualify")
@patch("ibkr_fut.preflight_check.get_roll_info")
@patch("ibkr_fut.preflight_check.ib_spec")
def test_preflight_qualify_failure_reported(mock_spec, mock_roll, mock_qual, mock_open):
    mock_spec.return_value = _MOCK_SPEC
    mock_roll.return_value = ("202609", None, 9999)
    mock_qual.return_value = None            # ambiguous / unknown contract
    ib = MagicMock()

    failures, open_now = check_contracts(ib, MagicMock(), {"ES": "equity"})

    assert [(i, s) for i, s, _ in failures] == [("ES", "QUALIFY")]
    assert open_now == []


@patch("ibkr_fut.preflight_check.is_contract_okay_to_trade", return_value=False)
@patch("ibkr_fut.preflight_check.qualify")
@patch("ibkr_fut.preflight_check.get_roll_info")
@patch("ibkr_fut.preflight_check.ib_spec")
def test_preflight_checks_next_month_in_roll_window(mock_spec, mock_roll,
                                                    mock_qual, mock_open):
    mock_spec.return_value = _MOCK_SPEC
    mock_roll.return_value = ("202609", "202612", 7)   # inside passive window
    mock_qual.return_value = MagicMock(conId=1)
    ib = MagicMock()
    ib.reqContractDetails.return_value = [_cd()]

    failures, _ = check_contracts(ib, MagicMock(), {"ES": "equity"})

    assert failures == []
    months = [c.args[2] for c in mock_qual.call_args_list]
    assert months == ["202609", "202612"]


@patch("ibkr_fut.preflight_check.is_contract_okay_to_trade", return_value=True)
@patch("ibkr_fut.preflight_check.qualify")
@patch("ibkr_fut.preflight_check.get_roll_info")
@patch("ibkr_fut.preflight_check.ib_spec")
def test_preflight_unmapped_tz_and_empty_hours(mock_spec, mock_roll,
                                               mock_qual, mock_open):
    mock_spec.return_value = _MOCK_SPEC
    mock_roll.return_value = ("202609", None, 9999)
    mock_qual.return_value = MagicMock(conId=1)
    ib = MagicMock()
    ib.reqContractDetails.return_value = [_cd(tz="Mars/Olympus", hours="")]

    failures, open_now = check_contracts(ib, MagicMock(), {"ES": "equity"})

    stages = [s for _, s, _ in failures]
    assert stages == ["HOURS", "HOURS"]      # unmapped tz + empty hours
    assert open_now == []                    # empty hours → never reaches md


def test_preflight_market_data_failure():
    ib = MagicMock()
    ib.reqMktData.side_effect = [_ticker(100.0, 100.25),       # good
                                 _ticker(float("nan"), -1.0)]  # no subscription
    failures = check_market_data(ib, [("GOOD", MagicMock(conId=1)),
                                      ("BAD", MagicMock(conId=2))])

    assert [(i, s) for i, s, _ in failures] == [("BAD", "MKTDATA")]
    assert ib.cancelMktData.call_count == 2


@patch("ibkr_fut.live_dynamic.ib_spec")
@patch("ibkr_fut.live_dynamic.UNIVERSE", {"ALPHA": "equity", "BETA": "bond"})
def test_positions_exact_exchange_match_beats_ambiguity(mock_spec):
    # A (symbol, exchange) match still resolves even when the symbol is shared.
    mock_spec.side_effect = lambda cfg, instr: {
        "ALPHA": _spec_for("FUT", "EUREX"),
        "BETA":  _spec_for("FUT", "LIFFE"),
    }[instr]
    ib = MagicMock()
    ib.run.side_effect = _ib_run_returns([_pos("FUT", "LIFFE", -3)])
    ib.reqContractDetails.return_value = []

    held, unknown = get_positions_by_instr(ib, MagicMock())

    assert held == {"BETA": {"202609": -3}}
    assert unknown == []


# ══════════════════════════════════════════════════════════════════════════════
# Group 9 — watchdog (heartbeat staleness, halt-file guard, alert suppression)
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

from ibkr_fut import watchdog
from ibkr_fut.watchdog import (ALERT_SUPPRESS_SECS, STALE_AFTER_SECS,
                               check_and_act, heartbeat_age_secs, should_alert)


@pytest.fixture
def wd_dir(tmp_path, monkeypatch):
    """Point every watchdog path at a temp dir; mock Discord + restart."""
    monkeypatch.setattr(watchdog, "HEARTBEAT_PATH", tmp_path / "daemon_heartbeat.txt")
    monkeypatch.setattr(watchdog, "HALT_FILE", tmp_path / "risk_halt.txt")
    monkeypatch.setattr(watchdog, "ALERT_MARKER", tmp_path / "watchdog_last_alert")
    discord = MagicMock()
    monkeypatch.setattr(watchdog, "_send_discord", discord)
    restart = MagicMock(return_value=True)
    monkeypatch.setattr(watchdog, "restart_daemon", restart)
    return tmp_path, discord, restart


def _touch_at(path: Path, ts: float):
    path.write_text("hb\n")
    os.utime(path, (ts, ts))


def test_watchdog_fresh_heartbeat_noop(wd_dir):
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.HEARTBEAT_PATH, now - 60)       # 1 min old

    assert check_and_act(now=now) == "fresh"
    discord.assert_not_called()
    restart.assert_not_called()


def test_watchdog_stale_heartbeat_restarts(wd_dir):
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.HEARTBEAT_PATH, now - STALE_AFTER_SECS - 60)

    assert check_and_act(now=now) == "restarted"
    restart.assert_called_once()
    discord.assert_called_once()
    assert watchdog.ALERT_MARKER.exists()              # suppression marker set


def test_watchdog_missing_heartbeat_restarts(wd_dir):
    # First deployment / file deleted: treated like stale → alert + restart.
    tmp, discord, restart = wd_dir
    assert heartbeat_age_secs(now=1_000_000.0) is None

    assert check_and_act(now=1_000_000.0) == "restarted"
    restart.assert_called_once()
    discord.assert_called_once()
    assert "MISSING" in discord.call_args[0][0]


def test_watchdog_halt_file_blocks_restart(wd_dir):
    # CRITICAL: never resurrect a daemon past the kill switch.
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.HEARTBEAT_PATH, now - STALE_AFTER_SECS - 60)
    watchdog.HALT_FILE.write_text("circuit breaker tripped\n")

    assert check_and_act(now=now) == "halt_no_restart"
    restart.assert_not_called()
    discord.assert_called_once()                       # still alerts
    assert "risk_halt.txt" in discord.call_args[0][0]


def test_watchdog_restart_failure_reported(wd_dir):
    tmp, discord, restart = wd_dir
    restart.return_value = False

    assert check_and_act(now=1_000_000.0) == "restart_failed"
    restart.assert_called_once()


def test_watchdog_alert_suppressed_but_restart_retried(wd_dir):
    # Within the 2 h window: no second Discord ping, but restart still attempted.
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.ALERT_MARKER, now - 600)        # alerted 10 min ago
    _touch_at(watchdog.HEARTBEAT_PATH, now - STALE_AFTER_SECS - 60)

    assert check_and_act(now=now) == "restarted"
    discord.assert_not_called()
    restart.assert_called_once()


def test_watchdog_alert_resumes_after_window(wd_dir):
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.ALERT_MARKER, now - ALERT_SUPPRESS_SECS - 60)
    _touch_at(watchdog.HEARTBEAT_PATH, now - STALE_AFTER_SECS - 60)

    assert should_alert(now=now)
    assert check_and_act(now=now) == "restarted"
    discord.assert_called_once()


def test_watchdog_halt_alert_also_suppressed(wd_dir):
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    watchdog.HALT_FILE.write_text("halt\n")
    _touch_at(watchdog.ALERT_MARKER, now - 600)

    assert check_and_act(now=now) == "halt_no_restart"
    discord.assert_not_called()
    restart.assert_not_called()


def test_watchdog_boundary_just_under_threshold_is_fresh(wd_dir):
    tmp, discord, restart = wd_dir
    now = 1_000_000.0
    _touch_at(watchdog.HEARTBEAT_PATH, now - (STALE_AFTER_SECS - 1))

    assert check_and_act(now=now) == "fresh"
    restart.assert_not_called()


def test_touch_heartbeat_writes_and_never_raises(tmp_path, monkeypatch):
    # The daemon-side helper: writes a UTC timestamp; swallows write errors.
    import ibkr_fut.live_dynamic as ld
    hb = tmp_path / "daemon_heartbeat.txt"
    monkeypatch.setattr(ld, "HEARTBEAT_PATH", str(hb))
    ld._touch_heartbeat()
    assert hb.exists() and hb.read_text().strip()

    monkeypatch.setattr(ld, "HEARTBEAT_PATH",
                        str(tmp_path / "no_such_dir" / "hb.txt"))
    ld._touch_heartbeat()                              # must not raise
