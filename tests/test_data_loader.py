"""
用假數據測試 data_loader，不需要聯網。運行：pytest
"""
import pandas as pd
import pytest

from src import data_loader as dl


def fake_hist(symbol, period, start_date, end_date, adjust):
    """模擬 AKShare 日線接口：返回 3 天數據，列名和真實接口一致。"""
    if symbol == "000002":
        return pd.DataFrame()  # 模擬沒有數據的股票
    dates = pd.date_range("2024-01-02", periods=3, freq="B").date
    return pd.DataFrame({
        "日期": dates, "股票代码": symbol,
        "开盘": [10.0, 10.5, 11.0], "收盘": [10.5, 11.0, 10.8],
        "最高": [10.6, 11.2, 11.1], "最低": [9.9, 10.4, 10.7],
        "成交量": [1000, 1200, 900], "成交额": [1e6, 1.3e6, 9.7e5],
        "振幅": [7.0, 7.6, 3.6], "涨跌幅": [5.0, 4.8, -1.8],
        "涨跌额": [0.5, 0.5, -0.2], "换手率": [1.0, 1.2, 0.9],
    })


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "RAW", tmp_path / "raw")
    monkeypatch.setattr(dl, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)

    calls = {"hist": 0, "flaky": 0}

    def hist(**kw):
        calls["hist"] += 1
        if kw["symbol"] == "000003":  # 第一次失敗，第二次成功：測試重試
            calls["flaky"] += 1
            if calls["flaky"] == 1:
                raise ConnectionError("模擬網絡錯誤")
        if kw["symbol"] == "000004":  # 一直失敗：測試失敗名單
            raise ConnectionError("模擬網絡錯誤")
        return fake_hist(**kw)

    monkeypatch.setattr(dl, "_hist_func", lambda: hist)
    monkeypatch.setattr(dl.ak, "index_stock_cons_csindex", lambda symbol: pd.DataFrame({
        "成分券代码": ["1", "2", "3", "4"],
        "成分券名称": ["平安银行", "万科A", "測試3", "測試4"],
    }), raising=False)
    monkeypatch.setattr(dl.ak, "index_zh_hist_em",
                        lambda **kw: fake_hist(symbol=kw["symbol"], period="daily",
                                               start_date="", end_date="", adjust=""),
                        raising=False)
    monkeypatch.setattr(dl.ak, "calendar_trade_date_hist_sina",
                        lambda: pd.DataFrame({"trade_date": pd.date_range("2024-01-01", periods=5).date}),
                        raising=False)
    return tmp_path, calls


def test_full_run(env):
    tmp, calls = env
    dl.main(["--universe", "csi300", "--start", "20240101", "--end", "20240110",
             "--adjust", "hfq", "--sleep", "0"])

    panel = pd.read_parquet(tmp / "processed" / "daily_hfq.parquet")
    # 000001 和 000003（重試後成功）有數據；000002 空；000004 失敗
    assert sorted(panel["code"].unique()) == ["000001", "000003"]
    assert len(panel) == 6
    assert list(panel.columns) == dl.STANDARD_COLUMNS
    assert panel["date"].is_monotonic_increasing

    failed = pd.read_csv(tmp / "raw" / "failed_hfq.csv", dtype={"code": str})
    assert failed["code"].tolist() == ["000004"]
    assert (tmp / "raw" / "index" / "000300.parquet").exists()
    assert (tmp / "raw" / "trade_calendar.csv").exists()


def test_resume_skips_existing(env):
    tmp, calls = env
    args = ["--universe", "csi300", "--adjust", "hfq", "--sleep", "0"]
    dl.main(args)
    first = calls["hist"]
    dl.main(args)
    # 第二次運行只會重試上次失敗的 000004（重試 3 次），其他都跳過
    assert calls["hist"] - first == 3


def test_combine_only_needs_no_network(env, monkeypatch):
    tmp, _ = env
    dl.main(["--universe", "csi300", "--adjust", "hfq", "--sleep", "0"])
    monkeypatch.setattr(dl, "get_universe", lambda name: pytest.fail("不應聯網"))
    dl.main(["--combine-only", "--adjust", "hfq"])
    assert (tmp / "processed" / "daily_hfq.parquet").exists()
