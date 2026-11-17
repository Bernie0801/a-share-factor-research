"""
A 股日線數據下載與整理（項目第一步）

用法（在項目根目錄下運行）：
    python -m src.data_loader --universe csi800     # 中證 800 當前成分股，約半小時
    python -m src.data_loader --universe all        # 全市場 + 已退市股票，約幾個小時，建議晚上掛著跑
    python -m src.data_loader --combine-only        # 不下載，只把已下載的文件合併成面板數據

輸出：
    data/raw/universe_<name>.csv            股票池（記錄當天的成分股名單）
    data/raw/daily_<adjust>/<code>.parquet  每隻股票一個文件，已存在就跳過（斷點續傳）
    data/raw/index/<code>.parquet           基準指數日線（滬深300 / 中證500 / 中證800）
    data/raw/trade_calendar.csv             交易日曆
    data/processed/daily_<adjust>.parquet   合併後的長表，每行是 (date, code) 一條記錄

adjust 的含義：
    hfq  = 後復權，用來計算收益率
    none = 不復權，是真實成交價，用來判斷漲跌停、價格水平等
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

INDEX_CODES = {"csi300": "000300", "csi500": "000905", "csi800": "000906"}
BENCHMARKS = ["000300", "000905", "000906"]

# AKShare 返回中文列名，統一換成英文，寫代碼更方便
COLUMN_MAP = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",      # 單位：手（1 手 = 100 股）
    "成交额": "amount",      # 單位：元
    "振幅": "amplitude",     # 單位：%
    "涨跌幅": "pct_chg",     # 單位：%
    "涨跌额": "chg",
    "换手率": "turnover",    # 單位：%
}
STANDARD_COLUMNS = ["date", "code", "open", "high", "low", "close",
                    "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]
ADJUST_ARG = {"hfq": "hfq", "qfq": "qfq", "none": ""}


# ----------------------------------------------------------------------------
# 股票池
# ----------------------------------------------------------------------------
def get_universe(name: str) -> pd.DataFrame:
    """返回股票池，列為 code, name, status（listed / delisted）。"""
    if name in INDEX_CODES:
        df = ak.index_stock_cons_csindex(symbol=INDEX_CODES[name])
        out = pd.DataFrame({
            "code": df["成分券代码"].astype(str).str.zfill(6),
            "name": df["成分券名称"],
            "status": "listed",
        })
    elif name == "all":
        listed = ak.stock_info_a_code_name()
        frames = [pd.DataFrame({
            "code": listed["code"].astype(str).str.zfill(6),
            "name": listed["name"],
            "status": "listed",
        })]
        # 已退市股票：不加進來就會有倖存者偏差
        try:
            sh = ak.stock_info_sh_delist(symbol="全部")
            frames.append(pd.DataFrame({
                "code": sh["公司代码"].astype(str).str.zfill(6),
                "name": sh["公司简称"],
                "status": "delisted",
            }))
        except Exception as e:  # 交易所網站偶爾會改版
            print(f"[警告] 上交所退市名單獲取失敗：{e!r}")
        try:
            sz = ak.stock_info_sz_delist(symbol="终止上市公司")
            frames.append(pd.DataFrame({
                "code": sz["证券代码"].astype(str).str.zfill(6),
                "name": sz["证券简称"],
                "status": "delisted",
            }))
        except Exception as e:
            print(f"[警告] 深交所退市名單獲取失敗：{e!r}")
        out = pd.concat(frames, ignore_index=True).drop_duplicates("code", keep="first")
        # 只保留滬深 A 股：6 開頭是上交所，0/3 開頭是深交所。
        # 北交所和 B 股的數據接口不同，第一版先不處理。
        out = out[out["code"].str[0].isin(["0", "3", "6"])]
    else:
        raise ValueError(f"未知股票池 {name}，可選：{list(INDEX_CODES)} 或 all")
    return out.sort_values("code").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 單隻股票下載
# ----------------------------------------------------------------------------
def _hist_func():
    """AKShare 新版本叫 stock_zh_a_hist_em，舊版本叫 stock_zh_a_hist，兩個都兼容。"""
    for name in ("stock_zh_a_hist_em", "stock_zh_a_hist"):
        func = getattr(ak, name, None)
        if func is not None:
            return func
    raise ImportError("找不到 AKShare 的日線接口，請運行 pip install -U akshare")


def standardize(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """中文列名轉英文、日期轉 datetime、按日期排序。"""
    df = df.rename(columns=COLUMN_MAP)
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in STANDARD_COLUMNS if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)


def download_one(code: str, start: str, end: str, adjust: str,
                 retries: int = 3) -> pd.DataFrame:
    """下載一隻股票的日線。網絡出錯時重試，每次等待時間翻倍（指數退避）。"""
    func = _hist_func()
    for attempt in range(1, retries + 1):
        try:
            df = func(symbol=code, period="daily", start_date=start,
                      end_date=end, adjust=ADJUST_ARG[adjust])
            if df is None or df.empty:
                return pd.DataFrame(columns=STANDARD_COLUMNS)
            return standardize(df, code)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return pd.DataFrame(columns=STANDARD_COLUMNS)  # 不會執行到這裡


def download_stocks(codes: list[str], start: str, end: str, adjust: str,
                    sleep: float = 0.3, refresh: bool = False) -> dict:
    """批量下載。已有文件默認跳過，所以中斷後重新運行同一條命令即可續傳。"""
    out_dir = RAW / f"daily_{adjust}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skipped": 0, "empty": [], "failed": []}

    for code in tqdm(codes, desc=f"下載日線 ({adjust})"):
        path = out_dir / f"{code}.parquet"
        if path.exists() and not refresh:
            stats["skipped"] += 1
            continue
        try:
            df = download_one(code, start, end, adjust)
        except Exception as e:
            stats["failed"].append({"code": code, "error": repr(e)[:200]})
            continue
        # 空數據也寫一個空文件，下次就不會重複請求
        df.to_parquet(path, index=False)
        if df.empty:
            stats["empty"].append(code)
        else:
            stats["ok"] += 1
        # 每次請求之間隨機停一下，避免請求太密被數據源限流
        time.sleep(sleep + random.uniform(0, sleep))

    if stats["failed"]:
        pd.DataFrame(stats["failed"]).to_csv(RAW / f"failed_{adjust}.csv", index=False)
    print(f"[{adjust}] 新下載 {stats['ok']}，跳過已有 {stats['skipped']}，"
          f"空數據 {len(stats['empty'])}，失敗 {len(stats['failed'])}")
    if stats["failed"]:
        print(f"  失敗名單在 data/raw/failed_{adjust}.csv，重新運行同一條命令即可補下")
    return stats


# ----------------------------------------------------------------------------
# 指數、交易日曆
# ----------------------------------------------------------------------------
def download_benchmarks(start: str, end: str) -> None:
    out_dir = RAW / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in BENCHMARKS:
        try:
            df = ak.index_zh_hist_em(symbol=code, period="daily",
                                     start_date=start, end_date=end)
            standardize(df, code).to_parquet(out_dir / f"{code}.parquet", index=False)
        except Exception as e:
            print(f"[警告] 指數 {code} 下載失敗：{e!r}")


def download_calendar() -> None:
    try:
        cal = ak.calendar_trade_date_hist_sina()
        cal.to_csv(RAW / "trade_calendar.csv", index=False)
    except Exception as e:
        print(f"[警告] 交易日曆下載失敗：{e!r}")


# ----------------------------------------------------------------------------
# 合併成面板數據
# ----------------------------------------------------------------------------
def combine(adjust: str) -> pd.DataFrame:
    """把每隻股票的文件合併成一張長表：每行是 (date, code)。"""
    files = sorted((RAW / f"daily_{adjust}").glob("*.parquet"))
    frames = [f for f in (pd.read_parquet(p) for p in files) if not f.empty]
    if not frames:
        print(f"[{adjust}] 沒有可合併的數據")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel = (panel.drop_duplicates(["date", "code"])
                  .sort_values(["date", "code"])
                  .reset_index(drop=True))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"daily_{adjust}.parquet"
    panel.to_parquet(out, index=False)
    print(f"[{adjust}] 合併完成：{panel['code'].nunique()} 隻股票，"
          f"{panel['date'].min():%Y-%m-%d} 至 {panel['date'].max():%Y-%m-%d}，"
          f"共 {len(panel):,} 行 → {out}")
    return panel


# ----------------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="下載 A 股日線數據")
    p.add_argument("--universe", default="csi800",
                   choices=[*INDEX_CODES, "all"], help="股票池")
    p.add_argument("--start", default="20150101", help="開始日期 YYYYMMDD")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"),
                   help="結束日期 YYYYMMDD，默認今天")
    p.add_argument("--adjust", default="hfq,none",
                   help="復權方式，逗號分隔：hfq / qfq / none")
    p.add_argument("--sleep", type=float, default=0.3, help="請求間隔秒數")
    p.add_argument("--refresh", action="store_true",
                   help="忽略已有文件、全部重新下載（更新數據時用）")
    p.add_argument("--combine-only", action="store_true",
                   help="只合併已下載的文件，不聯網")
    args = p.parse_args(argv)
    adjusts = [a.strip() for a in args.adjust.split(",") if a.strip()]

    if not args.combine_only:
        RAW.mkdir(parents=True, exist_ok=True)
        universe = get_universe(args.universe)
        universe.to_csv(RAW / f"universe_{args.universe}.csv", index=False)
        print(f"股票池 {args.universe}：{len(universe)} 隻")

        download_calendar()
        download_benchmarks(args.start, args.end)
        for adj in adjusts:
            download_stocks(universe["code"].tolist(), args.start, args.end,
                            adj, sleep=args.sleep, refresh=args.refresh)

    for adj in adjusts:
        combine(adj)


if __name__ == "__main__":
    main()
