# -*- coding: utf-8 -*-
"""
A股日度估值指标(市值/PE/PB/换手率)
==================================
rqdatac 实现(已实测确认,适配 rqdatac 3.4.x):
    - 市值/PE/PB:    get_factor(factor='market_cap'/'pe_ratio'/'pb_ratio'),单 factor 一次
    - 流通市值:      get_factor(factor='a_share_market_val_in_circulation')
    - 换手率:        get_turnover_rate,取 'today' 列(当日换手率,百分比)

单位:rqdatac market_cap / a_share_market_val_in_circulation 单位是【元】,
      原项目 Tushare 的 total_mv / circ_mv 单位是万元并在 data_process 里 ×10000 转元。
      本模块直接落盘 rqdatac 的元单位,不做放大,与原项目最终口径(元)一致。

产出(data_store/base/):
    stock_size.csv      总市值(market_cap,元)
    stock_size_cir.csv  流通市值(a_share_market_val_in_circulation,元)
    stock_turnover.csv  换手率(get_turnover_rate 的 today 列,百分比)
    stock_pe.csv        PE(pe_ratio)
    stock_pb.csv        PB(pb_ratio)
"""

import datetime as dt
import pandas as pd
from tqdm import tqdm

from config import get_batch_size
from .client import init_rqdatac
from . import io, universe

# rqdatac 因子库字段 -> 产出文件名(均用 get_factor 取)
FACTOR_FIELDS = {
    "market_cap": "stock_size.csv",
    "a_share_market_val_in_circulation": "stock_size_cir.csv",
    "pe_ratio": "stock_pe.csv",
    "pb_ratio": "stock_pb.csv",
    # Earnings Yield 分量(ETOP/CETOP 直接用)
    "ep_ratio_ttm": "ep_ratio_ttm.csv",       # 盈市率TTM = 归母净利润TTM/总市值(=ETOP)
    "pcf_ratio_ttm": "pcf_ratio_ttm.csv",     # 经营市现率TTM = 总市值/经营现金流TTM(CETOP取倒数)
}
# 换手率单独处理(get_turnover_rate,取 today 列)
TURNOVER_FILE = "stock_turnover.csv"


def _fetch_turnover_all(rqdatac, stocks, start_date, end_date):
    """用 get_turnover_rate 分批取换手率,返回拼接后的 MultiIndex DataFrame(含 today 列)。"""
    batch_size = get_batch_size()
    frames = []
    for i in tqdm(range(0, len(stocks), batch_size), desc="换手率"):
        batch = stocks[i: i + batch_size]
        try:
            df = rqdatac.get_turnover_rate(
                batch, start_date=start_date, end_date=end_date,
                expect_df=True, market="cn"
            )
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"批次 {i} 取换手率失败: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def update_valuation_data(start_date="2010-01-01", end_date=None):
    """下载全部 A 股日度估值指标,增量更新落盘。"""
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    # 各估值文件 + 换手率文件的最新日期
    latest_dates = []
    existing = {}
    for rq_field, fname in FACTOR_FIELDS.items():
        ld, edf = io.get_latest_date(fname, start_date)
        latest_dates.append(ld)
        existing[fname] = edf
    ld_tur, existing_tur = io.get_latest_date(TURNOVER_FILE, start_date)
    latest_dates.append(ld_tur)
    existing[TURNOVER_FILE] = existing_tur

    latest_date = min(latest_dates)
    if latest_date >= end_date:
        print("估值数据: 已是最新,无需更新")
        return

    stocks = universe.get_stock_list_rq()
    print(f"估值更新区间: {latest_date} ~ {end_date},共 {len(stocks)} 只股票")

    # 逐因子取数并落盘(get_factor)
    for rq_field, fname in FACTOR_FIELDS.items():
        raw = io.fetch_factor_batch(rqdatac, stocks, rq_field,
                                    latest_date, end_date, desc=f"估值-{rq_field}")
        if raw is None or raw.empty:
            print(f"警告: {rq_field} 无数据,跳过 {fname}")
            continue
        io.concat_and_save(fname, io.pivot_multiindex(raw, rq_field), existing[fname])

    # 换手率(get_turnover_rate,取 today 列)
    tur_raw = _fetch_turnover_all(rqdatac, stocks, latest_date, end_date)
    if not tur_raw.empty:
        io.concat_and_save(TURNOVER_FILE, io.pivot_multiindex(tur_raw, "today"),
                           existing[TURNOVER_FILE])
