# -*- coding: utf-8 -*-
"""
A股行情数据(后复权)
===================
对应原项目 data_process.updata_price_stock_data,产出 7 张行情宽表。

rqdatac 实现:
    - 后复权价:get_price(adjust_type='post')
    - 涨跌幅:  get_price_change_rate()(基于后复权)
    注:get_price 默认不含 pct_chg,涨跌幅必须单独取。

产出(data_store/base/):
    stock_ret.csv     涨跌幅(rqdatac 返回小数,无需 ×0.01)
    stock_open.csv    开盘价(后复权)
    stock_high.csv    最高价
    stock_low.csv     最低价
    stock_close.csv   收盘价
    stock_vol.csv     成交量
    stock_amount.csv  成交额(rqdatac 字段 total_turnover)
"""

import datetime as dt
import pandas as pd
from tqdm import tqdm

from config import get_batch_size
from .client import init_rqdatac
from . import io, universe


# rqdatac 行情字段 -> 产出文件名映射(后复权,供 Beta/Momentum 等需连续收益率的因子用)
_PRICE_FIELDS = {
    "open": "stock_open.csv",
    "high": "stock_high.csv",
    "low": "stock_low.csv",
    "close": "stock_close.csv",
    "volume": "stock_vol.csv",
    "total_turnover": "stock_amount.csv",
}

# 不复权收盘价:供 EPIBS 等"截面比值"因子用。
# EPIBS = 一致预期EPS(基于当前股本)/ 价格,须用当前真实价;
# 后复权 close 会随除权累积放大,系统性低估老股 EPIBS,故单独取不复权价。
UNADJUSTED_CLOSE_FILE = "stock_close_unadjusted.csv"


def update_price_data(start_date="2010-01-01", end_date=None, batch_size=None):
    """
    下载全部 A 股后复权行情 + 涨跌幅,增量更新落盘。

    参数:
        start_date: 起始日(文件存在时自动从最新日续算)
        end_date: 截止日,None 取今天
        batch_size: rqdatac 单次请求股票数;None 时读 config(settings.yaml rqdatac.batch_size)
    产出:
        data_store/base/ 下的 7 张行情 csv
    """
    if batch_size is None:
        batch_size = get_batch_size()
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    # 各行情文件的最新日期(取最小值作为续算起点)
    latest_dates = []
    existing = {}
    for field, fname in _PRICE_FIELDS.items():
        ld, edf = io.get_latest_date(fname, start_date)
        latest_dates.append(ld)
        existing[fname] = edf
    # 涨跌幅单列
    ld_ret, existing_ret = io.get_latest_date("stock_ret.csv", start_date)
    latest_dates.append(ld_ret)
    existing["stock_ret.csv"] = existing_ret
    # 不复权收盘价(EPIBS 用)
    ld_unadj, existing_unadj = io.get_latest_date(UNADJUSTED_CLOSE_FILE, start_date)
    latest_dates.append(ld_unadj)
    existing[UNADJUSTED_CLOSE_FILE] = existing_unadj

    latest_date = min(latest_dates)
    if latest_date >= end_date:
        print("行情数据: 已是最新,无需更新")
        return

    # 股票池(用 rqdatac 风格代码请求)
    stock_list_rq = universe.get_stock_list_rq()
    print(f"行情更新区间: {latest_date} ~ {end_date},共 {len(stock_list_rq)} 只股票")

    # ---- 分批请求行情价 ----
    # get_price(expect_df=True) 返回 MultiIndex(order_book_id, date) 行,列为行情字段
    price_frames = []
    for i in tqdm(range(0, len(stock_list_rq), batch_size), desc="行情价"):
        batch = stock_list_rq[i: i + batch_size]
        df = rqdatac.get_price(
            batch, start_date=latest_date, end_date=end_date,
            frequency="1d", adjust_type="post", expect_df=True, market="cn"
        )
        if df is not None and not df.empty:
            price_frames.append(df)

    # ---- 分批请求涨跌幅 ----
    # get_price_change_rate(expect_df=True) 返回宽表:index=date, columns=order_book_id
    ret_frames = []
    for i in tqdm(range(0, len(stock_list_rq), batch_size), desc="涨跌幅"):
        batch = stock_list_rq[i: i + batch_size]
        df = rqdatac.get_price_change_rate(
            batch, start_date=latest_date, end_date=end_date,
            expect_df=True, market="cn"
        )
        if df is not None and not df.empty:
            ret_frames.append(df)

    # ---- 行情价:MultiIndex 转长表再透视 ----
    if price_frames:
        price_all = pd.concat(price_frames)
        # 重置 MultiIndex 为普通列(order_book_id, date)
        price_all = price_all.reset_index()
        # 列名应为 order_book_id / date / 各行情字段
        date_col = "date"
        code_col = "order_book_id"

        # 落盘各行情字段(rqdatac 价已是后复权,单位元,无需换算)
        for field, fname in _PRICE_FIELDS.items():
            if field not in price_all.columns:
                print(f"警告: rqdatac 未返回字段 {field},跳过 {fname}")
                continue
            df_new = io.to_pivot(price_all, index=date_col, column=code_col, value=field)
            io.concat_and_save(fname, df_new, existing[fname])

    # ---- 不复权收盘价(EPIBS 用,单独请求 adjust_type="none")----
    # 后复权 close 随除权累积放大,不能用作 EPIBS 分母;不复权 close = 当日真实价
    unadj_frames = []
    for i in tqdm(range(0, len(stock_list_rq), batch_size), desc="不复权收盘价"):
        batch = stock_list_rq[i: i + batch_size]
        try:
            df = rqdatac.get_price(
                batch, start_date=latest_date, end_date=end_date,
                frequency="1d", adjust_type="none", expect_df=True, market="cn"
            )
            if df is not None and not df.empty:
                unadj_frames.append(df)
        except Exception as e:
            print(f"批次 {i} 取不复权价失败: {e}")
            continue
    if unadj_frames:
        unadj_all = pd.concat(unadj_frames).reset_index()
        if "close" in unadj_all.columns:
            df_new = io.to_pivot(unadj_all, index=date_col, column=code_col, value="close")
            io.concat_and_save(UNADJUSTED_CLOSE_FILE, df_new, existing[UNADJUSTED_CLOSE_FILE])
        else:
            print(f"警告: 不复权价未返回 close 字段,跳过 {UNADJUSTED_CLOSE_FILE}")
    else:
        print(f"警告: 不复权价无数据,跳过 {UNADJUSTED_CLOSE_FILE}")

    # ---- 涨跌幅:已是宽表,直接合并落盘 ----
    # rqdatac 返回小数,原项目 Tushare 的 pct_chg/100 也是小数,口径一致
    if ret_frames:
        # 各批返回的宽表按列(股票)合并
        ret_all = pd.concat(ret_frames, axis=1)
        # 去除可能重复的列(同一股票在多批中出现)
        ret_all = ret_all.loc[:, ~ret_all.columns.duplicated()]
        io.concat_and_save("stock_ret.csv", ret_all, existing["stock_ret.csv"])
