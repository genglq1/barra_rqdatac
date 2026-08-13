# -*- coding: utf-8 -*-
"""
股票池 + 交易日历
================
- update_stock_universe():  全 A 股股票清单(all_instruments)
- update_trade_cal():       沪深交易日历(get_trading_dates)
- get_stock_list():         读取已落盘的当前在市股票列表(Wind 风格代码)
"""

import os
import datetime as dt
import pandas as pd

from config import get_path
from .client import init_rqdatac, to_windcode
from . import io


def update_stock_universe(date=None):
    """
    获取全 A 股股票清单。

    参数:
        date: 指定日期(只返回当日存续的合约);None 取最新
    落盘:
        data_store/base/all_instruments.csv
        列: order_book_id, symbol(中文名), abbrev_symbol, sector_code,
            exchange, listed_date, de_listed_date, status
    """
    rqdatac = init_rqdatac()
    inst = rqdatac.all_instruments(type="CS", date=date, market="cn")
    # 转为 DataFrame(all_instruments 返回 list[Instrument],转 df 更稳)
    if not isinstance(inst, pd.DataFrame):
        records = []
        for x in inst:
            records.append({
                "order_book_id": getattr(x, "order_book_id", None),
                "symbol": getattr(x, "symbol", None),
                "abbrev_symbol": getattr(x, "abbrev_symbol", None),
                "sector_code": getattr(x, "sector_code", None),
                "exchange": getattr(x, "exchange", None),
                "listed_date": getattr(x, "listed_date", None),
                "de_listed_date": getattr(x, "de_listed_date", None),
                "status": getattr(x, "status", None),
            })
        inst = pd.DataFrame(records)

    out_path = os.path.join(get_path("base"), "all_instruments.csv")
    os.makedirs(get_path("base"), exist_ok=True)
    inst.to_csv(out_path, index=False)
    print(f"all_instruments.csv: 更新完成 ({len(inst)} 只)")
    return inst


def update_trade_cal(start_date="2010-01-01", end_date=None):
    """
    获取沪深交易日历。

    参数:
        start_date: 起始日
        end_date: 截止日,None 取今天
    落盘:
        data_store/base/trade_cal.csv
        结构:index=trade_date,含 year/quarter/month/week 等派生列(兼容原项目口径)
    """
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    dates = rqdatac.get_trading_dates(start_date, end_date, market="cn")
    cal = pd.DataFrame({"trade_date": pd.to_datetime(dates)})
    cal = cal.set_index("trade_date").sort_index()

    # 派生年/季/月/周(原项目 trade_cal 含这些列,下游可能用)
    cal["year"] = cal.index.year
    cal["quarter"] = cal.index.quarter
    cal["month"] = cal.index.month
    cal["week"] = cal.index.isocalendar().week

    out_path = os.path.join(get_path("base"), "trade_cal.csv")
    os.makedirs(get_path("base"), exist_ok=True)
    cal.to_csv(out_path)
    print(f"trade_cal.csv: 更新完成 ({len(cal)} 个交易日)")
    return cal


def _load_instruments(active_only=True):
    """
    读取已落盘的股票清单(all_instruments.csv)。
    注意:该表是普通数据表(非日期宽表),不能用 io.load_base(它会强制 index_col+parse_dates),
          故直接用 pd.read_csv。
    返回:
        DataFrame,含 order_book_id / status 等列
    """
    import os
    from config import get_path
    filepath = os.path.join(get_path("base"), "all_instruments.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"股票清单不存在: {filepath}(请先运行 data.universe.update_stock_universe())"
        )
    df = pd.read_csv(filepath)
    # 兼容:若 order_book_id 被当作 index(早期落盘格式),重置回来
    if "order_book_id" not in df.columns and df.index.name == "order_book_id":
        df = df.reset_index()
    if active_only and "status" in df.columns:
        df = df[df["status"] == "Active"]
    return df


def get_stock_list(active_only=True):
    """
    读取已落盘的股票清单,返回 Wind 风格代码列表(.SH/.SZ/.BJ)。
    用于与原项目因子库口径对齐。

    参数:
        active_only: True 仅返回在市(status=='Active')的股票
    返回:
        list[str],如 ['600519.SH', '000001.SZ', ...]
    """
    df = _load_instruments(active_only=active_only)
    # rqcode -> windcode,便于与因子计算(原项目口径)对齐
    return [to_windcode(c) for c in df["order_book_id"].tolist()]


def get_stock_list_rq(active_only=True):
    """
    读取已落盘的股票清单,返回 rqdatac 风格代码列表(.XSHG/.XSHE/.XBJG)。
    用于直接传给 rqdatac API(get_price / get_instrument_industry 等)。

    参数:
        active_only: True 仅返回在市(status=='Active')的股票
    返回:
        list[str],如 ['600519.XSHG', '000001.XSHE', ...]
    """
    df = _load_instruments(active_only=active_only)
    return df["order_book_id"].tolist()
