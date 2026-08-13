# -*- coding: utf-8 -*-
"""
通用数据 IO 工具(data 层叶子模块,仅依赖 config)
================================================
提供 rqdatac 长表 → 日期×股票宽表的透视、增量更新、落盘三类工具。

核心范式(每个数据下载函数都套用):
    latest_date = get_latest_date(filename)                  # 1. 读已有数据最后日期
    df_new = io.fetch_factor_batch(...) + io.pivot_multiindex(...)  # 2. 只取 latest_date 之后的新数据
    io.concat_and_save(filename, df_new, existing_df)        # 3. 新旧合并+去重+落盘
"""

import os
import pandas as pd
from tqdm import tqdm

from config import get_path, get_batch_size


def base_dir():
    """基础宽表目录:data_store/base/"""
    d = get_path("base")
    os.makedirs(d, exist_ok=True)
    return d


def get_latest_date(filename, sdate="2010-01-01"):
    """
    读取已存在文件的最大日期,无文件则返回默认起始日 sdate。

    参数:
        filename: csv 文件名(不含路径),如 "stock_ret.csv"
        sdate: 文件不存在时的默认起始日
    返回:
        (latest_date_str, existing_df)
    """
    filepath = os.path.join(base_dir(), filename)
    if os.path.exists(filepath):
        existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if not existing_data.empty:
            latest_date = existing_data.index.max().strftime("%Y-%m-%d")
        else:
            latest_date = sdate
    else:
        latest_date = sdate
        existing_data = pd.DataFrame()
    return latest_date, existing_data


# 兼容旧调用名(原以下划线开头,现跨模块复用,去掉下划线)
_get_latest_date = get_latest_date


def to_pivot(df, index="date", column="order_book_id", value=None):
    """
    将长表透视为"日期×股票"宽表。同一日期同一股票有多条时保留最新一条。

    参数:
        df: 长表,含 index/column/value 列
        value: 单字段名;None 则保留全部字段(多层列宽表)
    返回:
        DataFrame,index=日期,columns=股票代码
    """
    if value is not None:
        df_item = (
            df.sort_values(by=[index, column, value],
                           ascending=[True, True, False],
                           na_position="last")
            .drop_duplicates(subset=[index, column], keep="first")
        )
        df_pivot = pd.pivot(df_item, index=index, columns=column, values=value)
    else:
        df_pivot = df.pivot(index=index, columns=column)
    return df_pivot.sort_index()


def pivot_multiindex(df, value_col):
    """
    把 rqdatac 的 MultiIndex(order_book_id, date/tradedate) 长表透视为 日期×股票 宽表。
    兼容不同接口返回的日期列名(get_factor 是 date,get_turnover_rate 是 tradedate)。
    """
    df = df.reset_index()
    # 统一日期列名为 date
    date_candidates = [c for c in df.columns
                       if c and str(c).lower() in ("date", "tradedate", "datetime")]
    if date_candidates:
        df = df.rename(columns={date_candidates[0]: "date"})
    return to_pivot(df, index="date", column="order_book_id", value=value_col)


def fetch_factor_batch(rqdatac, stocks, factor, start_date, end_date, desc=None):
    """
    用 get_factor 分批取单个因子,返回拼接后的 MultiIndex DataFrame。

    参数:
        rqdatac: 已 init 的 rqdatac 模块
        stocks: rqdatac 风格代码列表
        factor: 因子名(如 'market_cap')
        start_date, end_date: 日期范围
        desc: 进度条描述
    """
    batch_size = get_batch_size()
    frames = []
    for i in tqdm(range(0, len(stocks), batch_size), desc=desc or factor):
        batch = stocks[i: i + batch_size]
        try:
            df = rqdatac.get_factor(
                batch, factor=factor,
                start_date=start_date, end_date=end_date,
                expect_df=True, market="cn"
            )
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"批次 {i} 取 {factor} 失败: {e}")
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def merge_and_save(filepath, df_new, existing_df):
    """
    通用合并去重落盘:新数据与已有数据纵向拼接,按索引(日期)去重保留最后一条。

    data/io.concat_and_save 与 factors/common.data_concat_and_save 共同委托此函数,
    消除两处重复的 pd.concat + groupby(index).last() 逻辑。
    """
    if df_new is None or (hasattr(df_new, "empty") and df_new.empty):
        return existing_df
    df = pd.concat([existing_df, df_new], axis=0)
    df = df.groupby(df.index).last()
    df.to_csv(filepath)
    return df


def concat_and_save(filename, df_new, existing_df=None):
    """
    将新数据与已有数据合并,按日期去重后落盘(基础数据层入口)。

    参数:
        filename: csv 文件名(如 "stock_ret.csv")
        df_new: 新数据宽表(index=日期),可为空
        existing_df: 已有数据;None 时内部读取
    返回:
        合并后的完整 DataFrame
    """
    filepath = os.path.join(base_dir(), filename)
    if existing_df is None:
        _, existing_df = get_latest_date(filename)

    if df_new is None or df_new.empty:
        print(f"{filename}: 无需更新")
        return existing_df

    df = merge_and_save(filepath, df_new, existing_df)
    print(f"{filename}: 更新完成 ({df.shape[0]} 行)")
    return df


def load_base(filename):
    """读取一张基础宽表(index_col=0, parse_dates)。"""
    filepath = os.path.join(base_dir(), filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"基础数据文件不存在: {filepath}(请先运行 data 层下载)")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
