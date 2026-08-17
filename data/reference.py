# -*- coding: utf-8 -*-
"""
参考数据:国债利率 / 指数行情 / 行业归属 / 指数权重
==================================================
对应原项目 data_process 的 updata_rf_data / updata_index_ret_data,
以及 data_download的行业 / 指数权重。

rqdatac 实现:
    - 无风险利率:  get_yield_curve(tenor='10Y')  返回小数年化(原项目/100,口径一致)
    - 指数收益率:  get_price_change_rate(指数代码)
    - 行业归属:    get_instrument_industry(source=..., level=1)(个股→行业)
    - 指数成分权重: index_weights(指数代码)

产出(data_store/base/):
    rf.csv         无风险日利率(10Y国债,小数)
    Rt.csv         基准指数(中证全指)日收益率(小数)
    industry_l1.csv 一级行业归属(stock_code, industry_code, industry_name, source;
                   文件名源无关,实际来源见 source 列与运行日志,默认中信 citics_2019)
    index_weights/{指数}.csv  指数成分权重(可选)
"""

import os
import datetime as dt
import pandas as pd

from config import get_settings, get_path
from .client import init_rqdatac, to_rqcode, to_windcode
from . import io


def update_rf_data(start_date="2010-01-01", end_date=None):
    """
    无风险利率:10 年期国债到期收益率。
    rqdatac get_yield_curve 返回【小数】年化,原项目 Tushare/100 也是小数,口径一致。
    产出: data_store/base/rf.csv(index=日期, 列='rf')
    """
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    settings = get_settings()
    tenor = settings["market"]["risk_free_tenor"]   # 默认 '10Y'

    curve = rqdatac.get_yield_curve(start_date, end_date, tenor=tenor, market="cn")
    # curve: index=date, 列为各期限;取 10Y 那一列
    if isinstance(curve, pd.DataFrame):
        rf = curve.iloc[:, [0]].copy()
    else:
        rf = curve.to_frame("rf")
    rf.columns = ["rf"]
    rf = rf.sort_index()

    out_path = os.path.join(get_path("base"), "rf.csv")
    rf.to_csv(out_path)
    print(f"rf.csv: 更新完成 ({len(rf)} 行)")
    return rf


def update_index_ret_data(index_code=None, start_date="2010-01-01", end_date=None):
    """
    基准指数日收益率(中证全指 000985,用于 Beta 因子回归)。
    rqdatac get_price_change_rate 返回小数,原项目 pct_chg/100 也是小数,口径一致。
    产出: data_store/base/Rt.csv(index=日期, 列='Rt')
    """
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    settings = get_settings()
    if index_code is None:
        index_code = settings["market"]["benchmark"]   # '000985.XSHG'

    ret = rqdatac.get_price_change_rate(
        index_code, start_date, end_date, expect_df=False, market="cn"
    )
    # expect_df=False: index=date, 单列(指数收益率)
    if isinstance(ret, pd.DataFrame):
        rt = ret.iloc[:, [0]].copy()
    else:
        rt = ret.to_frame("Rt")
    rt.columns = ["Rt"]
    rt = rt.sort_index()

    out_path = os.path.join(get_path("base"), "Rt.csv")
    rt.to_csv(out_path)
    print(f"Rt.csv: 更新完成 ({len(rt)} 行)")
    return rt


def update_industry(date=None):
    """
    行业归属(每只股票所属行业)。
    用 rqdatac.get_instrument_industry 获取"股票→行业"映射。
    优先用 config.industry_source(默认中信一级 citics_2019),失败则按 industry_fallback 回退。

    rqdatac 的 source 合法值:citics_2019(中信,默认)/ citics / gildata
    申万行业用独立接口 shenwan_instrument_industry(source 参数无效)。

    参数:
        date: 指定日期的行业快照;None 取最新
    产出: data_store/base/industry_l1.csv
        列: stock_code(Wind风格), industry_code, industry_name, source
    """
    rqdatac = init_rqdatac()
    settings = get_settings()
    primary = settings["market"]["industry_source"]               # 'citics_2019'
    fallback = settings["market"].get("industry_fallback", ["sws", "citis", "gildata"])
    sources_to_try = [primary] + [s for s in fallback if s != primary]

    # 获取全部 A 股代码(rqdatac 风格)
    from data.universe import get_stock_list_rq
    from config import get_batch_size
    stock_list_rq = get_stock_list_rq()
    batch_size = get_batch_size()

    def fetch_by_source(source):
        """用指定 source 分批取数,返回拼接后的 DataFrame。"""
        frames = []
        for i in range(0, len(stock_list_rq), batch_size):
            batch = stock_list_rq[i:i + batch_size]
            df = rqdatac.get_instrument_industry(
                batch, source=source, level=1, date=date, market="cn"
            )
            frames.append(df)
        return pd.concat(frames) if frames else pd.DataFrame()

    def fetch_shenwan():
        """申万行业用独立接口。"""
        frames = []
        for i in range(0, len(stock_list_rq), batch_size):
            batch = stock_list_rq[i:i + batch_size]
            df = rqdatac.shenwan_instrument_industry(
                batch, level=1, date=date, market="cn"
            )
            frames.append(df)
        return pd.concat(frames) if frames else pd.DataFrame()

    mapping = None
    used_source = None
    last_err = None
    for source in sources_to_try:
        try:
            if source in ("sws", "sw", "shenwan"):
                raw = fetch_shenwan()
            else:
                raw = fetch_by_source(source)
            if raw is None or raw.empty:
                raise ValueError("返回空数据")
            mapping = raw
            used_source = source
            print(f"行业数据取数成功(使用源:{source},{len(mapping)} 只)")
            break
        except Exception as e:
            last_err = e
            print(f"行业源 {source} 取数失败:{e},尝试下一个源...")
            continue

    if mapping is None:
        raise RuntimeError(f"所有行业源 {sources_to_try} 均取数失败,最后错误:{last_err}")

    # 统一索引为 order_book_id(若已在 index 则重置)
    if mapping.index.name != "order_book_id" and "order_book_id" not in mapping.columns:
        mapping = mapping.reset_index().rename(columns={mapping.index.name or "index": "order_book_id"})
    elif mapping.index.name == "order_book_id":
        mapping = mapping.reset_index()

    # 统一列名:不同源返回列名不同
    #   citics_2019: first_industry_code / first_industry_name
    #   shenwan:     index_code / index_name
    code_col = name_col = None
    for c in ("first_industry_code", "industry_code", "index_code"):
        if c in mapping.columns:
            code_col = c
            break
    for c in ("first_industry_name", "industry_name", "index_name"):
        if c in mapping.columns:
            name_col = c
            break

    if code_col is None or name_col is None:
        raise RuntimeError(f"无法识别行业列名,实际列:{list(mapping.columns)}")

    # 代码转 Wind 风格,便于下游与因子库对齐
    result = pd.DataFrame({
        "stock_code": mapping["order_book_id"].apply(to_windcode),
        "industry_code": mapping[code_col].astype(str),
        "industry_name": mapping[name_col],
        "source": used_source,
    })

    out_path = os.path.join(get_path("base"), "industry_l1.csv")
    result.to_csv(out_path, index=False)
    print(f"industry_l1.csv: 更新完成 ({len(result)} 只,源:{used_source})")
    return result


def update_index_weights(index_code, date=None):
    """
    指数成分股权重(用于持仓暴露分析)。
    rqdatac index_weights(指数代码)。

    ⚠️ 权重单位:rqdatac 历史版本可能返回百分比或小数,落盘前需校验量级。
       下游使用时按 0.05=5% 小数口径处理。

    参数:
        index_code: 指数代码(rqdatac 风格,如 '000300.XSHG')
        date: 指定日期;None 取最新
    产出:
        data_store/base/index_weights/{wind代码}.csv
    """
    rqdatac = init_rqdatac()
    weights = rqdatac.index_weights(index_code, date=date)

    if isinstance(weights, pd.Series):
        df = weights.to_frame("weight")
    else:
        df = weights.copy()
        if "weight" not in df.columns and df.shape[1] == 1:
            df.columns = ["weight"]

    # index 是成分股 order_book_id,转 Wind 风格
    df.index = [to_windcode(c) for c in df.index]
    df.index.name = "stock_code"

    out_dir = os.path.join(get_path("base"), "index_weights")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{to_windcode(index_code)}.csv"
    df.to_csv(os.path.join(out_dir, fname))
    print(f"{fname}: 更新完成 ({len(df)} 只成分股)")
    return df
