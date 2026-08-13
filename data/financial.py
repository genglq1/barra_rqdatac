# -*- coding: utf-8 -*-
"""
A股财务数据(资产负债表/现金流量表/利润表)
==========================================
rqdatac 3.4.x 仅 get_pit_financials_ex 支持利润表字段;资产负债表、现金流量表字段
在因子库 get_factor 中提供(基于最新已披露财报填充到日频,已是 PIT 对齐)。
故本模块统一用 get_factor 取所有财务字段。

字段映射(rqdatac 因子库字段 -> 原项目文件名):
    资产负债表:
        total_assets                        -> total_assets.csv
        oth_eqt_tools_p_shr                 -> oth_eqt_tools_p_shr.csv(优先股,因子库可能不提供)
        total_liabilities                   -> total_liab.csv
        non_current_liabilities             -> total_ncl.csv
        equity_parent_company               -> total_hldr_eqy_exc_min_int.csv
    现金流量表:
        cash_flow_from_operating_activities -> n_cashflow_act.csv
    利润表:
        basic_earnings_per_share            -> basic_eps.csv
        operating_revenue                   -> revenue_ps.csv
            (rqdatac 无 revenue_per_share,用营收总额;growth 的 sgro 用斜率/均值,结果等价)
"""

import datetime as dt

from .client import init_rqdatac
from . import io, universe

# rqdatac 因子库字段 -> 产出文件名
FIN_FIELDS = {
    # 资产负债表
    "total_assets": "total_assets.csv",
    "oth_eqt_tools_p_shr": "oth_eqt_tools_p_shr.csv",
    "total_liabilities": "total_liab.csv",
    "non_current_liabilities": "total_ncl.csv",
    "equity_parent_company": "total_hldr_eqy_exc_min_int.csv",
    # 现金流量表
    "cash_flow_from_operating_activities": "n_cashflow_act.csv",
    # 利润表
    "basic_earnings_per_share": "basic_eps.csv",
    "operating_revenue": "revenue_ps.csv",
}


def update_financials(start_date="2005-01-01", end_date=None):
    """
    下载全部财务字段(资产负债表/现金流/利润表),增量更新落盘。

    注:用 get_factor 取日频全量(基于最新已披露财报填充),leverage/earnings/growth
        因子各自按需做季度对齐,故此处统一存日频,不再单独过滤年报。
    """
    rqdatac = init_rqdatac()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    # 各字段文件的最新日期(取最小值作为续算起点)
    latest_dates = []
    existing = {}
    for rq_field, fname in FIN_FIELDS.items():
        ld, edf = io.get_latest_date(fname, start_date)
        latest_dates.append(ld)
        existing[fname] = edf

    latest_date = min(latest_dates)
    if latest_date >= end_date:
        print("财务数据: 已是最新,无需更新")
        return

    stocks = universe.get_stock_list_rq()
    print(f"财务更新区间: {latest_date} ~ {end_date},共 {len(stocks)} 只股票")

    # 逐字段取数并落盘
    for rq_field, fname in FIN_FIELDS.items():
        raw = io.fetch_factor_batch(rqdatac, stocks, rq_field,
                                    latest_date, end_date, desc=f"财务-{rq_field}")
        if raw is None or raw.empty:
            print(f"警告: {rq_field} 无数据(因子库可能不提供此字段),跳过 {fname}")
            continue
        df_new = io.pivot_multiindex(raw, rq_field)
        io.concat_and_save(fname, df_new, existing[fname])
