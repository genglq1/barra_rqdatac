# -*- coding: utf-8 -*-
"""
Growth 因子:egro / sgro / egibs / egibs_s
==========================================
Barra CNE-5 原文:Growth = 0.47·SGRO + 0.24·EGRO + 0.18·EGIBS + 0.11·EGIBS_s

历史增长(年报回归):
    EGRO:过去 5 年每股收益(EPS)对年份 OLS 回归的斜率 ÷ EPS 均值
    SGRO:过去 5 年营业收入对年份 OLS 回归的斜率 ÷ 营收均值
        ⚠️ SGRO 口径说明:Barra 原文用"每股营收"(sales per share),rqdatac 无现成字段,
        本实现用"营收总额"(operating_revenue)。slope÷mean 在股本不变时与每股口径等价;
        A 股送转/增发频繁,股本变动股票会混入股本扩张成分、产生偏差。

分析师预测增长(一致预期):
    EGIBS  :一致预期净利润增长率(长期 T+3)— 对应原文 Long-term Predicted Earnings Growth
    EGIBS_s:一致预期净利润增长率(短期 FTM)— 对应原文 Short-term Predicted Earnings Growth
        ⚠️ rqdatac 只有"净利润增长率"无"EPS 增长率",用净利润增长代理 earnings growth
        (股本变动时有偏差,数据固有限制无法消除)。

数据来源:历史用 get_pit_financials_ex 取近5年年报(Q4);预测用 consensus 一致预期字段。
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_batch_size, get_path
from data.client import init_rqdatac
from data import universe
from factors import common as F

VERSION = "cne5"

# Growth 因子参数
N_YEARS = 5          # 回归年数(取近5年年报)
ANN_QUARTER = "q4"   # 年报对应季度

# 年报本地缓存(get_pit_financials_ex 联网取数消耗 Quota,缓存避免重复下载)
ANNUAL_CACHE_FILE = "annual_reports.csv"

# 一致预期净利润增长率字段(rqdatac)→ 对应 Barra 描述因子
EGIBS_FIELD   = "comp_con_net_profit_growth_ratio_t3"   # 长期(T+3)→ EGIBS
EGIBS_S_FIELD = "comp_con_net_profit_growth_ratio_ftm"  # 短期(FTM)→ EGIBS_s


def _fetch_annual_reports(start_quarter, end_quarter):
    """
    用 get_pit_financials_ex 取全市场近5年年报(Q4),过滤去重。

    带本地缓存:data_store/base/annual_reports.csv。若缓存覆盖所需年份区间则直接复用
    (零联网);否则联网补齐后与缓存合并(按 order_book_id+year 去重,保留联网最新值)并落盘。
    联网失败且缓存存在时,退回缓存中可用部分并告警。

    返回:DataFrame,列为 [order_book_id, year, basic_earnings_per_share, operating_revenue]
          每只股票每年一行(年报),已按公告日去重保留最新。
    """
    start_year = int(start_quarter[:4])
    end_year = int(end_quarter[:4])

    cache_path = os.path.join(get_path("base"), ANNUAL_CACHE_FILE)
    cached = None
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 50:
        cached = pd.read_csv(cache_path)
        if (not cached.empty and cached["year"].min() <= start_year
                and cached["year"].max() >= end_year):
            print(f"成长因子-年报缓存命中({ANNUAL_CACHE_FILE}),"
                  f"覆盖 {cached['year'].min()}~{cached['year'].max()},跳过联网")
            return cached[(cached["year"] >= start_year)
                          & (cached["year"] <= end_year)].reset_index(drop=True)

    rqdatac = init_rqdatac()
    stocks = universe.get_stock_list_rq()
    batch_size = get_batch_size()
    fields = ["basic_earnings_per_share", "operating_revenue"]

    frames = []
    for i in tqdm(range(0, len(stocks), batch_size), desc="成长因子-取年报"):
        batch = stocks[i: i + batch_size]
        try:
            df = rqdatac.get_pit_financials_ex(
                batch, fields=fields,
                start_quarter=start_quarter, end_quarter=end_quarter,
                statements="all", market="cn",
            )
            if df is not None and not df.empty:
                frames.append(df.reset_index())
        except Exception as e:
            print(f"批次 {i} 取年报失败: {e}")
            continue

    if not frames:
        # 联网失败:退回缓存中可用部分(如有),否则返回空
        if cached is not None and not cached.empty:
            print(f"⚠️  年报联网取数失败,退回缓存 {ANNUAL_CACHE_FILE} 中可用年份")
            return cached[(cached["year"] >= start_year)
                          & (cached["year"] <= end_year)].reset_index(drop=True)
        print("⚠️  年报联网取数失败且无缓存,egro/sgro 无法计算")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    # 仅保留年报(Q4)
    all_df = all_df[all_df["quarter"].str.endswith(ANN_QUARTER)].copy()
    # 去重:同股票同季度保留公告日(info_date)最新的一条
    all_df = all_df.sort_values("info_date").drop_duplicates(
        ["order_book_id", "quarter"], keep="last"
    )
    # quarter -> 年份
    all_df["year"] = all_df["quarter"].str[:4].astype(int)
    fresh = all_df[["order_book_id", "year", "basic_earnings_per_share", "operating_revenue"]]

    # 与缓存合并落盘(联网数据在 concat 后部,去重 keep="last" 保证联网最新值优先)
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(["order_book_id", "year"], keep="last")
    else:
        merged = fresh
    merged.to_csv(cache_path, index=False)
    print(f"年报缓存已更新:{cache_path}({merged['year'].min()}~{merged['year'].max()})")

    return merged[(merged["year"] >= start_year)
                  & (merged["year"] <= end_year)].reset_index(drop=True)


def _load_predicted_growth(field, latest_date, end_date):
    """
    读取一致预期净利润增长率字段(已是增长率,无需回归),标准化后作为 egibs/egibs_s。

    rqdatac growth_ratio 单位不统一(可能百分比或小数),此处加量级自检:
        若截面|值|的中位数 > 1,判为百分比、/100 转小数;否则保持。

    参数:
        field: rqdatac 字段名(如 comp_con_net_profit_growth_ratio_t3)
    返回:
        标准化后的宽表 DataFrame;字段缺失/空则返回空 DataFrame。
    """
    import os
    path = os.path.join(get_path("base"), f"{field}.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 50:
        print(f"⚠️  {field} 不存在或为空,跳过对应预测增长分量(需先在 data 层下载)")
        return pd.DataFrame()

    df = F.load_base(f"{field}.csv").loc[latest_date:end_date]
    if df.empty or df.dropna(how="all").empty:
        print(f"⚠️  {field} 区间内无数据,跳过")
        return pd.DataFrame()

    # 量级自检:百分比(中位数>1)→ /100 转小数
    if np.nanmedian(np.abs(df.values)) > 1.0:
        df = df / 100.0

    df = F._standardize_with_weights(df, True)
    return df


def cal_growth_factor(start_date="2010-01-01", end_date=None):
    """
    计算成长因子 egro / sgro(历史,年报回归)+ egibs / egibs_s(分析师预测,一致预期增长率)。

    Barra CNE-5:Growth = 0.47·SGRO + 0.24·EGRO + 0.18·EGIBS + 0.11·EGIBS_s
    - EGRO/SGRO:5 年年报 EPS/营收 OLS 回归斜率 ÷ 因变量均值
    - EGIBS/EGIBS_s:一致预期净利润增长率(长期 T+3 / 短期 FTM)
    """
    if end_date is None:
        end_date = F.get_ndate()

    factor_name_list = ["egro", "sgro", "egibs", "egibs_s"]
    latest_date, existing_dict = F._get_existing_df(VERSION, factor_name_list, start_date)

    # ============ Part A: egro/sgro 历史增长(年报回归)============
    egro = pd.DataFrame()
    sgro = pd.DataFrame()
    trade_cal = F.get_trade_cal()

    end_year = int(end_date[:4])
    # EGRO/SGRO 为滚动时序:每个年报年用回溯5年算,映射到该年生效日(次年4/30),
    # 前向填充到下一年报。故需取足够历史年报供滑动窗口(多取 10 年,覆盖因子有效起始前几年)
    start_q = f"{end_year - N_YEARS - 6}q4"
    end_q = f"{end_year}q4"

    annual = _fetch_annual_reports(start_q, end_q)
    if annual.empty:
        print("⚠️  egro/sgro 跳过:无年报数据")
    else:
        egro_by_year = {}   # {年报年份: {股票: egro值}}
        sgro_by_year = {}
        for code, grp in tqdm(annual.groupby("order_book_id"), desc="成长因子-年报回归"):
            grp = grp.sort_values("year")
            if len(grp) < N_YEARS:
                continue   # 不足5年,跳过
            years_all = grp["year"].values.astype(float)
            eps_all = grp["basic_earnings_per_share"].values.astype(float)
            rev_all = grp["operating_revenue"].values.astype(float)
            # 滑动 5 年窗口:每个窗口末端年算一个 EGRO/SGRO(滚动时序),
            # 映射到该窗口末端年的生效日(次年4/30),前向填充到下一年报
            for start in range(len(grp) - N_YEARS + 1):
                years = years_all[start:start + N_YEARS]
                eps = eps_all[start:start + N_YEARS]
                rev = rev_all[start:start + N_YEARS]
                if np.isnan(eps).any() or np.isnan(rev).any():
                    continue
                # OLS 斜率:Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
                x_centered = years - years.mean()
                denom = (x_centered ** 2).sum()
                if denom == 0:
                    continue
                slope_eps = (x_centered * (eps - eps.mean())).sum() / denom
                slope_rev = (x_centered * (rev - rev.mean())).sum() / denom
                # Barra 口径:斜率 ÷ 因变量均值(增长率)
                eps_mean = eps.mean()
                rev_mean = rev.mean()
                end_window_year = int(years[-1])   # 窗口末端年(生效日=次年4/30)
                egro_by_year.setdefault(end_window_year, {})[code] = (
                    slope_eps / eps_mean if eps_mean != 0 else np.nan
                )
                sgro_by_year.setdefault(end_window_year, {})[code] = (
                    slope_rev / rev_mean if rev_mean != 0 else np.nan
                )

        if egro_by_year:
            # 年报值映射到披露日(次年4/30)→ 前向填充到日频
            daily_index = trade_cal[latest_date:end_date].index

            def _to_daily(by_year_dict):
                df = pd.DataFrame(by_year_dict).T   # index=年报年份, columns=股票
                df.index = df.index.map(lambda y: pd.Timestamp(y + 1, 4, 30))
                out = pd.DataFrame(index=daily_index)
                out["on"] = out.index.map(
                    lambda d: pd.Timestamp(d.year - 1, 4, 30) if d.month <= 4
                              else pd.Timestamp(d.year, 4, 30)
                )
                out = pd.merge(out, df, how="left", left_on="on", right_index=True)
                return out.drop(columns=["on"])

            egro = F._standardize_with_weights(_to_daily(egro_by_year), True)
            sgro = F._standardize_with_weights(_to_daily(sgro_by_year), True)
        else:
            print("⚠️  egro/sgro 跳过:无足够5年年报的股票")

    # ============ Part B: egibs/egibs_s 分析师预测增长(一致预期)============
    egibs   = _load_predicted_growth(EGIBS_FIELD,   latest_date, end_date)
    egibs_s = _load_predicted_growth(EGIBS_S_FIELD, latest_date, end_date)

    concat_dict = {"egro": egro, "sgro": sgro, "egibs": egibs, "egibs_s": egibs_s}
    F.data_concat_and_save(VERSION, existing_dict, concat_dict)
