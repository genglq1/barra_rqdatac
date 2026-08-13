# -*- coding: utf-8 -*-
"""
收益归因(基于净值)
==================
迁移自原 收益归因(基于净值).py。
用 252 日滚动窗口,对基金净值收益回归因子收益率,分解 Alpha/国家/行业/风格贡献。

改进:
    - 桌面绝对路径改为函数参数传入(由调用方决定数据来源)
    - 列顺序约定:[国家因子, 行业因子..., 风格因子...],factor_num=风格因子个数
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]       # 作图显示中文
matplotlib.rcParams["axes.unicode_minus"] = False


def perform_return_attribution(nav, factor_ret, window=252, factor_num=10):
    """
    进行收益归因分析。

    参数:
        nav (DataFrame): 基金净值,index=日期
        factor_ret (DataFrame): 因子收益,index=日期,列顺序为
                                [国家因子, 行业因子..., 风格因子...]
        window (int): 回归窗口,默认 252
        factor_num (int): 风格因子个数(末 N 列)

    返回:
        dict: {日期: {Alpha, Country_Attribution, Industry_Attribution, Style_Attribution}}
    """
    fund_returns = nav.pct_change().dropna()

    common_index = fund_returns.index.intersection(factor_ret.index)
    fund_returns = fund_returns.loc[common_index]
    factor_ret = factor_ret.loc[common_index]

    attribution_results = {}

    for end_date in fund_returns.index:
        if end_date < fund_returns.index[window - 1]:
            continue   # 窗口长度不足,跳过

        start_date = fund_returns.index[fund_returns.index.get_loc(end_date) - window + 1]
        fund_returns_window = fund_returns.loc[start_date:end_date]
        factor_ret_window = factor_ret.loc[start_date:end_date]

        X = factor_ret_window
        y = fund_returns_window

        model = sm.OLS(y, X)
        results = model.fit()

        betas = results.params
        alpha = results.resid.iloc[-1]

        # 列分配:country(1) + industry + style(末 factor_num)
        beta_country = betas[0]
        beta_industry = betas[1:factor_ret.shape[1] - factor_num]
        beta_style = betas[factor_ret.shape[1] - factor_num:]

        country_attribution = beta_country * factor_ret.loc[end_date, factor_ret.columns[0]]
        industry_attribution = factor_ret.loc[end_date, factor_ret.columns[1:factor_ret.shape[1] - factor_num]].mul(beta_industry).sum()
        style_attribution = factor_ret.loc[end_date, factor_ret.columns[factor_ret.shape[1] - factor_num:]].mul(beta_style).sum()

        attribution_results[end_date] = {
            "Alpha": alpha,
            "Country_Attribution": country_attribution,
            "Industry_Attribution": industry_attribution,
            "Style_Attribution": style_attribution,
        }

    return attribution_results


def plot_cumulative_returns(attribution_results, nav):
    """将收益归因结果转为累计收益并绘制折线图。"""
    attribution_df = pd.DataFrame(attribution_results).T

    common_index = attribution_df.index.intersection(nav.index)
    attribution_df = attribution_df.loc[common_index]
    nav = nav.loc[common_index]

    cumulative_returns = (attribution_df + 1).cumprod() - 1
    nav_cumulative_returns = (nav.pct_change().dropna() + 1).cumprod() - 1

    plt.figure(figsize=(12, 8))
    plt.plot(nav_cumulative_returns.index, nav_cumulative_returns.values,
             label="NAV Cumulative Returns", linewidth=2)
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column].values,
                 label=column, linestyle="--")
    plt.legend()
    plt.title("Cumulative Returns Attribution")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.show()


def run(nav_file, factor_return_file, window=252, factor_num=10):
    """
    一键运行收益归因(从文件读入)。

    参数:
        nav_file: 基金净值 Excel/CSV 路径(由调用方传入,不再硬编码桌面路径)
        factor_return_file: 因子收益 Excel/CSV 路径(可用 data_store/model/{version}/f_ret.csv)
        window, factor_num: 同 perform_return_attribution
    """
    nav_data = pd.read_excel(nav_file, index_col=0, parse_dates=True)
    factor_ret_data = pd.read_excel(factor_return_file, index_col=0, parse_dates=True)

    attribution_results = perform_return_attribution(
        nav_data, factor_ret_data, window=window, factor_num=factor_num
    )
    plot_cumulative_returns(attribution_results, nav_data)
    return attribution_results
