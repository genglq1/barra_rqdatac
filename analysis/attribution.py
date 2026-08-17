# -*- coding: utf-8 -*-
"""
收益归因(基于净值)
==================
读取基金日度净值序列(样例见 data_store/净值表/)+ 因子收益率 f_ret.csv,
用 252 日滚动窗口对净值收益回归因子收益率,逐日分解
Alpha / 国家 / 行业 / 风格 贡献(每日四项之和 = 当日净值收益)。

方法(原项目口径):
    滚动窗口内 OLS(无截距):r_t = Σ β_i · f_i,t + ε_t
    当日归因:Alpha_t = ε_t(当日残差),各因子组贡献 = Σ_{i∈组} β_i · f_i,t

净值文件格式(自动适配列名变体):
    日期列:净值日期/日期/估值日期(字符串或日期均可)
    净值列:优先"单位净值"(备选:复权净值/累计净值);
            单位≠累计(有分红)时提示并改用累计净值近似

CLI:
    python -m analysis.attribution "data_store/净值表/xxx.xlsx" [--window 252] [--show]

历史修复:
    - β 列分配由 label 索引 betas[0]/betas[1:n](对 f_ret 列名 country/10/11/... 会
      KeyError)改为位置索引 iloc;
    - 因子收益由 read_excel 改为 read_csv(f_ret.csv 为 CSV);
    - 净值多列(单位/累计)输入时提取单列,不再整表 pct_change;
    - 绘图默认保存 PNG 不弹窗(--show 弹窗)。
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # 默认无界面渲染(--show 时切回弹窗)
import matplotlib.pyplot as plt
import statsmodels.api as sm

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]       # 作图显示中文
matplotlib.rcParams["axes.unicode_minus"] = False

from config import get_model_dir, get_path


# ----------------------------------------------------------------------------
# 净值提取
# ----------------------------------------------------------------------------

def load_nav(nav_file):
    """
    从净值序列 Excel/CSV 提取日度净值。

    参数:
        nav_file: 净值文件路径(xlsx/xls/csv,表头在第 0 行;
                  样例格式:产品名称/产品代码/净值日期/累计净值/单位净值)
    返回:
        Series(index=DatetimeIndex 升序, value=净值)
    """
    if str(nav_file).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(nav_file, header=0)
    else:
        raw = pd.read_csv(nav_file, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    # ---- 日期列 ----
    date_col = next((c for c in ("净值日期", "日期", "估值日期", "trade_date", "date")
                     if c in raw.columns), None)
    if date_col is None:
        raise ValueError(f"未找到日期列(候选:净值日期/日期/估值日期),实际列:{list(raw.columns)}")

    # ---- 净值列:优先单位净值;单位≠累计(有分红)时改用累计净值近似 ----
    nav_col = next((c for c in ("单位净值", "复权净值", "累计净值", "净值")
                    if c in raw.columns), None)
    if nav_col is None:
        raise ValueError(f"未找到净值列(候选:单位净值/复权净值/累计净值/净值),实际列:{list(raw.columns)}")
    if nav_col == "单位净值" and "累计净值" in raw.columns:
        unit = pd.to_numeric(raw["单位净值"], errors="coerce")
        cum = pd.to_numeric(raw["累计净值"], errors="coerce")
        if (unit - cum).abs().max() > 1e-9:
            print("⚠️  单位净值 ≠ 累计净值(疑似有分红/拆分),改用累计净值近似(含分红再投资口径)")
            nav_col = "累计净值"

    nav = pd.Series(pd.to_numeric(raw[nav_col], errors="coerce").values,
                    index=pd.to_datetime(raw[date_col]), name="nav")
    nav = nav.dropna()
    n_dup = nav.index.duplicated().sum()
    if n_dup:
        print(f"⚠️  剔除重复日期 {n_dup} 行(保留最后一条)")
        nav = nav[~nav.index.duplicated(keep="last")]
    nav = nav.sort_index()
    print(f"净值提取: {os.path.basename(nav_file)} [{nav.index.min().date()} ~ "
          f"{nav.index.max().date()}] 共 {len(nav)} 个净值日(列:{nav_col})")
    return nav


# ----------------------------------------------------------------------------
# 归因计算
# ----------------------------------------------------------------------------

def perform_return_attribution(nav, factor_ret, window=252, factor_num=10):
    """
    滚动窗口收益归因:净值收益对 [country, 行业..., 风格...] 回归,逐日分解贡献。

    参数:
        nav (Series): 基金日度净值(index=日期)
        factor_ret (DataFrame): 因子收益(index=日期,列顺序 [country, 行业..., 风格...])
        window (int): 滚动回归窗口(交易日数,默认 252)
        factor_num (int): 风格因子个数(列末 N 个)
    返回:
        (attribution, style_detail):
        attribution - DataFrame(index=归因日, 列=[Alpha, Country, Industry, Style]),
                      每日四项之和 = 当日净值收益(残差恒等式)
        style_detail - DataFrame(index=归因日, 列=各风格因子),逐因子当日贡献 β_i·f_i,t
    """
    if isinstance(nav, pd.DataFrame):
        nav = nav.iloc[:, 0]                       # 兼容多列输入(取首列)
    fund_returns = nav.pct_change().dropna()

    common_index = fund_returns.index.intersection(factor_ret.index)
    fund_returns = fund_returns.loc[common_index]
    factor_ret = factor_ret.loc[common_index]
    if len(common_index) <= window:
        raise ValueError(f"净值与因子收益交集仅 {len(common_index)} 日,不足窗口 {window} 日"
                         f"(f_ret 起始日 {factor_ret.index.min().date()},"
                         f"请检查净值区间或调小 --window)")

    n_total = factor_ret.shape[1]
    # 分组基于原始列序(第1列 country,末 factor_num 列风格,中间行业);
    # 窗口内被剔除的列(如预热期 momentum)从各组交集自然移除,避免位置错位
    style_all = list(factor_ret.columns[n_total - factor_num:])
    industry_all = list(factor_ret.columns[1: n_total - factor_num])

    rows = {}
    style_rows = {}
    for i in range(window - 1, len(fund_returns)):
        end_date = fund_returns.index[i]
        fund_window = fund_returns.iloc[i - window + 1: i + 1]
        fr_window = factor_ret.iloc[i - window + 1: i + 1]

        # 窗口内全空的因子列剔除(如 momentum 预热期),保证 OLS 可逆
        valid_cols = fr_window.columns[fr_window.notna().any()]
        X = fr_window[valid_cols]
        if X.isna().any().any():
            mask = X.notna().all(axis=1)
            X, fund_window = X[mask], fund_window[mask]
            if fund_window.index[-1] != end_date:
                continue               # 当日行被剔除,残差非"当日"Alpha,跳过

        model = sm.OLS(fund_window, X)
        results = model.fit()

        betas = results.params                      # Series,index=因子列名
        alpha = float(results.resid.iloc[-1])       # 当日残差 = r_t - Σβ·f_t
        fr_today = factor_ret.loc[end_date]

        style_cols = [c for c in style_all if c in X.columns]
        industry_cols = [c for c in industry_all if c in X.columns]
        country = (betas["country"] * fr_today["country"]
                   if "country" in betas else np.nan)
        industry = float((betas[industry_cols] * fr_today.reindex(industry_cols)).sum())
        style = float((betas[style_cols] * fr_today.reindex(style_cols)).sum())

        rows[end_date] = {"Alpha": alpha, "Country": country,
                          "Industry": industry, "Style": style}
        style_rows[end_date] = betas[style_cols] * fr_today.reindex(style_cols)

    attribution = pd.DataFrame.from_dict(rows, orient="index")
    attribution.index.name = "date"
    style_detail = pd.DataFrame(style_rows).T
    style_detail.index.name = "date"
    return attribution, style_detail


def plot_cumulative_returns(attribution, nav, out_path=None, show=False):
    """
    归因结果与净值的累计收益对比图(默认保存 PNG,--show 弹窗)。
    返回累计贡献 DataFrame(index=日期, 列=净值/Alpha/Country/Industry/Style)。
    """
    cum = (1 + attribution).cumprod() - 1
    common = attribution.index
    nav_cum = (nav.loc[common].pct_change().dropna() + 1).cumprod() - 1

    plt.figure(figsize=(12, 8))
    plt.plot(nav_cum.index, nav_cum.values, label="净值累计收益", linewidth=2)
    labels = {"Alpha": "Alpha(残差)", "Country": "国家", "Industry": "行业", "Style": "风格"}
    for col in attribution.columns:
        plt.plot(cum.index, cum[col].values, label=labels.get(col, col), linestyle="--")
    plt.legend()
    plt.title("累计收益归因")
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"归因图已保存: {out_path}")
    if show:
        plt.show()
    plt.close()
    return cum


# 风格因子图中文名(缺失时回退英文列名)
STYLE_LABELS = {
    "size": "市值", "beta": "贝塔", "momentum": "动量", "non_linear_size": "非线性市值",
    "book_to_price": "账面市值比", "residual_volatility": "残差波动",
    "liquidity": "流动性", "earnings_yield": "盈利收益率", "growth": "成长",
    "leverage": "杠杆",
}


def plot_style_returns(style_detail, out_path=None, show=False):
    """
    风格因子累计收益贡献图:10 个风格因子各自 β_i·f_i 的累计曲线 + 风格合计。
    返回累计贡献 DataFrame。
    """
    cum = (1 + style_detail).cumprod() - 1
    total = (1 + style_detail.sum(axis=1)).cumprod() - 1

    plt.figure(figsize=(12, 8))
    for col in style_detail.columns:
        plt.plot(cum.index, cum[col].values,
                 label=STYLE_LABELS.get(col, col), linewidth=1.2)
    plt.plot(total.index, total.values, label="风格合计",
             color="black", linewidth=2.2, linestyle="--")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.legend(ncol=2)
    plt.title("风格因子累计收益贡献")
    plt.xlabel("日期")
    plt.ylabel("累计贡献")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"风格归因图已保存: {out_path}")
    if show:
        plt.show()
    plt.close()
    return cum


def run(nav_file, factor_return_file=None, window=252, factor_num=10,
        save=True, show=False):
    """
    一键运行收益归因:净值文件 + f_ret.csv -> 归因明细/累计贡献/PNG。

    参数:
        nav_file: 净值序列 Excel/CSV 路径
        factor_return_file: 因子收益 CSV(默认 data_store/model/cne5/f_ret.csv)
        window, factor_num: 同 perform_return_attribution
        save: True 时归因明细/累计贡献 CSV 与 PNG 存 data_store/analysis/
        show: True 时弹窗显示图表
    返回:
        (attribution, cum): 归因明细与累计贡献 DataFrame
    """
    nav = load_nav(nav_file)
    if factor_return_file is None:
        factor_return_file = os.path.join(get_model_dir("cne5"), "f_ret.csv")
    factor_ret = pd.read_csv(factor_return_file, index_col=0, parse_dates=True)

    attribution, style_detail = perform_return_attribution(
        nav, factor_ret, window=window, factor_num=factor_num)
    print(f"归因区间: {attribution.index.min().date()} ~ {attribution.index.max().date()}"
          f"(窗口 {window} 日,共 {len(attribution)} 个归因日)")

    # 数值恒等式自检:每日 Alpha+Country+Industry+Style ≈ 当日净值收益
    fund_ret = nav.pct_change().reindex(attribution.index)
    identity_err = (attribution.sum(axis=1) - fund_ret).abs().max()
    print(f"恒等式校验 |Σ贡献 - 当日净值收益| 最大误差: {identity_err:.2e}")
    if identity_err > 1e-8:
        print("⚠️  归因分解与净值收益偏差超阈值,请检查数据对齐")

    cum = plot_cumulative_returns(attribution, nav, show=show,
                                  out_path=None if not save else
                                  os.path.join(_analysis_dir(), _out_base(nav_file) + "_归因.png"))
    style_cum = plot_style_returns(style_detail, show=show,
                                   out_path=None if not save else
                                   os.path.join(_analysis_dir(), _out_base(nav_file) + "_风格归因.png"))

    if save:
        out_dir = _analysis_dir()
        attribution.to_csv(os.path.join(out_dir, _out_base(nav_file) + "_归因明细.csv"),
                           encoding="utf-8-sig")
        cum.to_csv(os.path.join(out_dir, _out_base(nav_file) + "_累计贡献.csv"),
                   encoding="utf-8-sig")
        style_detail.to_csv(os.path.join(out_dir, _out_base(nav_file) + "_风格贡献明细.csv"),
                            encoding="utf-8-sig")
        print(f"归因明细已保存: {os.path.join(out_dir, _out_base(nav_file) + '_归因明细.csv')}")

    # 年化贡献摘要(日均值 × 242)
    n_days = len(attribution)
    ann = attribution.mean() * 242
    print("\n年化贡献摘要(日均值×242):")
    for k, v in ann.items():
        print(f"  {k:10s} {v:+.2%}")
    print(f"  合计       {(ann.sum()):+.2%}(对照净值年化 "
          f"{fund_ret.mean() * 242:+.2%})")
    ann_style = style_detail.mean() * 242
    print("\n年化风格贡献排序:")
    for k, v in ann_style.sort_values(ascending=False).items():
        print(f"  {STYLE_LABELS.get(k, k)}({k}) {v:+.2%}")
    return attribution, cum


def _analysis_dir():
    """归因产出目录:data_store/analysis/"""
    d = os.path.join(get_path("data_store"), "analysis")
    os.makedirs(d, exist_ok=True)
    return d


def _out_base(nav_file):
    """产出文件名基名(净值文件名去扩展名)。"""
    return os.path.splitext(os.path.basename(nav_file))[0]


def main():
    parser = argparse.ArgumentParser(description="基于净值的收益归因(CNE-5 因子收益率)")
    parser.add_argument("nav_file", help="净值序列 Excel/CSV 路径")
    parser.add_argument("--window", type=int, default=252, help="滚动回归窗口(默认 252)")
    parser.add_argument("--factor-returns", default=None,
                        help="因子收益 CSV(默认 data_store/model/cne5/f_ret.csv)")
    parser.add_argument("--show", action="store_true", help="弹窗显示归因图(默认只保存 PNG)")
    args = parser.parse_args()

    run(args.nav_file, factor_return_file=args.factor_returns,
        window=args.window, save=True, show=args.show)


if __name__ == "__main__":
    main()
