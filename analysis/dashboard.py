# -*- coding: utf-8 -*-
"""
Barra CNE-5 可视化面板(Streamlit)
================================
启动(项目根目录):
    streamlit run analysis/dashboard.py

三个 Tab:
    1. 因子收益:每日因子收益率(f_ret.csv)的累计收益/月度热力图/滚动波动/绩效汇总
    2. 持仓暴露:估值表 -> 10 大风格因子暴露(可叠加基准指数对比)
    3. 净值归因:净值序列 -> Alpha/国家/行业/风格归因 + 风格因子拆分

计算层复用 analysis/exposure.py 与 analysis/attribution.py(已在 CLI 验证),
本面板只做交互与渲染;重计算用 st.cache_data 缓存(cne_5 全量加载/归因 OLS)。
"""

import os
import sys
import tempfile

# streamlit 以脚本目录(analysis/)为运行路径,注入项目根以导入 config/data/analysis
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config import get_model_dir, get_path
from analysis import exposure as EX
from analysis import attribution as AT
from analysis.attribution import STYLE_LABELS

st.set_page_config(page_title="Barra CNE-5 风险模型面板", layout="wide",
                   page_icon="📊")


# ----------------------------------------------------------------------------
# 数据加载(缓存)
# ----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="加载风格因子 cne_5.csv(约 760 万行)...")
def load_factor(version="cne5"):
    return EX._load_factor(version)


@st.cache_data(ttl=600)
def load_factor_returns(version="cne5"):
    path = os.path.join(get_model_dir(version), "f_ret.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data(ttl=600)
def load_industry_names():
    path = os.path.join(get_path("base"), "industry_l1.csv")
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path)[["industry_code", "industry_name"]].drop_duplicates()
    return dict(zip(m["industry_code"].astype(str), m["industry_name"]))


@st.cache_data(ttl=600)
def load_benchmark():
    """基准指数日收益(中证全指 Rt.csv);country 纯因子与其对比,差异=风格中性化调整。"""
    path = os.path.join(get_path("base"), "Rt.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["Rt"]


def _list_files(folder, patterns=(".xlsx", ".xls", ".csv")):
    full = os.path.join(get_path("data_store"), folder)
    if not os.path.isdir(full):
        return [], full
    files = sorted(f for f in os.listdir(full) if f.endswith(patterns))
    return [os.path.join(full, f) for f in files], full


def _save_upload(upfile):
    """上传文件落临时盘(计算函数接受路径)。"""
    suffix = os.path.splitext(upfile.name)[1] or ".xlsx"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(upfile.getvalue())
    tmp.close()
    return tmp.name


# ----------------------------------------------------------------------------
# Tab 1:每日因子收益
# ----------------------------------------------------------------------------

def tab_factor_returns(f_ret):
    style_cols = list(f_ret.columns[-10:])
    industry_cols = [c for c in f_ret.columns[1:-10]]
    ind_names = load_industry_names()

    with st.sidebar:
        st.subheader("因子收益筛选")
        date_strs = [d.strftime("%Y-%m-%d") for d in f_ret.index]
        default_start = date_strs[max(0, len(date_strs) - 485)]   # 默认近2年(约485交易日)
        col_a, col_b = st.columns(2)
        with col_a:
            d_start = st.selectbox("起始日期", date_strs,
                                   index=date_strs.index(default_start))
        with col_b:
            d_end = st.selectbox("结束日期", date_strs, index=len(date_strs) - 1)
        if d_start > d_end:
            d_start, d_end = d_end, d_start
            st.warning("起始晚于结束,已自动调换")
        group = st.radio("因子组", ["风格因子", "行业因子", "市场因子(country)"], horizontal=True)
        chart = st.radio("图表", ["累计收益", "月度收益热力图", "滚动年化波动"],
                         horizontal=True)
        if group == "风格因子":
            default = style_cols
            options = style_cols
            labels = {c: f"{STYLE_LABELS.get(c, c)}({c})" for c in style_cols}
        elif group == "行业因子":
            default = industry_cols
            options = industry_cols
            labels = {c: f"{ind_names.get(c, c)}({c})" for c in industry_cols}
        else:
            default = ["country"]
            options = ["country"]
            labels = {"country": "country(市场因子)"}
        sel = st.multiselect("因子(可多选)", options, default=default,
                             format_func=lambda c: labels.get(c, c))
        overlay_bench = (chart == "累计收益" and load_benchmark() is not None
                         and st.checkbox("叠加中证全指对照(Rt)", value=True))

    if not sel:
        st.info("请在左侧选择至少一个因子")
        return
    sub = f_ret.loc[d_start:d_end, sel]

    if chart == "累计收益":
        cum = (1 + sub).cumprod() - 1
        fig = go.Figure()
        for c in sel:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines",
                                     name=labels.get(c, c)))
        if overlay_bench:
            bench = load_benchmark().loc[d_start:d_end]
            bench_cum = (1 + bench).cumprod() - 1
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values,
                                     name="中证全指(基准)",
                                     line=dict(dash="dash", color="grey", width=2)))
        fig.update_layout(title=f"因子累计收益({d_start} ~ {d_end})",
                          yaxis_title="累计收益", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        if overlay_bench and "country" in sel:
            st.caption("注:country 为纯市场因子(√市值加权+风格/行业中性),"
                       "与基准的差 = √市值加权偏离 + 风格暴露调整项,属 Barra 口径特征。")
    elif chart == "月度收益热力图":
        monthly = sub.resample("ME").sum()
        monthly.index = monthly.index.strftime("%Y-%m")
        fig = go.Figure(go.Heatmap(
            z=monthly.values.T, x=list(monthly.index),
            y=[labels.get(c, c) for c in sel],
            colorscale="RdYlGn", zmid=0, texttemplate=".1%",
            textfont_size=9))
        fig.update_layout(title=f"月度因子收益热力图({d_start} ~ {d_end})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        roll = sub.rolling(63).std() * np.sqrt(242)
        fig = go.Figure()
        for c in sel:
            fig.add_trace(go.Scatter(x=roll.index, y=roll[c], mode="lines",
                                     name=labels.get(c, c)))
        fig.update_layout(title="滚动 63 日年化波动", yaxis_title="年化波动",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # 绩效汇总(先按列名索引构建再 rename;若构造时传 index=中文标签,
    # Series 原索引是因子列名,reindex 会把全部值变 NaN)
    n_days = len(sub)
    cum_ret = (1 + sub).cumprod().iloc[-1] - 1
    ann_ret = (1 + cum_ret) ** (242 / max(n_days, 1)) - 1
    ann_vol = sub.std() * np.sqrt(242)
    summary = pd.DataFrame({
        "累计收益": cum_ret.map("{:.2%}".format),
        "年化收益": ann_ret.map("{:.2%}".format),
        "年化波动": ann_vol.map("{:.2%}".format),
        "夏普(0利率)": (ann_ret / ann_vol.replace(0, np.nan)).map("{:.2f}".format),
    }).rename(index=lambda c: labels.get(c, c))
    st.dataframe(summary, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 2:估值表持仓暴露
# ----------------------------------------------------------------------------

def tab_exposure():
    files, folder = _list_files("估值表")
    choice = None
    c1, c2 = st.columns([2, 2])
    with c1:
        if files:
            choice = st.selectbox(
                f"选择估值表({folder})",
                files, format_func=lambda p: os.path.basename(p))
        else:
            st.info(f"{folder} 下暂无估值表文件,请上传")
        up = st.file_uploader("或上传估值表 Excel", type=["xlsx", "xls"])
    path = _save_upload(up) if up is not None else choice
    if path is None:
        return

    try:
        exposure, meta = EX.calc_fund_exposure(path, save=False)
    except Exception as e:
        st.error(f"估值表解析失败:{e}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("估值日", meta["value_date"])
    m2.metric("股票只数", f"{meta['n_stocks']}")
    m3.metric("因子覆盖", f"{meta['n_covered']}/{meta['n_stocks']}")
    m4.metric("股票仓位", f"{meta['position']:.1%}")

    # 基准对比(可选:index_weights 已下载时)
    bench_exp = None
    iw_files, _ = _list_files("base/index_weights")
    if iw_files:
        with c2:
            iw = st.selectbox("叠加基准指数暴露(可选)",
                              ["不对比"] + iw_files,
                              format_func=lambda p: "不对比" if p == "不对比"
                              else os.path.basename(p))
            if iw != "不对比":
                bench_exp = EX.calc_index_exposure(iw).iloc[-1]

    labels = [f"{STYLE_LABELS.get(c, c)}" for c in exposure.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=exposure.values, name="组合",
                         marker_color=np.where(exposure.values >= 0,
                                               "#d62728", "#2ca02c")))
    if bench_exp is not None:
        fig.add_trace(go.Bar(x=labels, y=bench_exp.reindex(exposure.index).values,
                             name="基准指数", marker_color="#1f77b4", opacity=0.55))
    for y in (-1, -0.5, 0.5, 1):
        fig.add_hline(y=y, line_dash="dot", line_color="grey",
                      line_width=0.8)
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(title=f"风格因子暴露(标准差口径,±0.5/±1 参考线)",
                      yaxis_title="暴露(z-score)", barmode="group",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("暴露为股票内归一化加权(0=全市场均值);现金/期货等非股票科目未计入。")


# ----------------------------------------------------------------------------
# Tab 3:净值归因
# ----------------------------------------------------------------------------

def tab_attribution(f_ret):
    files, folder = _list_files("净值表")
    c1, c2 = st.columns([3, 1])
    with c1:
        choice = None
        if files:
            choice = st.selectbox(f"选择净值序列({folder})", files,
                                  format_func=lambda p: os.path.basename(p))
        else:
            st.info(f"{folder} 下暂无净值文件,请上传")
        up = st.file_uploader("或上传净值序列 Excel/CSV", type=["xlsx", "xls", "csv"])
    with c2:
        window = st.slider("回归窗口(日)", 126, 504, 252, step=63)
    path = _save_upload(up) if up is not None else choice
    if path is None:
        return

    try:
        nav = AT.load_nav(path)
        attribution, style_detail = AT.perform_return_attribution(nav, f_ret, window=window)
    except Exception as e:
        st.error(f"归因计算失败:{e}")
        return

    # 恒等式校验
    fund_ret = nav.pct_change().reindex(attribution.index)
    err = (attribution.sum(axis=1) - fund_ret).abs().max()
    st.caption(f"归因区间 {attribution.index.min().date()} ~ "
               f"{attribution.index.max().date()}({len(attribution)} 个归因日)"
               f";恒等式 |Σ贡献-当日净值收益| 最大误差 {err:.1e}")

    # 图 1:累计归因
    cum = (1 + attribution).cumprod() - 1
    nav_cum = (1 + fund_ret).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_cum.index, y=nav_cum.values, name="净值累计收益",
                             line=dict(width=3, color="#1f77b4")))
    name_map = {"Alpha": "Alpha(残差)", "Country": "国家", "Industry": "行业", "Style": "风格"}
    for c in attribution.columns:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[c], name=name_map.get(c, c),
                                 line=dict(dash="dot" if c == "Alpha" else "solid")))
    fig.update_layout(title="累计收益归因", yaxis_title="累计收益",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 图 2:风格因子拆分
    style_cum = (1 + style_detail).cumprod() - 1
    total = (1 + style_detail.sum(axis=1)).cumprod() - 1
    fig2 = go.Figure()
    for c in style_detail.columns:
        fig2.add_trace(go.Scatter(x=style_cum.index, y=style_cum[c], mode="lines",
                                  name=STYLE_LABELS.get(c, c)))
    fig2.add_trace(go.Scatter(x=total.index, y=total, name="风格合计",
                              line=dict(width=3, color="black", dash="dash")))
    fig2.add_hline(y=0, line_color="grey", line_width=0.8)
    fig2.update_layout(title="风格因子累计收益贡献", yaxis_title="累计贡献",
                       hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    # 年化摘要
    ann = attribution.mean() * 242
    ann_style = (style_detail.mean() * 242).sort_values()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("年化贡献")
        st.dataframe(pd.DataFrame({
            "年化贡献": ann.map("{:+.2%}".format),
            "占比": (ann / ann.sum()).map("{:+.1%}".format),
        }).rename(index=name_map))
    with c4:
        st.subheader("年化风格贡献排序")
        fig3 = go.Figure(go.Bar(
            x=ann_style.values,
            y=[STYLE_LABELS.get(c, c) for c in ann_style.index],
            orientation="h",
            marker_color=np.where(ann_style.values >= 0, "#d62728", "#2ca02c")))
        fig3.update_layout(title=None, xaxis_title="年化贡献",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------

def main():
    st.title("📊 Barra CNE-5 风险模型面板")
    st.caption("因子收益 / 持仓暴露 / 净值归因 | 数据:data_store(f_ret、cne_5)")

    f_ret = load_factor_returns()
    if f_ret is None:
        st.error("未找到 f_ret.csv,请先运行:python pipeline.py --step factor_return")
        return

    t1, t2, t3 = st.tabs(["📈 因子收益", "🧭 持仓暴露", "🔍 净值归因"])
    with t1:
        try:
            tab_factor_returns(f_ret)
        except Exception as e:
            st.exception(e)
    with t2:
        try:
            tab_exposure()
        except Exception as e:
            st.exception(e)
    with t3:
        try:
            tab_attribution(f_ret)
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
