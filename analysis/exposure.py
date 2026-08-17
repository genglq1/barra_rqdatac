# -*- coding: utf-8 -*-
"""
持仓暴露分析(基于估值表)
========================
读取估值表/指数权重 + CNE-5 风格因子,按市值加权计算基金/指数的因子暴露。

适配真实私募 4 级科目估值表(券商导出,样例见 data_store/估值表/):
    - 表头行含"科目代码";第 2 行单元格含"估值日期:YYYY-MM-DD"(自动提取,文件名日期兜底);
    - 股票持仓行科目代码为 14 位层级码 = 8 位三级科目 + 6 位股票代码,
      如 11020101600000 = 11020101(交易性股票投资_上交所) + 600000(浦发银行);
    - 仅取三级科目 x01 结尾的"主仓行",天然排除 x99 估值增值行(同股票的估值增值
      科目,避免重复计算)与红利税等其他四级科目;信用账户(11021201 等)/科创板
      (1102C101 等)主仓行同样匹配;
    - 权重列"市值占净值%"为百分数(92.28 = 92.28%),股票内归一化后加权,
      暴露口径为"股票部分"(现金/股指期货 3102 等非股票科目不纳入,打印仓位注明);
    - 股票代码直接加 rqdatac 风格后缀(与 cne_5.csv 的 code 列对齐,
      旧版误生成 .SH/.SZ Wind 后缀导致 merge 全空)。

入口:
    calc_fund_exposure(holding_file, value_date=None, version="cne5")
        估值表 -> 基金股票部分因子暴露
    calc_index_exposure(index_weights_file, version="cne5")
        指数成分权重 -> 指数因子暴露

CLI:
    python -m analysis.exposure fund <估值表.xls> [估值日]
    python -m analysis.exposure index <权重csv>
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

from config import get_model_dir, get_path
from data.client import to_rqcode

# 4 级科目股票主仓行:1102(交易性股票投资) + 2位账户/板块 + 01(主仓) + 6位股票代码
STOCK_CODE_PATTERN = re.compile(r"^1102[0-9A-Z]{2}01\d{6}$")

# 估值表中的日期单元格,如"估值日期:2026-04-30"
DATE_CELL_PATTERN = re.compile(r"估值日期[:：]\s*(\d{4}-\d{2}-\d{2})")


def add_exchange_suffix(code):
    """
    6 位裸股票代码加交易所后缀(rqdatac 风格,与 cne_5.csv 的 code 列直接对齐)。
    6 开头 -> .XSHG;0/3 开头 -> .XSHE;4/8 开头 -> .XBJG(北交所)

    注:cne_5.csv 的 code 为 rqdatac 风格(.XSHG/.XSHE/.XBJG),旧版输出 .SH/.SZ
    Wind 后缀导致 merge 全空,故直接生成 rq 风格,不再依赖 to_rqcode 二次转换。
    """
    if isinstance(code, str):
        if code.startswith("6"):
            return code + ".XSHG"
        elif code.startswith(("0", "3")):
            return code + ".XSHE"
        elif code.startswith(("4", "8")):
            return code + ".XBJG"
    return code


def _load_factor(version="cne5"):
    """
    读取 CNE-5(或 CNE-6)风格因子长表,返回 MultiIndex(trade_date, code) 的 DataFrame。
    """
    path = os.path.join(get_model_dir(version), "cne_5.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"风格因子文件不存在: {path}(请先运行 model/{version}.py 合成)"
        )
    factor = pd.read_csv(path)
    factor = factor.set_index(["trade_date", "code"])
    return factor


def _locate_factor_date(factor, value_date):
    """
    取估值日对应的因子截面:当日无因子(非交易日/数据未覆盖)时回退最近因子日。
    factor 的 trade_date 索引为 ISO 字符串,可直接字典序比较。
    """
    try:
        fac = factor.loc[value_date]
        return fac, value_date
    except KeyError:
        pass
    dates = factor.index.get_level_values("trade_date").unique()
    earlier = dates[dates <= value_date]
    if len(earlier) == 0:
        raise KeyError(f"估值日 {value_date} 之前无任何因子数据")
    used = earlier[-1]
    if used != value_date:
        print(f"估值日 {value_date} 无因子数据(非交易日或未覆盖),回退最近因子日 {used}")
    return factor.loc[used], used


def _parse_valuation_table(holding_file, value_date=None):
    """
    解析私募 4 级科目估值表,提取股票主仓持仓。

    参数:
        holding_file: 估值表 Excel 路径(.xls/.xlsx)
        value_date: 估值日 "YYYY-MM-DD";None 时自动从表内提取(文件名日期兜底),
                    传入时与表内日期校验一致
    返回:
        (weights, meta):
        weights - Series,index=rq 风格股票代码,value=股票内归一化权重(和为 1;
                  同股票多账户持仓已合并)
        meta    - dict: value_date(估值日), position(股票仓位,小数),
                  n_stocks(股票只数), loose_fallback(是否宽松回退解析)
    """
    raw = pd.read_excel(holding_file, header=None, dtype=str)

    # ---- 估值日期:扫单元格提取,文件名日期兜底 ----
    found_date = None
    for cell in raw.astype(str).values.flatten():
        m = DATE_CELL_PATTERN.search(cell)
        if m:
            found_date = m.group(1)
            break
    if found_date is None:
        m = re.search(r"(\d{8})", os.path.basename(holding_file))
        if m:
            d = m.group(1)
            found_date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    if value_date is None:
        if found_date is None:
            raise ValueError(f"未能从 {holding_file} 提取估值日期,请显式传入 value_date")
        value_date = found_date
    elif found_date and found_date != value_date:
        raise ValueError(f"传入估值日 {value_date} 与表内估值日期 {found_date} 不一致")

    # ---- 表头行定位(含"科目代码"的行,兼容表头位置差异)----
    header_idx = None
    for i in range(min(20, len(raw))):
        if raw.iloc[i].astype(str).str.contains("科目代码").any():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"未在 {holding_file} 前 {min(20, len(raw))} 行中找到含'科目代码'的表头")
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = raw.iloc[header_idx]

    codes = df["科目代码"].astype(str).str.strip()

    # ---- 股票主仓行:14 位 4 级科目码;匹配 0 行时回退宽松 6 位提取(其他券商格式)----
    loose_fallback = False
    stock_rows = df[codes.str.match(STOCK_CODE_PATTERN)].copy()
    if stock_rows.empty:
        loose_fallback = True
        print("⚠️  未匹配到 1102 4 级科目主仓行,回退宽松 6 位代码提取")
        pattern = r"(\d{6})"
        stock_rows = df[codes.str.contains(pattern, regex=True, na=False)].copy()
        stock_rows["_code6"] = codes[stock_rows.index].str.extract(pattern, expand=False)
    else:
        stock_rows["_code6"] = stock_rows["科目代码"].str.strip().str[-6:]
    if stock_rows.empty:
        raise ValueError("估值表中未找到股票持仓行")

    stock_rows["stock_code"] = stock_rows["_code6"].map(add_exchange_suffix)

    # ---- 权重列:市值占净值%(百分数),兼容其他列名 ----
    wcol = next((c for c in ("市值占净值%", "市值占比", "占净值比例")
                 if c in df.columns), None)
    if wcol is None:
        raise ValueError(f"未找到权重列(候选:市值占净值%/市值占比/占净值比例),实际列:{list(df.columns)}")
    w = pd.to_numeric(stock_rows[wcol], errors="coerce")
    stock_rows = stock_rows.assign(_w=w).dropna(subset=["_w"])

    # 仓位量级自检:权重合计>1.5 判为百分数口径,折算小数
    position = w_sum = stock_rows["_w"].sum()
    if w_sum > 1.5:
        position = w_sum / 100.0

    # 股票内归一化;同股票多账户(普通+信用)合并
    weights = stock_rows.groupby("stock_code")["_w"].sum()
    weights = weights / weights.sum()

    meta = {
        "value_date": value_date,
        "position": position,
        "n_stocks": len(weights),
        "loose_fallback": loose_fallback,
    }
    return weights, meta


def calc_fund_exposure(holding_file, value_date=None, version="cne5", save=True):
    """
    计算基金的因子暴露(基于估值表,口径为股票部分)。

    参数:
        holding_file: 估值表 Excel 路径(4 级科目估值表,自动提取估值日期)
        value_date: 估值日 "YYYY-MM-DD"(None 自动提取;传入则与表内校验)
        version: 因子版本
        save: True 时结果落盘 data_store/analysis/
    返回:
        Series, index=各风格因子(加权暴露值)
    """
    factor = _load_factor(version)
    weights, meta = _parse_valuation_table(holding_file, value_date)
    value_date = meta["value_date"]

    fac, used_date = _locate_factor_date(factor, value_date)

    merged = fac.reindex(weights.index)
    n_covered = int(merged["size"].notna().sum()) if "size" in merged else int(merged.notna().any(axis=1).sum())
    print(f"估值日 {value_date}(因子截面 {used_date}):股票 {meta['n_stocks']} 只,"
          f"因子覆盖 {n_covered} 只,股票仓位 {meta['position']:.1%}"
          f"{'(宽松回退解析)' if meta['loose_fallback'] else ''}")
    if n_covered < meta["n_stocks"]:
        print(f"⚠️  {meta['n_stocks'] - n_covered} 只股票无因子数据(未覆盖/停牌过久),按 0 中性暴露计入")

    exposure = pd.Series(dtype=float)
    for factor_name in fac.columns:
        exposure[factor_name] = (weights * merged[factor_name].fillna(0)).sum()

    if save:
        out_dir = os.path.join(get_path("data_store"), "analysis")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(holding_file))[0]
        out_path = os.path.join(out_dir, f"{base}_暴露.csv")
        exposure.rename("exposure").to_csv(out_path, encoding="utf-8-sig")
        print(f"暴露结果已保存: {out_path}")

    return exposure


def calc_index_exposure(index_weights_file, version="cne5"):
    """
    计算指数的因子暴露。

    参数:
        index_weights_file: 指数成分权重 CSV 路径
            (data_store/base/index_weights/{指数}.csv,index=stock_code,列含 weight)
        version: 因子版本
    返回:
        DataFrame, index=日期, columns=各风格因子(加权暴露值)
    """
    factor = _load_factor(version)

    try:
        index_holding = pd.read_csv(index_weights_file, index_col=0)
    except UnicodeDecodeError:
        index_holding = pd.read_csv(index_weights_file, index_col=0, encoding="GBK")

    # 代码统一 rq 风格后与因子对齐(权重文件多为 Wind 风格 .SH/.SZ)
    index_holding.index = [to_rqcode(str(c)) for c in index_holding.index]
    weight_col = "weight" if "weight" in index_holding.columns else index_holding.columns[0]

    def _exposure(group, fac):
        w = pd.to_numeric(group[weight_col], errors="coerce").dropna()
        w = w / w.sum()                       # 归一化(兼容百分比/小数两种量级)
        merged = fac.reindex(w.index)
        return pd.Series({c: (w * merged[c].fillna(0)).sum() for c in fac.columns})

    all_dates = sorted(factor.index.get_level_values("trade_date").unique())
    date_set = set(all_dates)
    rows = {}
    if "date" in index_holding.columns:
        for date, group in index_holding.groupby("date"):
            if date not in date_set:
                continue
            rows[date] = _exposure(group, factor.loc[date])
    else:
        # 单日权重:用因子最新"数据完整"日(最新交易日估值字段可能尚未发布,
        # 当日因子截面除时序类因子外全空,需回退)
        fac = factor.loc[all_dates[-1]]
        for date in reversed(all_dates):
            fac = factor.loc[date]
            merged = fac.reindex(index_holding.index)
            # 完整性判据:size(依赖当日估值数据、覆盖最高)非空占比;无 size 列时用整体非空率
            completeness = (merged["size"].notna().mean() if "size" in merged.columns
                            else merged.notna().mean().mean())
            if completeness >= 0.5:
                if date != all_dates[-1]:
                    print(f"因子最新日 {all_dates[-1]} 数据不全,回退最近完整日 {date}")
                break
        rows[date] = _exposure(index_holding, fac)

    exposure = pd.DataFrame.from_dict(rows, orient="index")
    exposure.index.name = "date"
    return exposure


def main():
    parser = argparse.ArgumentParser(description="持仓/指数风格暴露分析(CNE-5)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fund = sub.add_parser("fund", help="估值表 -> 基金因子暴露")
    p_fund.add_argument("file", help="估值表 Excel 路径")
    p_fund.add_argument("date", nargs="?", default=None, help="估值日 YYYY-MM-DD(默认自动提取)")
    p_idx = sub.add_parser("index", help="指数权重 -> 指数因子暴露")
    p_idx.add_argument("file", help="指数成分权重 CSV 路径")
    args = parser.parse_args()

    if args.cmd == "fund":
        exposure = calc_fund_exposure(args.file, args.date)
    else:
        exposure = calc_index_exposure(args.file)
    print()
    print(exposure.round(4).to_string())


if __name__ == "__main__":
    main()
