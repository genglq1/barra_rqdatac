# -*- coding: utf-8 -*-
"""
持仓暴露分析(基于估值表)
========================
迁移自原 持仓暴露分析.py。
读取估值表 + CNE-5 因子,按市值加权计算指数/基金的因子暴露。

修复原项目 bug:
    ① barra_df 未定义(应为 factor)
    ② factor_files_data 未定义
    ③ 桌面/E 盘绝对路径 -> 参数传入 / config
    ④ 交易日历路径不一致 -> 读 data_store/base/trade_cal.csv
    ⑤ 代码后缀转换(add_exchange_suffix)保留,适配 rqdatac 风格

提供两个入口:
    calc_index_exposure(index_weights_file, version="cne5")
        指数成分权重 -> 指数因子暴露
    calc_fund_exposure(holding_file, value_date, version="cne5")
        估值表 -> 基金因子暴露
"""

import os
import re
import pandas as pd
import numpy as np

from config import get_model_dir


def add_exchange_suffix(code):
    """
    6 位裸股票代码加交易所后缀(转 Wind 风格,与 cne_5.csv 的 code 列对齐)。
    6 开头 -> .SH;0/3 开头 -> .SZ;8 开头 -> .BJ

    注:此处处理"裸代码→带后缀";data/client.to_rqcode 处理"Wind后缀→rqdatac后缀"。
    两者职责不同(裸代码无后缀无法用 to_rqcode),故独立实现,非冗余。
    """
    if isinstance(code, str):
        if code.startswith("6"):
            return code + ".SH"
        elif code.startswith(("0", "3")):
            return code + ".SZ"
        elif code.startswith("8"):
            return code + ".BJ"
    return code


def _load_factor(version="cne5"):
    """
    读取 CNE-5(或 CNE-6)风格因子长表,返回 MultiIndex(trade_date, code) 的 DataFrame。
    修复原 bug:原代码用 barra_df 但实际应为 factor。
    """
    path = os.path.join(get_model_dir(version), "cne_5.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"风格因子文件不存在: {path}(请先运行 model/{version}.py 合成)"
        )
    factor = pd.read_csv(path)
    factor = factor.set_index(["trade_date", "code"])
    return factor


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

    index_holding = pd.read_csv(index_weights_file, index_col=0, encoding="GBK")
    # 兼容:weight 列名可能是 'weight' 或首列
    if "weight" not in index_holding.columns:
        weight_col = index_holding.columns[0]
    else:
        weight_col = "weight"

    exposure = pd.DataFrame()
    # 按日期分组计算(若权重文件含多日)
    if "date" in index_holding.columns:
        for date, group in index_holding.groupby("date"):
            try:
                fac = factor.loc[date]
            except KeyError:
                continue
            merged = pd.merge(group, fac, how="left",
                              left_index=True, right_index=True)
            for column in fac.columns:
                exposure.loc[date, column] = (
                    group[weight_col] / 100 * merged[column]
                ).sum()
    else:
        # 单日权重:用因子最新日
        latest_date = factor.index.get_level_values("trade_date").max()
        fac = factor.loc[latest_date]
        merged = pd.merge(index_holding, fac, how="left",
                          left_index=True, right_index=True)
        for column in fac.columns:
            exposure.loc[latest_date, column] = (
                index_holding[weight_col] / 100 * merged[column]
            ).sum()

    return exposure


def calc_fund_exposure(holding_file, value_date, version="cne5"):
    """
    计算基金的因子暴露(基于估值表)。

    参数:
        holding_file: 估值表 Excel 路径(由调用方传入,不再硬编码桌面路径)
        value_date: 估值日 "YYYY-MM-DD"
        version: 因子版本
    返回:
        Series, index=各风格因子(加权暴露值)

    注:估值表格式需含"科目代码"和"市值占比"列(原项目口径)。
        不同券商估值表表头位置可能不同,自动定位表头行。
    """
    factor = _load_factor(version)

    holding_df = pd.read_excel(holding_file)

    # 自动定位表头行(含"科目代码"的行)
    header_row_index = holding_df[
        holding_df.apply(lambda row: row.astype(str).str.contains("科目代码").any(), axis=1)
    ].index[0]
    holding_df.columns = holding_df.iloc[header_row_index]
    holding_df["科目代码"] = holding_df["科目代码"].astype(str)

    # 提取 6 位股票代码
    pattern = r"\b(\d{6})\b"
    holding_df = holding_df[holding_df["科目代码"].str.contains(pattern, regex=True, na=False)]
    holding_df["科目代码"] = holding_df["科目代码"].str.extract(pattern, expand=False)
    holding_df["科目代码"] = holding_df["科目代码"].map(add_exchange_suffix)
    holding_df.set_index("科目代码", inplace=True)

    # 市值占比归一化
    holding_df["市值占比"] = holding_df["市值占比"] / holding_df["市值占比"].sum()

    # 对齐到估值日的因子
    try:
        fac = factor.loc[value_date]
    except KeyError:
        raise KeyError(f"估值日 {value_date} 无因子数据")

    exposure = pd.Series(dtype=float)
    merged = pd.merge(holding_df, fac, how="left",
                      left_index=True, right_index=True)
    for factor_name in fac.columns:
        exposure[factor_name] = (
            holding_df["市值占比"] * merged[factor_name]
        ).sum()

    return exposure
