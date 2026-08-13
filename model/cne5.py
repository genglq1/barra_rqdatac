# -*- coding: utf-8 -*-
"""
CNE-5 风格因子合成
==================
迁移自原 barra_cne5.py 的 get_barra_cne_5。
读取 data_store/factors/cne5/ 下的描述因子,按 config/cne5_weights.yaml 的权重
线性合成为 9 大风格因子,产出 data_store/model/cne5/cne_5.csv。

改进(相对原项目):
    - 合成权重从硬编码改为读 config/cne5_weights.yaml
    - 输出路径改为 data_store/model/cne5/

输出结构(长表):
    trade_date, code, size, beta, momentum, residual_volatility,
    non_linear_size, book_to_price, liquidity, earnings_yield, growth, leverage
"""

import os
import pandas as pd

from config import get_weights, get_factor_dir, get_model_dir


def _get_factor_df(factor_name, to_name, start_date="2017-01-01", end_date=None):
    """
    读取单个描述因子宽表,转为长表(行=日期×股票)。
    迁移自原 barra_cne5._get_factor_df。

    若因子文件不存在或为空(如 beta 因窗口不足未产出),返回 None,
    由调用方决定是否跳过该风格因子。
    """
    path = os.path.join(get_factor_dir("cne5"), f"{factor_name}.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 50:
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return None
    df = df[start_date:end_date]
    df.index.name = "trade_date"
    df.columns.name = "code"
    return df.stack().reset_index(name=to_name)


def _merge_into(barra_df, merge_df):
    """把单列因子长表合并进主表(outer join on trade_date, code)。"""
    if barra_df.empty:
        return merge_df
    return pd.merge(barra_df, merge_df, how="outer", on=["trade_date", "code"])


def _compose_style(barra_df, name_list, style_name, weights):
    """
    把多个描述因子按权重合成一个风格因子,并入主表。
    参数:
        name_list: 描述因子名列表(如 ['dastd','cmra','hsigma'])
        style_name: 合成后的风格因子列名(如 'residual_volatility')
        weights: {描述因子: 权重} 取自 config

    容忍缺失分量:若某描述因子文件不存在(返回 None),跳过该分量,
    用其余分量按原权重合成(权重未重新归一,仅缺失分量贡献为0)。
    若全部分量都缺失,跳过该风格因子。
    """
    df = pd.DataFrame()
    available = []
    for name in name_list:
        _df = _get_factor_df(name, name)
        if _df is None:
            print(f"  警告: 描述因子 {name} 缺失,{style_name} 将不含此分量")
            continue
        # 跳过全空因子(文件存在但值全 NaN)
        if _df.empty or _df[name].dropna().empty:
            print(f"  警告: 描述因子 {name} 数据全空,{style_name} 将不含此分量")
            continue
        available.append(name)
        df = _df if df.empty else pd.merge(df, _df, how="outer", on=["trade_date", "code"])

    if not available:
        print(f"  跳过风格因子 {style_name}(全部分量缺失或为空)")
        return barra_df

    # 加权合成(仅用可用的分量,缺失分量贡献为0)
    df[style_name] = sum(df[name] * weights.get(name, 0.0) for name in available)
    merge_df = df[["trade_date", "code", style_name]]
    return _merge_into(barra_df, merge_df)


def _orthogonalize_style(barra_df, style_name, ref_names):
    """
    将指定风格因子对参照因子做市值加权截面正交化(取残差),再重新标准化以保持量纲一致。

    对应 Barra CNE5 原文:
        - Residual Volatility: "orthogonalized with respect to Beta and Size"(主文档附录 A)
        - Liquidity:           "orthogonalized with respect to Size"(主文档 3.3/附录 A)
    流程:逐日截面 WLS 回归 style ~ refs(含截距,回归权重=√流通市值,Barra regression-weighted
    口径)→ 残差 → 重新缩尾+标准化(与 NLSIZE 流程一致)。

    仅当 barra_df 同时含 style 与全部参照列时执行;否则打印警告并原样返回。
    """
    import numpy as np
    from factors import common as F

    missing = [c for c in [style_name] + list(ref_names) if c not in barra_df.columns]
    if missing:
        print(f"  跳过正交化:{style_name} 缺少列 {missing}(需全部存在)")
        return barra_df

    weight_df = F.get_stock_weight(is_cir=True, sqrt=True)

    # 长表 → 宽表(行=日期,列=股票),便于逐日截面回归
    style_wide = barra_df[style_name].unstack("code")
    refs_wide = [barra_df[r].unstack("code") for r in ref_names]
    resid_wide = pd.DataFrame(np.nan, index=style_wide.index, columns=style_wide.columns)

    for td in style_wide.index:
        if td not in weight_df.index or any(td not in rw.index for rw in refs_wide):
            continue
        y = style_wide.loc[td]
        X = pd.concat([rw.loc[td] for rw in refs_wide], axis=1)
        X.columns = ref_names
        w_row = weight_df.loc[td]
        valid = y.notna() & X.notna().all(axis=1) & w_row.notna()
        if valid.sum() < len(ref_names) + 2:        # 至少 k+2 个观测(含截距)
            continue
        yv, Xv, wv = y[valid], X.loc[valid], w_row.loc[valid].values
        _, _, resid = F._regress(yv, Xv, intercept=True, weight=wv, verbose=True)
        resid_wide.loc[td, yv.index] = resid.values[:, 0]

    # 正交后重新标准化(缩尾 + 市值加权去均值 + z-score),保持量纲一致
    resid_wide = F._standardize_with_weights(resid_wide, is_cir=True)
    resid_long = resid_wide.stack()
    resid_long.index.set_names(["trade_date", "code"], inplace=True)
    barra_df[style_name] = resid_long.reindex(barra_df.index)
    print(f"  正交化完成:{style_name} 已对 {ref_names} 做√流通市值加权截面正交并重新标准化")
    return barra_df


def get_barra_cne5(start_date="2017-01-01", end_date=None):
    """
    合成 CNE-5 的 9 大风格因子,产出 cne_5.csv。
    权重来自 config/cne5_weights.yaml。
    """
    if end_date is None:
        import datetime as dt
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    weights = get_weights("cne5")
    barra_df = pd.DataFrame()

    for style_name, sub_weights in weights.items():
        desc_factors = list(sub_weights.keys())

        if len(desc_factors) == 1 and sub_weights[desc_factors[0]] == 1.0:
            # 单因子风格:直接取(如 size/beta/momentum)
            merge_df = _get_factor_df(desc_factors[0], style_name,
                                      start_date=start_date, end_date=end_date)
            if merge_df is None:
                print(f"  跳过风格因子 {style_name}(描述因子 {desc_factors[0]} 缺失)")
                continue
            # 跳过全空因子
            if merge_df[style_name].dropna().empty:
                print(f"  跳过风格因子 {style_name}(描述因子 {desc_factors[0]} 数据全空)")
                continue
            barra_df = _merge_into(barra_df, merge_df)
        else:
            # 多因子合成风格(内部容忍缺失分量)
            barra_df = _compose_style(barra_df, desc_factors, style_name, sub_weights)

    if barra_df.empty:
        print("cne_5.csv: 无可用风格因子,跳过合成")
        return None

    barra_df = barra_df.set_index(["trade_date", "code"])
    barra_df = barra_df.sort_index()

    # 正交化(Barra 原文,降低共线性):
    #   Residual Volatility 对 Beta + Size 正交(主文档附录 A)
    #   Liquidity 对 Size 正交(主文档 3.3/附录 A)
    barra_df = _orthogonalize_style(barra_df, "residual_volatility", ["beta", "size"])
    barra_df = _orthogonalize_style(barra_df, "liquidity", ["size"])

    out_dir = get_model_dir("cne5")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cne_5.csv")
    barra_df.to_csv(out_path)
    print(f"cne_5.csv: 合成完成 ({len(barra_df)} 行, {barra_df.shape[1]} 个风格因子)")
    return barra_df


if __name__ == "__main__":
    get_barra_cne5()
