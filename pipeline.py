# -*- coding: utf-8 -*-
"""
全流程编排入口
==============
按依赖顺序一键执行:data 下载 -> 因子计算(拓扑排序)-> 模型合成 -> 因子收益率。

用法:
    # 跑完整流程(CNE-5)
    python pipeline.py --version cne5

    # 仅跑数据层
    python pipeline.py --version cne5 --step data

    # 仅跑因子层(依赖 data 已完成)
    python pipeline.py --version cne5 --step factors

    # 跑到模型合成为止(不含因子收益率回归)
    python pipeline.py --version cne5 --step model

    # 指定日期范围
    python pipeline.py --version cne5 --start 2020-01-01 --end 2024-12-31

注:CNE-6 当前为预留骨架,--version cne6 会在因子层提示未实现。
"""

import argparse
import datetime as dt

from config import get_settings, get_start_date, get_factor_return_start_date
from data import universe, price, valuation, financial, reference, consensus
from factors.cne5 import registry as cne5_reg
from factors.cne6 import registry as cne6_reg
from model import cne5 as model_cne5
from model import factor_return


def run_data(start_date="2010-01-01", end_date=None):
    """数据下载层:基础宽表全部更新。"""
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("【步骤 1/4】数据下载层")
    print("=" * 60)

    universe.update_stock_universe()
    universe.update_trade_cal(start_date, end_date)
    reference.update_rf_data(start_date, end_date)
    reference.update_index_ret_data(start_date=start_date, end_date=end_date)
    reference.update_industry()
    price.update_price_data(start_date, end_date)
    valuation.update_valuation_data(start_date, end_date)
    # 财务数据(资产负债表/现金流/利润表)统一入口,起始日需更早(供 TTM/5年回归)
    financial.update_financials(start_date="2005-01-01", end_date=end_date)
    # 一致预期数据(分析师预测,供 EPIBS 因子)
    consensus.update_consensus_eps(start_date=start_date, end_date=end_date)


def run_factors(version="cne5", start_date="2010-01-01", end_date=None):
    """因子计算层:按 registry 依赖顺序计算全部风格因子。"""
    print("=" * 60)
    print(f"【步骤 2/4】因子计算层 ({version})")
    print("=" * 60)

    if version == "cne5":
        cne5_reg.calculate_all(start_date=start_date, end_date=end_date)
    elif version == "cne6":
        cne6_reg.calculate_all(start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"未知版本: {version}(支持 cne5/cne6)")


def run_model_compose(version="cne5", start_date="2017-01-01", end_date=None):
    """模型合成层:描述因子 -> 风格因子(cne_5.csv)。"""
    print("=" * 60)
    print(f"【步骤 3/4】模型合成层 ({version})")
    print("=" * 60)

    if version == "cne5":
        model_cne5.get_barra_cne5(start_date=start_date, end_date=end_date)
    elif version == "cne6":
        from model import cne6 as model_cne6
        model_cne6.get_barra_cne6(start_date=start_date, end_date=end_date)


def run_factor_return(version="cne5", start_date=None, end_date=None):
    """
    因子收益率回归层:WLS 截面回归 -> f_ret.csv。

    start_date 为 None 时取 config date_range.factor_return_start_date(默认 2022-01-01:
    momentum 需 504 个交易日预热,更早的日期该风格列全空)。
    """
    print("=" * 60)
    print(f"【步骤 4/4】因子收益率回归 ({version})")
    print("=" * 60)

    if start_date is None:
        start_date = get_factor_return_start_date()
    if end_date is None:
        end_date = dt.datetime.now().strftime("%Y-%m-%d")
    print(f"回归区间: {start_date} ~ {end_date}")
    factor_return.get_factor_return(start_date, end_date, version=version)


def main():
    parser = argparse.ArgumentParser(description="Barra CNE-5/CNE-6 全流程编排")
    parser.add_argument("--version", default="cne5", choices=["cne5", "cne6"],
                        help="Barra 版本(默认 cne5)")
    parser.add_argument("--step", default="all",
                        choices=["all", "data", "factors", "model", "factor_return"],
                        help="执行步骤(默认 all,全流程)")
    parser.add_argument("--start", default=None,
                        help="起始日期(默认:data/factors/model 取 date_range.start_date,"
                             "factor_return 取 date_range.factor_return_start_date)")
    parser.add_argument("--end", default=None, help="截止日期(默认今天)")
    args = parser.parse_args()

    # 校验 license 配置(提前失败,避免跑到一半才报错)
    settings = get_settings()
    if not settings["rqdatac"]["license"].strip():
        print("⚠️  警告:rqdatac license 未配置!")
        print("   请在 config/settings.yaml 的 rqdatac.license 字段填入 license key。")
        print("   仅当 --step 不涉及 data 层时可能仍可运行(读已有 data_store)。")
        print()

    step = args.step
    generic_start = args.start or get_start_date()
    if step in ("all", "data"):
        run_data(start_date=generic_start, end_date=args.end)
    if step in ("all", "factors"):
        run_factors(version=args.version, start_date=generic_start, end_date=args.end)
    if step in ("all", "model"):
        run_model_compose(version=args.version, start_date=generic_start, end_date=args.end)
    if step in ("all", "factor_return"):
        # 未显式指定 --start 时用配置的 factor_return_start_date(None 透传)
        run_factor_return(version=args.version, start_date=args.start, end_date=args.end)

    print("\n✅ 流程完成。")


if __name__ == "__main__":
    main()
