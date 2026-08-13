# -*- coding: utf-8 -*-
"""
CNE-5 因子注册表与依赖编排
==========================
声明每个风格因子的:计算函数、产出描述因子、依赖的基础数据、依赖的其他因子。
pipeline.py 据此做拓扑排序,保证依赖先算完。

依赖关系(关键):
    size       -> (无依赖)
    beta       -> (无依赖)
    momentum   -> (无依赖)
    value      -> (无依赖)
    liquidity  -> (无依赖)
    earnings   -> (无依赖)
    growth     -> (无依赖)
    volatility -> depends_on [beta, sigma, lncap]   ⚠️ 必须在 size/beta 之后
    leverage   -> (无依赖)
"""

from . import (
    size as _size,
    beta as _beta,
    momentum as _momentum,
    volatility as _volatility,
    liquidity as _liquidity,
    value as _value,
    earnings as _earnings,
    growth as _growth,
    leverage as _leverage,
)

VERSION = "cne5"


# 因子注册表:每个风格因子对应一个 entry
#   func:       计算入口函数
#   outputs:    产出的描述因子名列表
#   depends_on: 依赖的其他【风格因子名】(同版本内),无依赖为 []
#
# 注:depends_on 用风格因子名而非描述因子名。例如 hsigma 复用 sigma 描述因子,
#     sigma 由 "beta" 风格因子产出,故 residual_volatility 的 depends_on = ["beta"]。
#     故 residual_volatility 的 depends_on = ["beta", "size"]。
REGISTRY = {
    "size":                {"func": _size.cal_size_factor,             "outputs": ["lncap"],               "depends_on": []},
    "non_linear_size":     {"func": _size.cal_nonlinearsize_factor,    "outputs": ["nlsize"],              "depends_on": []},
    "beta":                {"func": _beta.cal_beta_factor,             "outputs": ["beta", "sigma"],       "depends_on": []},
    "momentum":            {"func": _momentum.cal_momentum_factor,     "outputs": ["rstr"],                "depends_on": []},
    "residual_volatility": {"func": _volatility.cal_residualvolatility_factor,
                            "outputs": ["dastd", "cmra", "hsigma"],    "depends_on": ["beta"]},
    "liquidity":           {"func": _liquidity.cal_liquidity_factor,   "outputs": ["stom", "stoq", "stoa"], "depends_on": []},
    "book_to_price":       {"func": _value.cal_booktoprice_factor,     "outputs": ["btop"],                "depends_on": []},
    "earnings_yield":      {"func": _earnings.cal_earningsyield_factor,"outputs": ["cetop", "etop", "epibs"], "depends_on": []},
    "growth":              {"func": _growth.cal_growth_factor,         "outputs": ["egro", "sgro", "egibs", "egibs_s"], "depends_on": []},
    "leverage":            {"func": _leverage.cal_leverage_factor,     "outputs": ["mlev", "dtoa", "blev"], "depends_on": []},
}


# 所有描述因子名(展开 outputs)
ALL_DESC_FACTORS = []
for entry in REGISTRY.values():
    ALL_DESC_FACTORS.extend(entry["outputs"])

# 所有风格因子名(与 cne5_weights.yaml 的 key 对应)
ALL_STYLE_FACTORS = list(REGISTRY.keys())


def get_execution_order():
    """
    拓扑排序:返回按依赖关系排好序的风格因子计算顺序。
    保证被依赖的因子先算(如 residual_volatility 在 beta/size 之后)。
    """
    order = []
    visited = set()
    visiting = set()   # 用于检测循环依赖

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"检测到循环依赖: {name}")
        visiting.add(name)
        for dep in REGISTRY[name]["depends_on"]:
            if dep not in REGISTRY:
                raise KeyError(f"因子 {name} 依赖未注册的因子 {dep}")
            visit(dep)
        visiting.discard(name)
        visited.add(name)
        order.append(name)

    for name in REGISTRY:
        visit(name)
    return order


def calculate_all(start_date="2010-01-01", end_date=None):
    """
    按依赖顺序计算全部 CNE-5 因子。
    供 pipeline 调用。
    """
    order = get_execution_order()
    print(f"CNE-5 因子计算顺序: {order}")
    for name in order:
        print(f"\n>>> 计算 {name} ...")
        REGISTRY[name]["func"](start_date=start_date, end_date=end_date)
