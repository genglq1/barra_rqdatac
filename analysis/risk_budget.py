# -*- coding: utf-8 -*-
"""
风险预算 / 风险平价
==================
迁移自原 risk_budget_module.py 的 Risk_Budgeting 类。

改进:
    - 修复 pandas 2.x 废弃的 fillna(method='ffill') -> .ffill()
    - 测试数据获取(akshare)改为可选,主流程接收外部 ret_df

算法:
    Risk_Budgeting 类:风险平价 / 风险预算优化
        risk_budget_objective: 最小化各资产风险贡献(TRC)与目标的差异平方和
        solve_risk_budget: scipy.minimize SLSQP,约束 权重和=1 / 权重∈[0,1]
        不传 PRC_target = 风险平价;传 = 风险预算
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class Risk_Budgeting:
    """
    风险平价和风险预算模型求解。

    用法:
        rb = Risk_Budgeting(ret_df)         # ret_df: 资产日度收益率 DataFrame
        weights, sigma = rb.solve_risk_budget()                 # 风险平价
        weights, sigma = rb.solve_risk_budget(PRC_target=[...])  # 风险预算
    """

    def __init__(self, ret_df=None):
        self.ret_df = ret_df

    def risk_budget_objective(self, weights, cov, PRC_target=None):
        """风险预算模型的最优化目标函数。"""
        weights = np.array(weights)
        sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))   # 组合标准差
        MRC = np.dot(cov, weights) / sigma                        # 边际风险贡献
        TRC = weights * MRC                                      # 各资产风险贡献

        if PRC_target is None:
            delta_TRC = [sum((i - TRC) ** 2) for i in TRC]        # 风险平价
        else:
            TRC_target = np.array(PRC_target) * sum(TRC)
            delta_TRC = (TRC - TRC_target) ** 2                   # 风险预算
        return sum(delta_TRC)

    def solve_risk_budget(self, bnds_list=[], cons_list=[], PRC_target=None):
        """求解风险预算下的最优权重。"""
        R_cov = self.ret_df.cov()
        cov = np.array(R_cov)

        x0 = np.ones(cov.shape[0]) / cov.shape[0]   # 初始权重

        if len(bnds_list) == 0:
            bnds = tuple((0, 1) for _ in x0)
        else:
            bnds = bnds_list

        if len(cons_list) == 0:
            cons = LinearConstraint(np.ones(cov.shape[0]), lb=1, ub=1)   # 权重和=1
        else:
            cons = cons_list

        options = {"disp": False, "maxiter": 10000, "ftol": 1e-30}

        solution = minimize(
            self.risk_budget_objective, x0, args=(cov, PRC_target),
            bounds=bnds, constraints=cons, method="SLSQP", options=options
        )

        if solution.success:
            print("优化成功!")
            final_weights = solution.x
        else:
            print("优化失败:", solution.message)
            final_weights = solution.x

        sigma = np.sqrt(np.dot(final_weights, np.dot(cov, final_weights)))
        print(f"资产组合标准差为:{sigma}")
        for i in range(len(final_weights)):
            print(f"{final_weights[i]:.2%} 投资于 {R_cov.columns[i]}")

        return final_weights, sigma
