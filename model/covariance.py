# -*- coding: utf-8 -*-
"""
因子协方差矩阵(预留,本期不实现)
================================
Barra 完整风险模型还需因子收益率协方差矩阵 V(f),用于风险预测。
本项目当前未实现(原项目技术文档"局限"中也标注为后续优化方向)。

未来实现方向:
    1. EWMA 协方差估计(半衰期 90/250 日)
    2. 或用 Leda/LW 收缩估计提升数值稳定性
    3. 输出: data_store/model/{version}/factor_cov.csv

接口预留(占位):
    def estimate_factor_cov(f_ret, half_life=90):
        ...
"""
