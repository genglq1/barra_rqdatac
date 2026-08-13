# -*- coding: utf-8 -*-
"""
风格因子层 (factors)
====================
按 Barra 版本分包(版本子包隔离):
    factors/cne5/   CNE-5 因子集(本次实现)
    factors/cne6/   CNE-6 因子集(预留骨架)

每个版本子包内,每个风格因子独立一个文件,通过 registry.py 注册依赖。
版本无关的共享工具(标准化/缩尾/WLS回归/指数权重/TTM/日频对齐)统一在 common.py,
被所有版本复用。

扩展:
    - 加新版本:新建 factors/{version}/,填因子文件 + registry
    - 加新因子:在该版本子包加文件 + registry 一行
"""
