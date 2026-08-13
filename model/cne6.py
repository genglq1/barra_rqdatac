# -*- coding: utf-8 -*-
"""
CNE-6 风格因子合成(预留骨架)
============================
本期未实现。未来实现 CNE-6 时,参照 cne5.py 结构:
    1. 读取 data_store/factors/cne6/ 下的描述因子
    2. 按 config/cne6_weights.yaml 合成
    3. 产出 data_store/model/cne6/cne_5.csv(可重命名为 cne_6.csv)
"""

import os
from config import get_model_dir


def get_barra_cne6(start_date="2017-01-01", end_date=None):
    """CNE-6 风格合成(占位)。"""
    print("CNE-6 风格合成尚未实现(预留骨架)。请先实现 factors/cne6/ 并填 config/cne6_weights.yaml。")
    return None
