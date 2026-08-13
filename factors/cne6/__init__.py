# -*- coding: utf-8 -*-
"""
CNE-6 因子集(预留骨架)
======================
本期未实现。未来实现 CNE-6 时:
    1. 在本目录新建各风格因子文件(size.py/beta.py/...),实现计算函数
    2. 在 registry.py 的 REGISTRY 中注册(参照 cne5/registry.py 结构)
    3. 在 config/cne6_weights.yaml 填写合成权重
    4. 实现 model/cne6.py 的风格合成

CNE-6 与 CNE-5 的主要差异(参考 Barra CNE6 文档):
    - 风格因子集调整(部分因子定义/权重变化)
    - 可能新增/移除因子
    - 共享工具在 factors/common.py,无需重写
"""
