# -*- coding: utf-8 -*-
"""
数据下载层 (data)
=================
基于 rqdatac,合并原项目的 data_download_module.py + data_process.py 两层,
直接从 rqdatac 取数并透视为"日期×股票"基础宽表,落盘到 data_store/base/。

模块:
    client      rqdatac 连接封装(单例)+ 代码后缀转换
    io          增量更新机制(_get_latest_date / 透视 / 合并落盘)
    universe    股票池 + 交易日历
    price       行情(后复权价 + 涨跌幅)
    valuation   估值(市值/PE/PB/换手)
    financial   财务三表(PIT 严格)
    reference   国债利率/指数行情/申万行业/指数权重
    consensus   一致预期数据(分析师预测,供 EPIBS 因子)

字段映射详见 README "字段映射表"。财务统一用 get_pit_financials_ex 杜绝未来函数。
"""
