# Barra CNE-5 / CNE-6 风险因子模型(rqdatac 版本)

基于米筐 **rqdatac** 数据源重构的 Barra 多因子风险模型,采用**领域分包 + 版本分包**架构。
本次实现 **CNE-5**(10 大风格因子、22 个描述因子、因子收益率 WLS 回归),并预留 **CNE-6** 骨架。

> 本项目从原 `因子模型/`(Tushare/Wind/Akshare 数据源)迁移而来:数据源统一为 rqdatac,
> 配置全部抽离到 `config/`,因子口径逐因子对照 Barra CNE-5 原文核对(见第五节),
> 整体计算结构与原项目保持同构,便于交叉验证。

---

## 一、快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 配置 license
编辑 `config/settings.yaml`,在 `rqdatac.license` 填入你的 license key(米筐邮件获取):
```yaml
rqdatac:
  license: "你的license_key"
```

### 3. 运行全流程(CNE-5)
```bash
# 在 barra_rqdatac/ 目录下执行
python pipeline.py --version cne5

# 或分步执行
python pipeline.py --step data           # 仅下载数据
python pipeline.py --step factors        # 仅算因子
python pipeline.py --step model          # 仅合成风格因子
python pipeline.py --step factor_return  # 仅算因子收益率
```

产出会写入 `data_store/`:
- `data_store/base/`          基础宽表(行情/估值/财务/一致预期/参考,见第三节)
- `data_store/factors/cne5/`  CNE-5 描述因子(lncap/beta/... 共 22 张,见第四节)
- `data_store/model/cne5/`    cne_5.csv(10 大风格因子)+ f_ret.csv(因子收益率)

---

## 二、项目架构

```
barra_rqdatac/
├── config/          配置层(settings.yaml + 各版本权重 yaml + 加载器)
├── data_store/      数据产出(独立隔离,base/factors/model 三层)
├── data/            数据下载层(rqdatac,client/universe/price/valuation/financial/reference/io)
├── factors/         风格因子层(common 共享工具 + cne5 实现 + cne6 占位)
├── model/           模型层(cne5 合成 + factor_return 回归 + cne6/covariance 占位)
├── analysis/        分析层(exposure 持仓暴露 / attribution 归因 / performance 业绩指标)
├── pipeline.py      全流程编排(支持 --version / --step)
├── requirements.txt
└── README.md
```

### 分层职责
| 层 | 职责 |
|---|---|
| `config/` | 集中管理 license、路径、因子合成权重,告别硬编码 |
| `data/` | rqdatac 取数 → 透视为"日期×股票"宽表,增量更新 |
| `factors/` | 基础宽表 → 描述因子,每个风格因子独立文件 |
| `model/` | 描述因子 → 风格因子合成 + 因子收益率回归 |
| `analysis/` | 下游应用:持仓暴露、收益归因、业绩指标 |

---

## 三、数据字段(data_store/base)

数据层将 rqdatac 各接口统一透视为**"日期×股票"宽表**(index=交易日,columns=股票代码 rqdatac 风格,
如 `600519.XSHG`),按日增量更新落盘。财务宽表自 **2005-01-01** 起存(供 TTM 与 5 年回归回溯),
其余默认自 `config/date_range.start_date`(2010-01-01)起。

### 3.1 行情字段

| 宽表文件 | 含义 | rqdatac 接口 + 字段 | 单位/口径 | 下游因子 |
|---|---|---|---|---|
| stock_open / high / low / close / vol / amount | 开盘/最高/最低/收盘/成交量/成交额 | `get_price(adjust_type='post')` 的 open/high/low/close/volume/total_turnover | 后复权,元 | 复盘与扩展用 |
| stock_close_unadjusted | 不复权收盘价 | `get_price(adjust_type='none')` 的 close | 当日真实价,元 | EPIBS 分母 |
| stock_ret | 日涨跌幅 | `get_price_change_rate`(已是宽表) | 小数 | beta/momentum/volatility/factor_return |

> 后复权价保证收益率连续,用于 Beta/Momentum 等时序因子;截面比值因子(EPIBS)必须用
> 不复权真实价,否则除权会使老股预期收益率被系统性低估。

### 3.2 估值字段

| 宽表文件 | 含义 | rqdatac 接口 + 字段 | 单位/口径 | 下游因子 |
|---|---|---|---|---|
| stock_size | 总市值 | `get_factor(market_cap)` | 元 | lncap / MLEV |
| stock_size_cir | 流通市值 | `get_factor(a_share_market_val_in_circulation)` | 元 | 标准化权重、正交回归权重 |
| stock_pe | 市盈率 | `get_factor(pe_ratio)` | TTM | —(扩展用) |
| stock_pb | 市净率 | `get_factor(pb_ratio)` | — | btop = 1/PB |
| stock_turnover | 换手率 | `get_turnover_rate` 取 `today` 列 | 百分比 | stom / stoq / stoa |
| ep_ratio_ttm | 盈市率 TTM | `get_factor(ep_ratio_ttm)` | 小数(归母净利润TTM/总市值) | etop(直接取) |
| pcf_ratio_ttm | 经营市现率 TTM | `get_factor(pcf_ratio_ttm)` | 小数(总市值/经营现金流TTM) | cetop = 1/pcf |

### 3.3 财务字段(2005 起存,日频 PIT 对齐)

| 宽表文件 | 含义 | rqdatac 接口 + 字段 | 下游因子 |
|---|---|---|---|
| total_assets | 总资产 TA | `get_factor(total_assets)` | dtoa |
| total_liab | 总负债 TD | `get_factor(total_liabilities)` | dtoa |
| total_ncl | 非流动负债 LD | `get_factor(non_current_liabilities)` | mlev / blev |
| total_hldr_eqy_exc_min_int | 归母权益 BE | `get_factor(equity_parent_company)` | blev |
| oth_eqt_tools_p_shr | 优先股 PE | 不下载(因子库不提供,leverage 按 0 兜底) | mlev / blev |
| n_cashflow_act | 经营活动现金流净额 | `get_factor(cash_flow_from_operating_activities)` | —(扩展用) |
| basic_eps | 基本每股收益 | `get_factor(basic_earnings_per_share)` | egro(年报回归) |
| revenue_ps | 营业收入(总额) | `get_factor(operating_revenue)` | sgro(斜率/均值,见 5.2.8) |

> rqdatac 因子库的财务字段**已是日频且按最新已披露财报填充(PIT 对齐)**,无需再做季频→日频转换;
> 年报口径(EGRO/SGRO 的 5 年回归)则单独用 `get_pit_financials_ex`(严格 PIT)取,
> 落盘为缓存 `annual_reports.csv`(列:order_book_id, year, basic_earnings_per_share, operating_revenue)。

### 3.4 一致预期字段(consensus,日频)

| 宽表文件 | 含义 | rqdatac 接口 + 字段 | 下游因子 |
|---|---|---|---|
| comp_con_eps_ftm | 一致预期 EPS(FTM) | `consensus.get_comp_indicators` fields=comp_con_eps_ftm | epibs 分子(元/股) |
| comp_con_net_profit_growth_ratio_t3 | 一致预期净利润增长率(长期 T+3) | `consensus.get_comp_indicators` | egibs |
| comp_con_net_profit_growth_ratio_ftm | 一致预期净利润增长率(短期 FTM) | `consensus.get_comp_indicators` | egibs_s |

> 一致预期数据只覆盖分析师跟踪的股票(约 45%),小盘股缺失属数据源固有属性;
> growth_ratio 字段单位可能为百分比或小数,计算时按截面中位数量级自检统一为小数。

### 3.5 参考字段

| 宽表文件 | 含义 | rqdatac 接口 | 下游用途 |
|---|---|---|---|
| rf | 10 年期国债收益率 | `get_yield_curve(tenor='10Y')` | 无风险利率(年化小数,因子内 /365) |
| Rt | 中证全指(000985.XSHG)日收益率 | `get_price_change_rate` | Beta 回归的市场组合代理 |
| industry_l1 | 行业归属(默认中信一级 citics_2019,约 30 个;可切申万;文件名源无关,实际来源见 source 列) | `get_instrument_industry(source=..., level=1)` | 因子收益率的行业哑变量 |
| trade_cal | 沪深交易日历(含年/季/月/周派生列) | `get_trading_dates` | 日频对齐、年报生效日映射 |
| all_instruments | 全 A 股清单 | `all_instruments(type='CS')` | 股票池 |
| annual_reports | 年报缓存(EPS/营收,按股票×年份) | `get_pit_financials_ex` | EGRO/SGRO |
| index_weights/{指数}.csv | 指数成分权重(可选) | `index_weights` | 持仓暴露分析 |

### 3.6 rqdatac 3.4.x 接口要点(实跑确认)
- **无 `get_eod_factor`**,估值/财务统一用 `get_factor`(因子库,单字段一次,返回 MultiIndex)
- 行情 `get_price(expect_df=True)` 返回 MultiIndex(order_book_id, date),需 reset_index 后 pivot
- 涨跌幅 `get_price_change_rate(expect_df=True)` 已是宽表,直接合并无需 pivot
- 利润表年报可用 `get_pit_financials_ex`(PIT 严格),资产负债表/现金流只能在因子库 `get_factor` 取
- 换手率用独立接口 `get_turnover_rate`(非 get_price 字段)
- 行业用 `get_instrument_industry`(个股→行业),不是 `get_industry_mapping`(行业→成分,反向)
- 代码后缀统一在 `data/client.py` 转换(.SH↔.XSHG / .SZ↔.XSHE)

---

## 四、因子计算

`pipeline.py` 按 `factors/cne5/registry.py` 的依赖声明做**拓扑排序**自动编排:

```
无依赖(可并行): size, non_linear_size, beta, momentum, liquidity, book_to_price, earnings_yield, growth, leverage
依赖先行:       residual_volatility(hsigma 复用 beta 产出的 sigma)
```

### 4.1 描述因子清单(22 个:公式与参数)

| 风格因子 | 描述因子 | 计算公式(实现口径) | 参数 |
|---|---|---|---|
| Size | lncap | ln(总市值) | — |
| Non-linear Size | nlsize | 标准化 LNCAP 的立方对 LNCAP 做 WLS 回归取残差,再标准化 | 回归权重 √流通市值 |
| Beta | beta | 个股超额收益对市场超额收益的滚动 WLS 斜率 | 252 日窗口,半衰期 63,缺失>42 日跳过 |
| Beta | sigma | 上式回归的残差时序标准差 | 同 beta(供 hsigma 复用) |
| Momentum | rstr | Σ w·[ln(1+r)−ln(1+rf)],剔除最近 21 日 | 504 日窗口,半衰期 126,最少 483 个有效值 |
| Residual Volatility | dastd | 日超额收益的加权标准差 | 252 日窗口,半衰期 42 |
| Residual Volatility | cmra | Z(T)=窗口内最后 21T 个交易日的对数超额累计和(T=1..12),CMRA=max{Z}−min{Z} | 252 日窗口 |
| Residual Volatility | hsigma | 复用 beta 产出的 sigma(不再回归、不重复标准化) | — |
| Liquidity | stom | ln(Σ 21 日换手率) | 21 日,缺失>3 日不计算 |
| Liquidity | stoq | ln(Σ 63 日换手率 / 3) | 63 日,缺失>9 日不计算 |
| Liquidity | stoa | ln(Σ 252 日换手率 / 12) | 252 日,缺失>36 日不计算 |
| Book-to-Price | btop | 1 / PB | — |
| Earnings Yield | cetop | 1 / pcf_ratio_ttm(经营现金流 TTM/市值) | TTM |
| Earnings Yield | etop | ep_ratio_ttm(归母净利润 TTM/总市值) | TTM |
| Earnings Yield | epibs | comp_con_eps_ftm / 不复权收盘价 | 一致预期 FTM |
| Growth | egro | 近 5 年年报 EPS 对年份 OLS 回归斜率 ÷ EPS 均值 | 滑动 5 年窗,次年 4/30 生效并前向填充 |
| Growth | sgro | 近 5 年年报营业收入对年份 OLS 回归斜率 ÷ 营收均值 | 同上 |
| Growth | egibs | 一致预期净利润增长率(长期 T+3) | 量级自检统一为小数 |
| Growth | egibs_s | 一致预期净利润增长率(短期 FTM) | 同上 |
| Leverage | mlev | (ME + PE + LD) / ME | ME=当日总市值;PE 缺失按 0 |
| Leverage | dtoa | TD / TA | 总负债/总资产 |
| Leverage | blev | (BE + PE + LD) / BE | 归母权益 |

### 4.2 统一后处理:缩尾与标准化(每个描述因子都要过)

每个描述因子按日截面执行三步(与主文档 3.3 一致:风格因子市值加权均值 0、等权标准差 1):

1. **缩尾 winsorize**:±3σ 截断,中心 μ 用**流通市值加权均值**、σ 用等权标准差(Barra 惯例);
2. **市值中性**:减去流通市值加权均值;
3. **z-score**:除以等权标准差。

Liquidity 的 log(0)=-inf(长期停牌股)会在缩尾前清为 NaN,避免污染当日整截面。

### 4.3 风格因子合成与正交化(model/cne5.py)

合成权重来自 `config/cne5_weights.yaml`,均按 Barra CNE-5 原文:

| 风格因子 | 合成公式 |
|---|---|
| residual_volatility | 0.74·dastd + 0.16·cmra + 0.10·hsigma |
| liquidity | 0.35·stom + 0.35·stoq + 0.30·stoa |
| earnings_yield | 0.68·epibs + 0.21·cetop + 0.11·etop |
| growth | 0.47·sgro + 0.24·egro + 0.18·egibs + 0.11·egibs_s |
| leverage | 0.38·mlev + 0.35·dtoa + 0.27·blev |
| size / beta / momentum / non_linear_size / book_to_price | 单描述因子,权重 1.0 |

缺失分量容错:某描述因子文件缺失或全空时跳过该分量(权重不重新归一),全空则跳过整个风格因子。

**正交化**(按 Barra 原文消除共线性,回归权重=√流通市值,残差后再标准化):
- nlsize:描述层即完成(对 Size 的立方回归残差,regression-weighted);
- residual_volatility:合成后逐日对 **beta + size** 双正交(主文档 3.3/附录 A);
- liquidity:合成后逐日对 **size** 正交(主文档 3.3/附录 A)。

### 4.4 因子收益率回归(model/factor_return.py)

对每个交易日做带约束的 WLS 截面回归,解释变量 `X = [country, 行业哑变量(中信一级), 10 个风格因子]`:

```
V = diag(√流通市值 / Σ√流通市值)            市值开方加权
R = 行业约束矩阵(消除行业哑变量共线性)      末行业系数 = −w_i / w_last,约束 Σ w_i·f_industry = 0
W = R · (R'X'VXR')⁻¹ · R'X'V
f = W · r                                    当日因子收益率
```

**每日回归样本构建(三层防御,Barra 规范)**:
1. **估计域**:有行业归属 + 当日有流通市值 + 当日有收益(停牌股剔除)的股票,样本 ~4400-5200 只;
2. **缺失暴露填 0**:风格因子已截面标准化,0 = 截面均值 = 中性暴露。不再要求 10 因子全覆盖
   (旧版 dropna 使样本萎缩到一致预期覆盖的 ~1400 只,小行业被整体剔空导致矩阵奇异/垃圾值);
3. **动态列**:当日全空的风格列(如 momentum 预热期)与空行业不进入当日回归,
   行业哑变量与约束 R 基于当日最终样本自洽计算;
4. **数值防御**:求逆前检查条件数(阈值 1e10),超过阈值或 inv 奇异时改用 pinv 最小范数解并告警,
   条件数非有限则跳过该日——杜绝裸 inv 接近奇异时静默返回垃圾值。

产出 `f_ret.csv`(index=日期,列=[country, 行业…, 风格…] 共 41 列)。
回归起始日默认 `config/date_range.factor_return_start_date`(2022-01-01:
momentum 需 504 个交易日预热,行情数据自 2020-01 起,更早日期该列全空),`--start` 可覆盖。

---

## 五、CNE-5 因子口径与 Barra 原文对照

本节对照两份 Barra 官方文档:**CNE5 Descriptor Details**(MSCI, 2013-09,描述因子定义)与
**China Equity Model CNE5 Empirical Notes**(2012-07,主文档/附录 A,风格因子与正交化要求)。
两份原文在个别口径上表述不同(见 5.2.4/5.2.7),本项目按较新的 Descriptor Details 为准并在对照中注明。

> 本节呈现最终口径与 Barra 原文的对照。

### 5.1 对照总表

| 风格因子 | 原文定义(摘要) | 本项目实现 | 结论 |
|---|---|---|---|
| Size (lncap) | 总市值的自然对数 | ln(market_cap),标准化 | ✅ 一致 |
| Non-linear Size (nlsize) | 标准化 Size 暴露的立方,对 Size 回归加权正交,再缩尾标准化 | 同左,回归权重 √流通市值 | ✅ 一致 |
| Beta (beta/sigma) | 252 日窗口、半衰期 63,个股超额对市值加权市场超额回归;σ=残差 std | 同左;市场组合用中证全指代理 | ✅ 一致(见 5.2.2) |
| Momentum (rstr) | 504 日、lag 21、半衰期 126 的对数超额收益加权和 | 逐字一致 | ✅ 一致 |
| Residual Volatility (dastd/cmra/hsigma) | 0.74·DASTD+0.16·CMRA+0.10·HSIGMA;对 Beta(+Size)正交 | 同左;CMRA 按 12 个月度点;对 Beta+Size 双正交 | ✅ 一致(见 5.2.4) |
| Liquidity (stom/stoq/stoa) | 0.35/0.35/0.30;对数换手;对 Size 正交 | 公式等价实现;对 Size 正交 | ✅ 一致(见 5.2.5) |
| Book-to-Price (btop) | 最近报告普通股账面价值/当前总市值 | 1/PB(净资产/总市值) | ✅ 一致 |
| Earnings Yield (cetop/etop/epibs) | 0.68·EPIBS+0.21·CETOP+0.11·ETOP | 三分量全实现 | ✅ 一致(见 5.2.7) |
| Growth (egro/sgro/egibs/egibs_s) | 0.47·SGRO+0.24·EGRO+0.18·EGIBS+0.11·EGIBS_s | 四分量全实现 | ✅ 一致(见 5.2.8) |
| Leverage (mlev/dtoa/blev) | 0.38/0.35/0.27;MLEV=(ME+PE+LD)/ME 等 | 同左 | ✅ 一致(见 5.2.9) |

### 5.2 逐因子对照

#### 5.2.1 Size / Non-linear Size

> 原文:"Natural log of market cap"、NLSIZE:"First, the standardized Size exposure (i.e., log of
> market cap) is cubed. The resulting factor is then orthogonalized to the Size factor on a
> regression-weighted basis. Finally, the factor is winsorized and standardized."

- 实现:lncap = ln(总市值) → 标准化;nlsize = 标准化 LNCAP³ 对 LNCAP 做 √流通市值加权 WLS 回归取残差 → 再缩尾标准化。
- 要点:立方回归在**标准化后**的 Size 暴露(均值≈0、std≈1)上进行,而非原始 log(市值)上;
  回归权重采用 √流通市值(对应原文 regression-weighted 的市值回归权重口径)。

#### 5.2.2 Beta

> 原文:"slope coefficient in a time-series regression of excess stock return against the cap-weighted
> excess return of the estimation universe… trailing 252 trading days… half-life of 63 trading days."

- 实现:252 日滚动 WLS(半衰期 63),个股超额收益 ~ 基准超额收益;窗口内缺失>42 日跳过(实现细节,原文未写容差);σ=残差时序标准差。
- 近似:原文用市值加权估计域的超额收益,实现用**中证全指(000985.XSHG)日收益率**作为市场组合代理。

#### 5.2.3 Momentum

> 原文:"sum of excess log returns over the trailing T=504 trading days with a lag of L=21 trading
> days… exponential weight with a half-life of 126 trading days."

- 实现:shift(21) 后 504 日滚动窗口,权重半衰期 126,窗口内至少 483 个有效值(483=504−21)。与原文逐字一致。

#### 5.2.4 Residual Volatility

> 原文:0.74·DASTD + 0.16·CMRA + 0.10·HSIGMA;DASTD 为过去 252 日超额收益波动率(半衰期 42);
> Z(T) = Σ[ln(1+r)−ln(1+rf)],T=1..12(每月=21 交易日),CMRA = Z_max − Z_min;HSIGMA = std(e_t)(即式(1)回归残差)。

- 实现:DASTD=252 日加权标准差(半衰期 42);CMRA 取 **12 个月度累计点**的极差(max−min,不加权);
  HSIGMA 直接复用 beta 产出的 sigma(两者本就是同一次回归的残差标准差,避免重复回归)。
- 原文差异说明(两份文档):
  - 主文档(2012-07)式(A4)写作 CMRA = ln(1+Z_max) − ln(1+Z_min);Descriptor Details(2013-09,更新)式(4)为 Z_max − Z_min(直接极差)。**本项目按较新的 Descriptor Details 实现**。
  - Descriptor Details 注记仅要求对 Beta 正交;主文档 3.3 明确要求对 **Beta 与 Size** 双正交。**本项目按主文档对 Beta+Size 双正交**。

#### 5.2.5 Liquidity

> 原文:0.35·STOM + 0.35·STOQ + 0.30·STOA;STOM = ln(Σ V_t/S_t,21 日);
> STOQ = ln((1/3)Σ exp(STOM_τ));STOA = ln((1/12)Σ exp(STOM_τ));对 Size 正交。

- 实现:stom=ln(Σ21日换手率)、stoq=ln(Σ63日/3)、stoa=ln(Σ252日/12),截面标准化后合成,并对 Size 正交。
- 等价性:exp(STOM_τ)=单月换手合计,故原文 STOQ/STOA 与"Σ63/3、Σ252/12 再取对数"数学等价;每日换手之和 ÷ 月数 = 月均换手。
- 近似:原文 V_t/S_t 用总股本(shares outstanding),rqdatac 换手率为**流通股本**口径。

#### 5.2.6 Book-to-Price

> 原文:"Last reported book value of common equity divided by current market capitalization."

- 实现:btop = 1/PB,即净资产(归母)/总市值,与原文口径一致。

#### 5.2.7 Earnings Yield

> 原文(Descriptor Details):0.68·EPIBS + 0.11·ETOP + 0.21·CETOP;ETOP=trailing 12-month earnings/当前市值;
> CETOP=trailing 12-month cash earnings/当前价格;EPIBS=分析师预测的盈市率。
> 主文档:0.68·EPFWD + 0.21·CETOP + 0.11·ETOP(权重数值一致,EPFWD=12 个月前瞻盈利/市值,当前与下一财年预测加权)。

- 实现:EPIBS = comp_con_eps_ftm(一致预期 EPS,FTM)/ 不复权收盘价(每股口径等价于前瞻盈利/市值);
  ETOP = ep_ratio_ttm(归母净利润 TTM/总市值,TTM 口径与原文一致);
  CETOP = 1/pcf_ratio_ttm(经营现金流 TTM/市值,cash earnings 取经营现金流口径)。
- 近似:一致预期数据仅覆盖分析师跟踪股票(EPIBS 覆盖率约 45%),缺失股票合成时自动由 ETOP+CETOP 补全。

#### 5.2.8 Growth

> 原文:0.47·SGRO + 0.24·EGRO + 0.18·EGIBS + 0.11·EGIBS_s;
> EGRO/SGRO = 过去 5 个财年的年报每股盈利/每股营收对时间回归,斜率 ÷ 均值;
> EGIBS/EGIBS_s = 分析师预测的长期/短期盈利增长。

- 实现:EGRO = 近 5 年年报 EPS 对年份 OLS 斜率 ÷ EPS 均值;SGRO = 近 5 年年报营业收入对年份 OLS 斜率 ÷ 营收均值
  (年报取数 `get_pit_financials_ex`,严格 PIT;每个年报年用回溯 5 年滚动计算,映射到次年 4/30 生效并前向填充);
  EGIBS/EGIBS_s = 一致预期净利润增长率(T+3 / FTM)。
- 近似(数据源固有限制):
  - 原文 SGRO 用**每股营收**(sales per share),rqdatac 无现成字段,实现用**营收总额**——斜率÷均值在股本不变时等价,A 股送转/增发频繁时股本变动股票会混入股本扩张成分;
  - 原文 EGIBS 为 **EPS 增长率**,rqdatac 只有净利润增长率,以净利润增长代理。

#### 5.2.9 Leverage

> 原文:0.38·MLEV + 0.35·DTOA + 0.27·BLEV;
> MLEV=(ME+PE+LD)/ME,ME 为最后交易日普通股市值;DTOA=TD/TA,TD=长期债务+流动负债;BLEV=(BE+PE+LD)/BE。

- 实现:MLEV=(ME+PE+LD)/ME(ME 用**当日**总市值,与原文 "on the last trading day" 一致);
  DTOA=总负债/总资产;BLEV=(BE+PE+LD)/BE(归母权益);财务宽表(PIT 对齐)直接按日频计算。
- 近似:
  - PE(优先股)rqdatac 不提供,按 0(A 股优先股极少);
  - LD 用**非流动负债**代理原文的 long-term debt;
  - DTOA 的 TD 用**总负债**(含其他非流动负债,如递延所得税负债),与原文"长期债务+流动负债"有口径差异,对多数公司影响较小。

### 5.3 标准化、正交化与回归约束对照

| 项 | Barra 原文 | 本项目实现 |
|---|---|---|
| 因子标准化 | 市值加权均值 0、等权标准差 1(主文档 3.3);NLSIZE 先缩尾再标准化 | 缩尾(±3σ,μ=流通市值加权、σ=等权)→ 流通市值加权去均值 → 等权 z-score |
| Residual Volatility 正交化 | 对 Beta 与 Size 双正交(主文档 3.3/附录 A;Descriptor Details 仅写 Beta) | 合成后对 [beta, size] 做 √流通市值加权 WLS 取残差,再标准化 |
| Liquidity 正交化 | 对 Size 正交(主文档 3.3/附录 A) | 合成后对 [size] 正交,同上流程 |
| NLSIZE 正交化 | 对 Size regression-weighted 正交 | 描述层完成,√流通市值加权 WLS |
| 行业约束 | 行业因子收益率的市值加权和为 0 | 约束矩阵 R 末行业系数 −w_i/w_last,约束 Σ w_i·f_industry = 0 |

### 5.4 数据源固有限制(不可消除的近似)

| 限制 | 影响 | 原因 |
|---|---|---|
| 一致预期覆盖约 45% | earnings_yield 非空率约 36%、growth 约 43% | EPIBS/EGIBS 依赖分析师跟踪股票,小盘股天然缺失 |
| 优先股字段缺失 | Leverage 的 PE 按 0 | rqdatac 因子库不提供 `oth_eqt_tools_p_shr` |
| 财务字段口径 | LD 用非流动负债代理长期借款;TD 用总负债 | rqdatac 字段限制 |
| 市场组合 | Beta 回归基准用中证全指 | 原文为市值加权估计域 |
| 换手率口径 | 流通股本 vs 原文总股本 | `get_turnover_rate` 口径 |

---

## 六、扩展指南(核心价值)

### 估值表风格暴露测算(analysis/exposure.py)

对真实私募 4 级科目估值表(样例见 `data_store/估值表/`)计算 10 大风格因子暴露:

```bash
# 估值表 -> 基金股票部分暴露(估值日期自动从表内提取,结果存 data_store/analysis/)
python -m analysis.exposure fund "data_store/估值表/SEZ753_赫富1000指数增强一号_4级科目估值表_20260430.xls"

# 指数成分权重 -> 指数暴露(需先 data 层下载 index_weights)
python -m analysis.exposure index data_store/base/index_weights/000300.XSHG.csv
```

适配要点:
- 股票持仓行 = 14 位 4 级科目码(`1102` + 2 位账户/板块 + `01` 主仓 + 6 位股票代码),
  仅取 x01 主仓行,天然排除 x99 估值增值行与红利税科目;信用账户/科创板同样匹配;
- 权重取"市值占净值%"(百分数),股票内归一化,暴露口径为"股票部分"(打印仓位注明,
  股指期货 3102/现金等非股票科目不纳入);
- 6 位代码直接加 rqdatac 后缀(.XSHG/.XSHE/.XBJG)与 cne_5.csv 对齐;
  merge 不到因子的股票按 0 中性暴露计入并告警;
- 估值日非交易日时回退最近因子日;指数入口单日权重取最近"数据完整"因子日
  (最新交易日估值字段可能尚未发布)。

实跑样例(赫富1000指增 2026-04-30,592 只股票、仓位 92.3%):
size -1.33 / non_linear_size +0.51(显著小盘,符合 1000 指增特征)。

### 基于净值的收益归因(analysis/attribution.py)

从净值序列文件提取日度净值,对 CNE-5 因子收益率做 252 日滚动 OLS,
逐日分解 Alpha/国家/行业/风格贡献(每日四项之和 = 当日净值收益,机器精度恒等):

```bash
python -m analysis.attribution "data_store/净值表/极量精选指增_净值序列_20260703.xlsx" [--window 252] [--show]
```

- 净值文件格式:表头第 0 行,日期列(净值日期/日期/估值日期)+ 净值列
  (优先单位净值;单位≠累计时提示分红并改用累计净值近似),Excel/CSV 均可;
- 产出(data_store/analysis/):归因明细 CSV、累计贡献 CSV、归因对比 PNG;
- 年化贡献摘要打印,合计与净值年化严格一致。

### 加一个新风格因子(如 LongTermReversal)
1. `factors/cne5/reversal.py` 写计算函数(复用 `factors/common.py` 工具)
2. `factors/cne5/registry.py` 注册一行:`"reversal": {"func": ..., "outputs": [...], "depends_on": [...]}`
3. `config/cne5_weights.yaml` 加该风格因子的合成权重
4. `pipeline.py` 自动按 registry 拓扑排序执行,**无需改其他文件**

### 加一个 Barra 新版本(如 CNE-6)
1. `factors/cne6/` 填因子文件 + `registry.py` 注册
2. `config/cne6_weights.yaml` 填权重
3. `model/cne6.py` 实现风格合成
4. `python pipeline.py --version cne6` 即可;`common.py` 和整个 `data/` 跨版本共享

### 加一个新分析(如 Brinson 归因)
1. `analysis/brinson.py` 加文件,按需指定用哪个版本的因子产出
2. 无需改动 data/factors/model

---

## 七、验证说明

由于 license 需自行填入,改造完成后的验证分两阶段:

1. **阶段一(静态)**:代码结构完整、字段映射标注、import 关系正确、registry 依赖图正确。
2. **阶段二(实跑)**:填 license 后单步跑 `data` 验证字段名/单位 → 跑 `factors` 对比因子值与原项目一致性 → 跑 `model` 产出 cne_5.csv 和 f_ret.csv。

实跑基线(全量重算):

| 验证项 | 结果 |
|---|---|
| 描述因子 | 22 个全部落盘,10 大风格因子 761 万行(cne_5.csv) |
| 因子收益率 | 1117 个交易日(2022-01 起)× 41 列(country + 30 行业 + 10 风格),逐日样本 4400~5200 只 |
| 正交化效果 | 正交后 RV 与 beta/size、liquidity 与 size 的 √市值加权截面相关(均值绝对值)< 0.01 |
| 行业约束 | 新约束下 Σ w_i·f_industry 最大误差 4.0e-6 ≈ 0(2022-01~2026-08 逐日抽样重算) |
| 因子收益量级 | 逐日 \|f\| 最大值中位数 0.022、极值 0.09;country 日均值 0.07%/std 1.1%,符合 A 股波动 |
| 非空率 | size/nlsize/btop/leverage 97.5%、liquidity 82.9%、beta 81.4%、RV 80.5%、momentum 63.7%、growth 42.7%、earnings_yield 36.4%(后两者受一致预期覆盖限制,见 5.4) |
| 量级抽样 | 万科/保利 mlev≈+6.1(高杠杆)、茅台 mlev≈−0.5(低杠杆);茅台 EGRO≈15.7%/SGRO≈15.0%,符合经济直觉 |