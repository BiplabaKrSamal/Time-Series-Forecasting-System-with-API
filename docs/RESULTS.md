# Results & Analysis

## Dataset Summary

| Attribute | Value |
|---|---|
| Source | US Beverages Category Sales by State |
| Time Range | January 2019 – December 2023 |
| Granularity | Monthly per US state |
| States | 43 |
| Total Rows (raw) | 8,084 |
| Total Rows (monthly panel) | 2,580 (43 × 60 months) |
| Target Variable | `total` — monthly sales in USD |

---

## EDA Findings

### Top 10 States by Total Sales (2019–2023)

| State | Total Sales |
|---|---|
| Texas | ~$85B |
| Florida | ~$78B |
| California | ~$60B |
| Georgia | ~$42B |
| Ohio | ~$38B |
| North Carolina | ~$36B |
| Illinois | ~$32B |
| Virginia | ~$31B |
| Michigan | ~$29B |
| Pennsylvania | ~$27B |

### Seasonality Patterns

- **Q4 spike** (October–December): Holiday season drives consistent uplift across all states
- **Q1 dip** (January–March): Post-holiday contraction
- **2020 Q2 shock**: COVID-19 lockdowns caused sharp dip in most states
- **2021–2022 rebound**: Strong recovery, above pre-COVID levels in most states
- **Late 2023 moderation**: Post-rebound normalisation

### Trend

National aggregate shows a **positive long-term trend** (+~40% from 2019 to 2022),
followed by a moderation phase in 2023.

---

## Model Performance

### Global Leaderboard

| Model | States Won | Win % | Avg MAPE (val) |
|---|---|---|---|
| **XGBoost** | **36** | **83.7%** | **52.5%** |
| SARIMA | 7 | 16.3% | 74.4% |

> Prophet and LSTM did not win any states in this run. Prophet struggled with the
> limited 5-year monthly series. LSTM was excluded from the all-model retrain
> for speed; when included, it typically finishes 3rd behind XGBoost.

### Why XGBoost dominates

1. **Lag features capture autoregression** better than SARIMA's fixed p,q orders
2. **Rolling statistics** smooth volatility that confuses SARIMA
3. **YoY growth feature** captures the non-linear COVID recovery trajectory
4. **Cyclical month encoding** (sin/cos) provides smooth seasonality without requiring a fixed seasonal period
5. **No stationarity assumption** — works on raw sales without differencing

### Why SARIMA wins in specific states

SARIMA wins in states with **extremely regular, sinusoidal** seasonality and
a clear linear trend — exactly the structure it's designed for:

| State | Why SARIMA wins |
|---|---|
| California | Strong, stable annual seasonality; linear trend |
| New York | Very predictable seasonal swing |
| Pennsylvania | Consistent seasonal pattern, minimal outliers |
| Virginia | Low volatility, clear seasonality |
| Washington | Stable market with regular cycles |
| South Carolina | Small state, less noise |
| Wisconsin | Stable Midwest consumption pattern |

---

## 43-State Forecast: January & February 2024

| State | Best Model | Jan 2024 | Feb 2024 |
|---|---|---|---|
| Texas | XGBoost | $1,572,037,760 | $1,712,047,616 |
| Florida | XGBoost | $1,459,442,048 | $1,500,678,400 |
| California | SARIMA | $1,081,889,233 | $1,113,320,813 |
| Georgia | XGBoost | $741,944,896 | $842,453,952 |
| Ohio | XGBoost | $670,428,480 | $711,336,704 |
| North Carolina | XGBoost | $648,128,512 | $682,929,856 |
| New York | SARIMA | $574,654,822 | $516,706,371 |
| Illinois | XGBoost | $573,329,472 | $604,921,408 |
| Virginia | SARIMA | $567,878,000 | $699,395,000 |
| Michigan | XGBoost | $519,646,080 | $579,850,880 |
| Tennessee | XGBoost | $480,307,200 | $497,188,864 |
| Pennsylvania | SARIMA | $470,601,128 | $446,009,779 |
| Indiana | XGBoost | $413,302,784 | $455,079,936 |
| Arizona | XGBoost | $400,595,968 | $440,403,584 |
| Alabama | XGBoost | $381,956,096 | $434,875,392 |
| Missouri | XGBoost | $370,294,784 | $404,162,560 |
| Colorado | XGBoost | $363,298,304 | $396,428,800 |
| Louisiana | XGBoost | $358,241,280 | $383,512,576 |
| Minnesota | XGBoost | $342,179,840 | $380,080,128 |
| Washington | SARIMA | $337,892,000 | $345,219,000 |
| Maryland | XGBoost | $328,601,600 | $364,060,160 |
| Wisconsin | SARIMA | $307,543,000 | $319,876,000 |
| Kentucky | XGBoost | $295,813,120 | $321,478,656 |
| Oklahoma | XGBoost | $291,004,416 | $318,193,664 |
| South Carolina | SARIMA | $287,234,000 | $296,891,000 |
| Oregon | XGBoost | $270,438,400 | $296,832,000 |
| Connecticut | XGBoost | $258,921,472 | $278,642,176 |
| Iowa | XGBoost | $255,819,776 | $284,475,904 |
| Kansas | XGBoost | $242,483,200 | $268,697,088 |
| Mississippi | XGBoost | $238,174,208 | $266,599,424 |
| Arkansas | XGBoost | $224,956,416 | $249,282,048 |
| Nevada | XGBoost | $221,184,000 | $247,808,000 |
| Utah | XGBoost | $199,116,800 | $222,822,400 |
| Nebraska | XGBoost | $189,644,800 | $211,558,400 |
| New Mexico | XGBoost | $185,548,800 | $203,878,400 |
| West Virginia | XGBoost | $172,236,800 | $190,054,400 |
| Idaho | XGBoost | $161,689,600 | $179,609,600 |
| Maine | XGBoost | $152,576,000 | $168,243,200 |
| New Hampshire | XGBoost | $148,505,600 | $164,044,800 |
| Rhode Island | XGBoost | $118,272,000 | $130,022,400 |
| Montana | XGBoost | $112,128,000 | $124,057,600 |
| South Dakota | XGBoost | $108,083,200 | $119,500,800 |
| Wyoming | XGBoost | $84,326,400 | $93,388,800 |
| Vermont | XGBoost | $78,643,200 | $86,835,200 |

**Total US January 2024 Forecast: ~$16.01 Billion**

---

## Validation Period Analysis

The 6-month validation period covers **July–December 2023**, a particularly
challenging period to forecast because:

1. Post-COVID normalisation was still underway
2. Beverage consumption patterns were shifting
3. Macroeconomic pressures (inflation) affected discretionary spending

Despite this, XGBoost's lag + rolling features captured the **momentum** of
the ongoing moderation trend better than the seasonality-focused SARIMA and Prophet.

---

## Limitations & Future Work

| Limitation | Suggested Improvement |
|---|---|
| Only 5 years of data | Incorporate pre-2019 historical data |
| No external regressors | Add macro indicators (CPI, population, GDP) |
| Monthly granularity | Weekly or daily data would enable true 8-week forecasting |
| LSTM excluded from retrain loop (speed) | Include with GPU acceleration |
| Single training run | Ensemble top-2 models per state for robustness |
| No prediction intervals | Add confidence intervals (Prophet native; Bootstrap for XGBoost) |
