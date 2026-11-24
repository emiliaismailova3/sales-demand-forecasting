# %% [markdown]
# # Sales Demand Forecasting: daily orders on the Olist marketplace
#
# **Task:** forecast the number of orders per day for the next **28 days** for a Brazilian
# e-commerce marketplace (Olist, ~99k orders, 2016–2018).
#
# **Why it matters:** a 4-week demand forecast drives warehouse staffing, courier capacity,
# customer-support shifts and marketing budgets.
#
# **Approach**
# 1. Build a clean daily series and choose a stable period (verified on a plot).
# 2. EDA: trend, weekly seasonality, Black Friday and two real 2018 disruptions.
# 3. Evaluation design: **every decision is made on 8 rolling validation windows**; the last
#    28 days are the test set and are used exactly once, at the end.
# 4. Models with the same 28-day horizon: **seasonal naive** baseline, **SARIMA**, **Prophet** with
#    Brazilian holidays, **LightGBM** with calendar + lag features, and their averages.
# 5. Metrics: WAPE (main), MAPE, MAE, RMSE, MASE, and the coverage of the forecast intervals.
# 6. Many series: forecasts for the 5 largest states + the rest, with a global LightGBM model
#    and bottom-up totals.

# %%
import logging
import warnings
from itertools import product
from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)

SEED = 42
HORIZON = 28  # forecast horizon in days = size of the test set

# Figures go to ../reports/figures when run inside the repo, to ./figures elsewhere (e.g. Kaggle)
FIG_DIR = Path("../reports/figures") if Path("../reports").exists() else Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Chart colours: actual = ink, baseline = grey, models = fixed categorical order
C = {"actual": "#0b0b0b", "naive": "#8a8986", "SARIMA": "#2a78d6",
     "Prophet": "#eb6834", "LightGBM": "#1baf7a", "Ensemble": "#8e5ad6", "muted": "#52514e"}
plt.rcParams.update({
    "figure.figsize": (11, 4), "figure.dpi": 110, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.edgecolor": "#8a8986", "axes.titleweight": "bold",
    "axes.titlesize": 12, "legend.frameon": False, "lines.linewidth": 1.6,
})

# %% [markdown]
# ## 1. Load the data
#
# One file is needed: `olist_orders_dataset.csv` (one row per order, with its status and purchase
# timestamp). The cell looks for it on Kaggle (`olistbr/brazilian-ecommerce` attached as input),
# then in `../data/`, and otherwise downloads it from Olist's public GitHub repository.

# %%
CANDIDATES = [
    Path("/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv"),
    Path("../data/olist_orders_dataset.csv"),
    Path("data/olist_orders_dataset.csv"),
]
SOURCE_URL = ("https://raw.githubusercontent.com/olist/work-at-olist-data/"
              "master/datasets/olist_orders_dataset.csv")

path = next((p for p in CANDIDATES if p.exists()), None)
orders = pd.read_csv(path or SOURCE_URL, parse_dates=["order_purchase_timestamp"])
print(f"Loaded from: {path or SOURCE_URL}")
print(f"{len(orders):,} orders, {orders.order_purchase_timestamp.min():%Y-%m-%d} → "
      f"{orders.order_purchase_timestamp.max():%Y-%m-%d}")
orders.order_status.value_counts()

# %% [markdown]
# ## 2. Build the daily series
#
# * **Remove `canceled` and `unavailable` orders**: they are not real demand that has to be
#   fulfilled. Every other status (delivered, shipped, invoiced, ...) is a placed order.
# * Count orders per calendar day of `order_purchase_timestamp` (the moment the customer
#   bought). `asfreq("D")` makes sure days without orders appear as 0 instead of disappearing.

# %%
placed = orders[~orders.order_status.isin(["canceled", "unavailable"])]
daily_all = (placed.set_index("order_purchase_timestamp")
             .resample("D").size().asfreq("D", fill_value=0).rename("orders"))
print(f"Removed {len(orders) - len(placed):,} canceled/unavailable orders")

fig, ax = plt.subplots()
ax.plot(daily_all, color=C["actual"], lw=0.9)
for d, label in [("2017-02-01", "start 2017-02-01"), ("2018-08-20", "end 2018-08-20")]:
    ax.axvline(pd.Timestamp(d), color=C["SARIMA"], ls="--", lw=1)
    ax.annotate(label, (pd.Timestamp(d), 900), xytext=(4, 0), textcoords="offset points",
                color=C["muted"], fontsize=9)
ax.set(title="Daily orders, full history: choosing the stable period", ylabel="orders per day")
fig.tight_layout(); fig.savefig(FIG_DIR / "01_full_history.png", dpi=150); plt.show()

# %% [markdown]
# **Choosing the period (verified on the plot and in the numbers below):**
# * **Start 2017-02-01.** 2016 has only a handful of days with orders, and January 2017 is
#   still the platform's ramp-up (it starts at zero and averages ~25 orders a day, against
#   60–100 a day from February on).
#   Fitting on the ramp-up would teach the models a growth rate that no longer exists.
# * **End 2018-08-20.** After that the count collapses (186 → 11 per day within a week, then
#   zero). That is the dataset export being cut off, not a real fall in demand.

# %%
print("January 2017 (ramp-up):", daily_all["2017-01-01":"2017-01-31"].describe()[["mean", "min", "max"]].round(1).to_dict())
print("End of the export:")
print(daily_all["2018-08-17":"2018-09-03"].to_string())

# %%
y = daily_all["2017-02-01":"2018-08-20"].astype(float)
print(f"Series: {y.index.min():%Y-%m-%d} → {y.index.max():%Y-%m-%d}, {len(y)} days, "
      f"days with zero orders: {(y == 0).sum()}")

# %% [markdown]
# ## 3. Exploratory analysis
#
# ### 3.1 Trend, Black Friday and 2018 disruptions

# %%
# Scheduled events are on the calendar in advance; the strike was a surprise.
SCHEDULED = {"Black Friday", "FIFA World Cup"}
EVENTS = {  # dates are public knowledge, fixed before looking at the test period
    "Black Friday": ("2017-11-24", "2017-11-27"),
    "Truckers' strike": ("2018-05-21", "2018-05-31"),
    "FIFA World Cup": ("2018-06-14", "2018-07-15"),
}

fig, ax = plt.subplots()
ax.plot(y, color=C["muted"], lw=0.7, alpha=0.6, label="daily orders")
ax.plot(y.rolling(28, center=True).mean(), color=C["actual"], lw=2, label="28-day rolling mean")
for name, (a, b) in EVENTS.items():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color=C["Prophet"], alpha=0.15, lw=0)
ax.annotate("Truckers' strike", (pd.Timestamp("2018-05-21"), 560), xytext=(-4, 0), textcoords="offset points",
            ha="right", fontsize=9, color=C["muted"])
ax.annotate("FIFA World Cup", (pd.Timestamp("2018-07-16"), 560), xytext=(4, 0), textcoords="offset points",
            fontsize=9, color=C["muted"])
ax.annotate(f"Black Friday 2017-11-24: {y['2017-11-24']:.0f} orders", (pd.Timestamp("2017-11-24"), y["2017-11-24"]),
            xytext=(-210, -8), textcoords="offset points", fontsize=9, color=C["muted"],
            arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.8))
ax.set(title="Growth through 2017, a plateau in 2018, and three special events", ylabel="orders per day")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIG_DIR / "02_trend_events.png", dpi=150); plt.show()

# %%
bf = y["2017-11-17":"2017-11-30"]
print(f"Black Friday 2017-11-24: {y['2017-11-24']:.0f} orders vs "
      f"{y['2017-11-10':'2017-11-23'].mean():.0f} on an average day of the two weeks before (×{y['2017-11-24'] / y['2017-11-10':'2017-11-23'].mean():.1f})")
weekly = y.resample("W-SUN").mean()
print("\nWeekly mean orders per day, spring-summer 2018:")
print(weekly["2018-05-06":"2018-08-19"].round(0).astype(int).to_string())

# %% [markdown]
# **What the plot shows**
# * **Trend:** strong growth through 2017 (~100 → ~250 orders a day), then a plateau of
#   ~230–270 in 2018.
# * **Black Friday (24.11.2017):** 1,166 orders, about **6× a normal day**, with an elevated
#   tail over the following weekend. A one-off outlier that a model has to treat as a special
#   day, otherwise it inflates the trend and the variance.
# * **Truckers' strike (21–31 May 2018):** a nationwide road-freight strike in Brazil;
#   the weekly average fell to ~135 orders a day.
# * **FIFA World Cup (14 June–15 July 2018):** demand dropped during the tournament, with a
#   second dip after Brazil was knocked out on 6 July, and recovered immediately after the
#   final.
#
# The last two matter a lot for us: **they sit right before the test window**, and demand
# recovers just as the test starts. A model that reads the World Cup dip as a new downward
# trend will under-forecast. This is a realistic forecasting problem: recent history is not
# always representative.

# %% [markdown]
# ### 3.2 Weekly seasonality

# %%
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
calm = y.drop(pd.concat([y[a:b] for a, b in EVENTS.values()]).index)  # exclude special events
dow_index = calm.groupby(calm.index.dayofweek).mean() / calm.mean()

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(dow_names, dow_index.values, color=C["SARIMA"], width=0.6)
ax.axhline(1, color=C["muted"], lw=0.8, ls="--")
for b, v in zip(bars, dow_index.values):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=9, color=C["muted"])
ax.set(title="Orders by day of week (1.00 = average day)", ylabel="relative level", ylim=(0, 1.25))
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(FIG_DIR / "03_weekday.png", dpi=150); plt.show()

# %% [markdown]
# A clear **weekly cycle**: Monday and Tuesday are the busiest days, and Saturday is ~25%
# below average. So the models need a **7-day seasonality**.
#
# The STL decomposition below separates trend, weekly seasonality and the residual. The
# weekly swing grows as the level grows (smaller in early 2017, larger later), so the
# seasonality is roughly **multiplicative**: a percentage of the level, not a fixed number of
# orders. That is why SARIMA is fitted on `log(orders)` and Prophet uses multiplicative
# seasonality.

# %%
stl = STL(y, period=7, robust=True).fit()
fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
for ax, (comp, title) in zip(axes, [(stl.trend, "Trend"), (stl.seasonal, "Weekly seasonality"), (stl.resid, "Residual")]):
    ax.plot(comp, color=C["actual"], lw=0.9); ax.set_title(title, loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG_DIR / "04_stl.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Evaluation design: choose on validation windows, test once
#
# **No random split.** Neighbouring days are strongly correlated, so a random split would put
# days *between* known neighbours into the test set: the model would be interpolating, not
# forecasting. Every evaluation below imitates production: fit on the past, forecast the next
# 28 days.
#
# **One test window is not enough to choose models,** and choosing on it is cheating. So the
# protocol is:
#
# 1. **Test set: the last 28 days (2018-07-24 → 2018-08-20).** Used exactly once, at the very end.
# 2. **8 validation windows:** the eight 28-day periods before the test (Dec 2017 → Jul 2018).
#    For each one, every model is refitted on the data before it and forecasts it (rolling origin).
# 3. **Every decision is made on the validation windows only:** the SARIMA orders, whether to
#    use the special-event information, and which models to combine in the ensemble.
#
# **Events and what was known when.** A forecaster knows the calendar but not the news:
# * **Scheduled events** (Black Friday, the FIFA World Cup) are announced long in advance, so the
#   model gets their full dates, also when they fall inside the forecast window.
# * **Unscheduled events** (the truckers' strike) are only known once they have started, and only
#   up to the forecast date. In the window starting 2018-05-01 nobody knew about the strike, so
#   the model does not either.
#
# Section 9.4 repeats the whole selection with a stricter rule (scheduled events treated as
# surprises too) to show how much the conclusion depends on this choice.

# %%
N_VALID = 8
WINDOW_ENDS = [len(y) - k * HORIZON for k in range(N_VALID, -1, -1)]   # 9 windows, the last one is the test
train, test = y.iloc[:-HORIZON], y.iloc[-HORIZON:]
dev_start = y.index[WINDOW_ENDS[0] - HORIZON]

windows = pd.DataFrame({
    "role": ["validation"] * N_VALID + ["test"],
    "first day": [y.index[e - HORIZON].date() for e in WINDOW_ENDS],
    "last day": [y.index[e - 1].date() for e in WINDOW_ENDS],
    "mean orders/day": [round(y.iloc[e - HORIZON:e].mean()) for e in WINDOW_ENDS],
})
print(f"Test: {test.index.min():%Y-%m-%d} → {test.index.max():%Y-%m-%d}, mean {test.mean():.0f} orders/day, "
      f"min {test.min():.0f}")
windows

# %% [markdown]
# ## 5. Metrics
#
# * **WAPE** (main metric): total absolute error divided by total orders. "We miss 9% of the
#   orders over the 4 weeks." Unlike MAPE, a single low day cannot blow it up.
# * **MAPE**: average absolute error in % of each day's actual. Familiar, but it divides by the
#   actual value, so it explodes when actuals are near zero (it is safe here: the smallest
#   test value is 175 orders) and it punishes over-forecasts more than under-forecasts.
# * **MAE / RMSE**: errors in orders; RMSE penalises large misses more (one bad day can mean
#   no couriers).
# * **MASE**: MAE divided by the in-sample MAE of the "same weekday last week" forecast.
#   Below 1 means better than that simple rule; it is comparable across series of any size.

# %%
def metrics(actual, forecast, insample=None) -> dict:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    out = {"MAE": np.mean(np.abs(a - f)),
           "RMSE": np.sqrt(np.mean((a - f) ** 2)),
           "MAPE, %": np.mean(np.abs((a - f) / a)) * 100,
           "WAPE, %": np.abs(a - f).sum() / a.sum() * 100}
    if insample is not None:
        s = np.asarray(insample, float)
        out["MASE"] = out["MAE"] / np.mean(np.abs(s[7:] - s[:-7]))
    return out

# %% [markdown]
# ## 6. The models
#
# All models forecast the **same 28 days at once from the same history.**
#
# ### 6.1 Seasonal naive (baseline)
# "Each day equals the same weekday one week earlier." With a 28-day horizon only the last
# observed week is known, so it is **repeated four times**. Every real model has to beat this.
#
# ### 6.2 SARIMA with weekly seasonality
# **SARIMA(p,d,q)(P,D,Q,7)** models the series from its own past values (AR, `p`) and past errors
# (MA, `q`) after differencing: `d=1` removes the trend, `D=1, s=7` removes the weekly pattern,
# and `P, Q` are the seasonal terms at lag 7. It is fitted on **`log(orders)`** because the weekly
# swing is proportional to the level (see STL): the log makes it additive. The orders are chosen
# by **AIC on the data before the first validation window**, so not even the validation windows
# influence them.

# %%
log_dev = np.log(y[:dev_start - pd.Timedelta(days=1)])
grid = []
for p, q, P, Q in product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    try:
        fit = SARIMAX(log_dev, order=(p, 1, q), seasonal_order=(P, 1, Q, 7)).fit(disp=False)
        grid.append(((p, 1, q), (P, 1, Q, 7), fit.aic))
    except Exception:
        continue
grid = pd.DataFrame(grid, columns=["order", "seasonal_order", "AIC"]).sort_values("AIC")
best_order, best_seasonal = grid.iloc[0, :2]
print(f"Best by AIC on data before {dev_start:%Y-%m-%d}: SARIMA{best_order}x{best_seasonal}")
grid.head(5)

# %% [markdown]
# ### 6.3 Prophet with Brazilian holidays
# Prophet (Meta) fits **trend + weekly seasonality + holiday effects** as a regression, so each
# part is interpretable. `add_country_holidays("BR")` adds Carnival, Tiradentes, Independence
# Day and so on; multiplicative seasonality matches the STL finding; yearly seasonality is off
# (1.5 years of data cannot estimate it). Two variants: **with and without the special events**
# as extra "holidays". The validation windows decide which one we keep.
#
# ### 6.4 LightGBM with calendar and lag features
# Gradient boosting turns forecasting into a **tabular regression**: one row per day, target =
# orders, features = calendar + the past.
#
# **The leakage rule:** at forecast time we know the data up to the last training day and must
# predict up to 28 days ahead. For the last day the newest known value is 28 days old, so **every
# lag must be ≥ 28 days** and rolling means are computed on the series **shifted by 28 days**.
# With `lag_1` or `lag_7` the model would need values from inside the forecast period, which do
# not exist yet (data leakage, and an offline score no one can reproduce in production).
#
# | Feature | Meaning |
# |---|---|
# | `dayofweek`, `month`, `day` | calendar position (weekly cycle, month-end effects) |
# | `is_holiday` | Brazilian public holiday (`holidays` package) |
# | `lag_28`, `lag_35`, `lag_42` | orders on the same weekday 4, 5 and 6 weeks earlier |
# | `rolling_mean_7`, `rolling_mean_28` | average level of the week / 4 weeks ending 28 days earlier |
# | `is_event` (variant) | the day is inside a known special event; 0 for every future day |
# | `lag_28_is_event` (variant) | the day used by `lag_28` was inside an event, so that lag is not representative |
#
# ### 6.5 Ensembles
# Averages of two or three models. They are only useful if the models make **different errors**;
# the validation windows decide which combination, if any, is worth it.

# %%
br_holidays = holidays.Brazil(years=range(2016, 2019))
holiday_days = pd.to_datetime(list(br_holidays.keys()))

feat = pd.DataFrame({"orders": y})
feat["dayofweek"] = feat.index.dayofweek
feat["month"] = feat.index.month
feat["day"] = feat.index.day
feat["is_holiday"] = feat.index.isin(holiday_days).astype(int)
for lag in (28, 35, 42):
    feat[f"lag_{lag}"] = feat["orders"].shift(lag)
for w in (7, 28):
    feat[f"rolling_mean_{w}"] = feat["orders"].shift(HORIZON).rolling(w).mean()
BASE_FEATURES = ["dayofweek", "month", "day", "is_holiday", "lag_28", "lag_35", "lag_42",
                 "rolling_mean_7", "rolling_mean_28"]
EVENT_FEATURES = ["is_event", "lag_28_is_event"]
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)


def known_events(last_train_day: pd.Timestamp, strict: bool = False) -> dict:
    """What a planner knows on the forecast date.

    Scheduled events (Black Friday, the World Cup) are on the calendar, so their full dates are
    known, also when they fall inside the forecast window. An unscheduled event (the strike) is
    only known once it has started, and only up to the forecast date. With strict=True scheduled
    events are treated like surprises too (the sensitivity check in section 9.4).
    """
    horizon_end = last_train_day + pd.Timedelta(days=HORIZON)
    known = {}
    for n, (a, b) in EVENTS.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if n in SCHEDULED and not strict:
            if a <= horizon_end:
                known[n] = (a, b)
        elif a <= last_train_day:
            known[n] = (a, min(b, last_train_day))
    return known


def event_flags(index: pd.DatetimeIndex, events: dict) -> pd.Series:
    flags = pd.Series(0, index=index)
    for a, b in events.values():
        flags[a:b] = 1
    return flags


def fc_naive(tr):
    return np.tile(tr.iloc[-7:].values, HORIZON // 7), None, None


def fc_sarima(tr, log=np.log, exp=np.exp):
    res = SARIMAX(log(tr), order=best_order, seasonal_order=best_seasonal).fit(disp=False).get_forecast(HORIZON)
    ci = res.conf_int(alpha=0.2).values   # 80% interval
    return exp(res.predicted_mean.values), exp(ci[:, 0]), exp(ci[:, 1])


def fc_prophet(tr, events):
    hol = (pd.concat([pd.DataFrame({"holiday": n, "ds": pd.date_range(a, b)}) for n, (a, b) in events.items()])
           if events else None)
    m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
                seasonality_mode="multiplicative", holidays=hol, interval_width=0.8, uncertainty_samples=500)
    m.add_country_holidays(country_name="BR")
    m.fit(tr.rename_axis("ds").reset_index(name="y"))
    np.random.seed(SEED)   # Prophet samples its intervals; fix the seed so reruns match
    p = m.predict(m.make_future_dataframe(periods=HORIZON)).iloc[-HORIZON:]
    return p["yhat"].values, p["yhat_lower"].values, p["yhat_upper"].values


def fc_lgbm(tr, events, return_model=False):
    f = feat.copy()
    if events is not None:
        flags = event_flags(y.index, events)
        f["is_event"], f["lag_28_is_event"] = flags, flags.shift(28)
        # future days are flagged only for scheduled events (known_events leaves surprises out)
    cols = BASE_FEATURES + (EVENT_FEATURES if events is not None else [])
    f_tr = f.loc[tr.index].dropna(subset=cols)          # the first 42 days have no lags yet
    f_fc = f.iloc[len(tr):len(tr) + HORIZON]
    assert f_fc[cols].notna().all().all()
    model = lgb.LGBMRegressor(**LGBM_PARAMS).fit(f_tr[cols], f_tr["orders"])
    pred = model.predict(f_fc[cols])
    return (pred, model, cols) if return_model else (pred, None, None)


def forecast_window(end: int, strict: bool = False, only_event_models: bool = False) -> dict:
    """Fit every model on y[:end-28] and forecast y[end-28:end]."""
    tr = y.iloc[:end - HORIZON]
    ev = known_events(tr.index[-1], strict=strict)
    out = {"Prophet + events": fc_prophet(tr, ev), "LightGBM + events": fc_lgbm(tr, ev)}
    if not only_event_models:
        out.update({"Seasonal naive": fc_naive(tr), "SARIMA": fc_sarima(tr),
                    "Prophet, no events": fc_prophet(tr, {}), "LightGBM, no events": fc_lgbm(tr, None)})
    return out

# %% [markdown]
# ## 7. Run every model on the 8 validation windows and the test window
# This is the expensive cell (about 30 model fits per window).

# %%
runs = {e: forecast_window(e) for e in WINDOW_ENDS}
actuals = {e: y.iloc[e - HORIZON:e] for e in WINDOW_ENDS}
insample = {e: y.iloc[:e - HORIZON] for e in WINDOW_ENDS}
VALID_ENDS, TEST_END = WINDOW_ENDS[:-1], WINDOW_ENDS[-1]


def window_table(names, metric="WAPE, %", ends=WINDOW_ENDS):
    return pd.DataFrame({y.index[e - HORIZON].date(): {n: metrics(actuals[e], point(e, n))[metric] for n in names}
                         for e in ends}).T


def point(end, name):
    """Point forecast of a single model or of an ensemble 'A + B (+ C)' average."""
    if name.startswith("Avg("):
        parts = name[4:-1].split(" | ")
        return np.mean([runs[end][p][0] for p in parts], axis=0)
    return runs[end][name][0]

# %% [markdown]
# ## 8. Decisions, made on the validation windows only
# ### 8.1 Do the special events help?

# %%
variants = ["Seasonal naive", "SARIMA", "Prophet, no events", "Prophet + events", "LightGBM, no events", "LightGBM + events"]
valid_wape = window_table(variants, ends=VALID_ENDS)
valid_wape.round(1)

# %%
choice = {}
for model in ("Prophet", "LightGBM"):
    a, b = f"{model}, no events", f"{model} + events"
    choice[model] = b if valid_wape[b].median() < valid_wape[a].median() else a
    print(f"{model}: median validation WAPE {valid_wape[a].median():.1f}% without events, "
          f"{valid_wape[b].median():.1f}% with events → keep '{choice[model]}'")

# %% [markdown]
# ### 8.2 Which ensemble, if any?
# Every average of two or three of the kept models is scored on the validation windows, together
# with the single models. The one with the lowest **median** validation WAPE becomes the final
# model. Median rather than mean, because three windows are dominated by shocks no model can
# foresee (see 8.3); the mean is shown too.

# %%
base = ["SARIMA", choice["Prophet"], choice["LightGBM"]]
candidates = ["Seasonal naive", *base]
for k in (2, 3):
    for combo in __import__("itertools").combinations(base, k):
        candidates.append(f"Avg({' | '.join(combo)})")
cand_valid = window_table(candidates, ends=VALID_ENDS)
selection = pd.DataFrame({
    "median WAPE, %": cand_valid.median(), "mean WAPE, %": cand_valid.mean(),
    "windows better than naive": cand_valid.lt(cand_valid["Seasonal naive"], axis=0).sum(),
}).sort_values("median WAPE, %").round(1)
selection.loc["Seasonal naive", "windows better than naive"] = np.nan
FINAL = selection.index[0]
print("Final model chosen on validation:", FINAL)
selection.to_csv(FIG_DIR.parent / "validation_selection.csv")
selection

# %% [markdown]
# ### 8.3 Where the models fail
# The chart shows the validation WAPE of each window. The bad windows are the same for every
# model: **December 2017** (right after Black Friday, then the holiday season), **May 2018**
# (the truckers' strike started inside the window) and **late June 2018** (World Cup). Nothing
# trained on history can foresee a strike; in practice this is handled by monitoring the error
# and by planners adding known events as inputs.

# %%
show = ["Seasonal naive", "SARIMA", choice["Prophet"], choice["LightGBM"], FINAL]
colors = [C["naive"], C["SARIMA"], C["Prophet"], C["LightGBM"], C["Ensemble"]]
wape_all = window_table(show)
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(wape_all)); w = 0.16
for i, (name, col) in enumerate(zip(show, colors)):
    ax.bar(x + (i - 2) * w, wape_all[name].clip(upper=60), width=w - 0.02, color=col, label=name)
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f0efec", lw=0, zorder=0)
ax.annotate("test", (x[-1], 57), ha="center", fontsize=9, color=C["muted"])
ax.set_xticks(x, [d.strftime("%d %b %y") for d in wape_all.index], fontsize=9)
ax.set(title="WAPE by 28-day window: 8 validation windows, then the test (capped at 60%)",
       ylabel="WAPE, %", xlabel="window start")
ax.grid(axis="x", visible=False); ax.legend(ncol=3, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(FIG_DIR / "08_backtest.png", dpi=150); plt.show()

# %% [markdown]
# ## 9. The test month, evaluated once

# %%
test_models = ["Seasonal naive", "SARIMA", "Prophet, no events", "Prophet + events",
               "LightGBM, no events", "LightGBM + events"] + [c for c in candidates if c.startswith("Avg(")]
results = pd.DataFrame({n: metrics(test, point(TEST_END, n), insample[TEST_END]) for n in test_models}).T
results.insert(0, "chosen on validation", ["final model" if n == FINAL else ("kept" if n in candidates else "")
                                          for n in results.index])
results = results.sort_values("WAPE, %").round(2)
results.index.name = "Model"
results.to_csv(FIG_DIR.parent / "results.csv")
results

# %% [markdown]
# ### 9.1 Forecast vs actual

# %%
final_fc = pd.Series(point(TEST_END, FINAL), index=test.index)
naive_fc = pd.Series(point(TEST_END, "Seasonal naive"), index=test.index)
sar = runs[TEST_END]["SARIMA"]
# Three lines only (actual, baseline, final model); every other model is in the table above.
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(y["2018-05-15":train.index.max()], color=C["muted"], lw=1, label="history")
ax.axvspan(test.index.min(), test.index.max(), color="#f0efec", lw=0, zorder=0)
ax.fill_between(test.index, sar[1], sar[2], color=C["SARIMA"], alpha=0.12, lw=0, label="80% interval (SARIMA)")
ax.plot(naive_fc, color=C["naive"], ls="--", lw=1.6,
        label=f"baseline: same weekday last week (WAPE {results.loc['Seasonal naive', 'WAPE, %']:.1f}%)")
ax.plot(final_fc, color=C["Ensemble"], lw=2.6,
        label=f"final model: SARIMA + Prophet average (WAPE {results.loc[FINAL, 'WAPE, %']:.1f}%)")
ax.plot(test, color=C["actual"], lw=2.2, label="actual orders")
for text_, day in [("truckers' strike", "2018-05-26"), ("World Cup", "2018-06-29"), ("test month", "2018-08-07")]:
    ax.annotate(text_, (pd.Timestamp(day), 60), fontsize=9, color=C["muted"], ha="center")
ax.set_ylim(40, None)
ax.set(title="28-day forecast vs actual daily orders (test month, never used for choosing)", ylabel="orders per day")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "07_forecast_vs_actual.png", dpi=150); plt.show()

# %% [markdown]
# ### 9.2 Are the forecast intervals honest?
# For planning, the interval matters more than the point forecast: staffing should cover the
# upper bound. An **80% interval should contain about 80% of the actual days.** We check this
# over all 8 validation windows (224 days) and on the test month.

# %%
def coverage(name, ends):
    inside = [((actuals[e].values >= runs[e][name][1]) & (actuals[e].values <= runs[e][name][2])) for e in ends]
    return np.concatenate(inside).mean() * 100

def width(name, ends):
    return np.mean([np.mean((runs[e][name][2] - runs[e][name][1]) / runs[e][name][0]) for e in ends]) * 100

coverage_table = pd.DataFrame({
    n: {"coverage, validation (224 days), %": coverage(n, VALID_ENDS), "coverage, test (28 days), %": coverage(n, [TEST_END]),
        "average width, % of forecast": width(n, VALID_ENDS)}
    for n in ["SARIMA", choice["Prophet"]]}).T.round(1)
coverage_table.to_csv(FIG_DIR.parent / "interval_coverage.csv")
coverage_table

# %% [markdown]
# ### 9.4 Sensitivity: what if scheduled events were treated as surprises?
# The first version of this notebook used a stricter rule: every event, including the World Cup,
# was only known up to the forecast date. Re-running the selection with that rule shows how much
# the final answer depends on it. Only the two "+ events" models change; everything else is reused.

# %%
strict_runs = {e: {**runs[e], **forecast_window(e, strict=True, only_event_models=True)} for e in WINDOW_ENDS}
saved_runs, runs = runs, strict_runs
strict_choice = {}
strict_valid = window_table(variants, ends=VALID_ENDS)
for model in ("Prophet", "LightGBM"):
    a, b = f"{model}, no events", f"{model} + events"
    strict_choice[model] = b if strict_valid[b].median() < strict_valid[a].median() else a
strict_base = ["SARIMA", strict_choice["Prophet"], strict_choice["LightGBM"]]
strict_cands = ["Seasonal naive", *strict_base] + [f"Avg({' | '.join(c)})" for k in (2, 3)
                                                   for c in __import__("itertools").combinations(strict_base, k)]
strict_final = window_table(strict_cands, ends=VALID_ENDS).median().idxmin()
sensitivity = pd.DataFrame({
    "rule for scheduled events": ["known in advance (main)", "treated as surprises (strict)"],
    "model chosen on validation": [FINAL, strict_final],
    "test WAPE, %": [results.loc[FINAL, "WAPE, %"], metrics(test, point(TEST_END, strict_final))["WAPE, %"]],
    "seasonal naive test WAPE, %": [results.loc["Seasonal naive", "WAPE, %"]] * 2,
}).round(1)
runs = saved_runs
sensitivity.to_csv(FIG_DIR.parent / "sensitivity_event_rule.csv", index=False)
sensitivity

# %% [markdown]
# ### 9.3 What LightGBM and Prophet learned

# %%
_, lgbm_model, lgbm_cols = fc_lgbm(train, known_events(train.index[-1]) if "events" in choice["LightGBM"] and "no" not in choice["LightGBM"] else None,
                                   return_model=True)
imp = pd.Series(lgbm_model.booster_.feature_importance("gain"), index=lgbm_cols).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(imp.index, imp / imp.sum(), color=C["LightGBM"], height=0.6)
ax.set(title="LightGBM feature importance (share of gain)", xlabel="share of total gain")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG_DIR / "06_lgbm_importance.png", dpi=150); plt.show()

ev = known_events(train.index[-1])
m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, seasonality_mode="multiplicative",
            holidays=pd.concat([pd.DataFrame({"holiday": n, "ds": pd.date_range(a, b)}) for n, (a, b) in ev.items()]))
m.add_country_holidays(country_name="BR")
m.fit(train.rename_axis("ds").reset_index(name="y"))
fig = m.plot_components(m.predict(m.make_future_dataframe(periods=HORIZON)), figsize=(10, 6))
fig.savefig(FIG_DIR / "05_prophet_components.png", dpi=150); plt.show()

# %% [markdown]
# ## 10. Many series: forecasting by state
#
# A real planning team does not need one national number; it needs the volume **per region**
# (warehouses and couriers are regional). This also shows where gradient boosting earns its
# place: one **global** LightGBM model learns from all series at once, while SARIMA and Prophet
# fit each series alone.
#
# * **Series:** orders per day for the 5 largest customer states (SP, RJ, MG, RS, PR) and
#   "Other" for the rest. The six series add up exactly to the national total.
# * **Global LightGBM:** one model, rows = (state, day). Features are the same as above plus the
#   state as a category. Lags and the target are divided by the state's own level (the 28-day
#   mean ending 28 days earlier), so a small and a large state share one pattern.
# * **Same protocol:** 8 validation windows, then the test month once, all with refits.
# * **Bottom-up:** summing the six state forecasts gives a national forecast that is consistent
#   with the regional plans by construction.

# %%
CUST_CANDIDATES = [Path("/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv"),
                   Path("../data/olist_customers_dataset.csv"), Path("data/olist_customers_dataset.csv")]
cpath = next((p for p in CUST_CANDIDATES if p.exists()), None)
customers = pd.read_csv(cpath or SOURCE_URL.replace("olist_orders_dataset", "olist_customers_dataset"),
                        usecols=["customer_id", "customer_state"])
with_state = placed.merge(customers, on="customer_id", how="left")
TOP_STATES = with_state["customer_state"].value_counts().index[:5].tolist()
with_state["region"] = with_state["customer_state"].where(with_state["customer_state"].isin(TOP_STATES), "Other")
panel = (with_state.groupby([pd.Grouper(key="order_purchase_timestamp", freq="D"), "region"]).size()
         .unstack(fill_value=0).asfreq("D", fill_value=0).reindex(y.index, fill_value=0)[TOP_STATES + ["Other"]]
         .astype(float))
assert np.allclose(panel.sum(axis=1), y), "states must add up to the national total"
print("Share of orders:", (panel.sum() / panel.sum().sum() * 100).round(1).to_dict())
print("Mean orders/day in the test month:", panel.loc[test.index].mean().round(0).to_dict())

# %%
use_events_lgbm = choice["LightGBM"] == "LightGBM + events"
use_events_prophet = choice["Prophet"] == "Prophet + events"


def panel_frame(events) -> pd.DataFrame:
    flags = event_flags(y.index, events) if events is not None else None
    parts = []
    for s in panel.columns:
        v = panel[s]
        level = v.shift(HORIZON).rolling(28).mean().clip(lower=1)
        d = pd.DataFrame({"orders": v, "series": s, "level": level, "target": v / level,
                          "dayofweek": y.index.dayofweek, "month": y.index.month, "day": y.index.day,
                          "is_holiday": y.index.isin(holiday_days).astype(int)}, index=y.index)
        for lag in (28, 35, 42):
            d[f"lag_{lag}"] = v.shift(lag) / level
        d["rolling_mean_7"] = v.shift(HORIZON).rolling(7).mean() / level
        if flags is not None:
            d["is_event"], d["lag_28_is_event"] = flags, flags.shift(28)
        parts.append(d)
    out = pd.concat(parts)
    out["series"] = out["series"].astype("category")
    return out


PANEL_FEATURES = ["series", "dayofweek", "month", "day", "is_holiday", "lag_28", "lag_35", "lag_42", "rolling_mean_7"]


def panel_window(end: int) -> dict:
    tr_idx, fc_idx = y.index[:end - HORIZON], y.index[end - HORIZON:end]
    ev = known_events(tr_idx[-1])
    out = {"Seasonal naive": {}, "SARIMA": {}, "Prophet": {}}
    for s in panel.columns:
        tr = panel[s].loc[tr_idx]
        out["Seasonal naive"][s] = fc_naive(tr)[0]
        out["SARIMA"][s] = fc_sarima(tr, log=np.log1p, exp=np.expm1)[0]
        out["Prophet"][s] = np.clip(fc_prophet(tr, ev if use_events_prophet else {})[0], 0, None)
    frame = panel_frame(ev if use_events_lgbm else None)
    cols = PANEL_FEATURES + (EVENT_FEATURES if use_events_lgbm else [])
    f_tr = frame[frame.index.isin(tr_idx)].dropna(subset=cols)
    f_fc = frame[frame.index.isin(fc_idx)]
    model = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": 500, "num_leaves": 31, "min_child_samples": 20})
    model.fit(f_tr[cols], f_tr["target"])
    pred = model.predict(f_fc[cols]) * f_fc["level"].values
    out["Global LightGBM"] = {s: np.clip(pred[(f_fc["series"] == s).values], 0, None) for s in panel.columns}
    out["Avg(SARIMA | Global LightGBM)"] = {s: (out["SARIMA"][s] + out["Global LightGBM"][s]) / 2 for s in panel.columns}
    return out


panel_runs = {e: panel_window(e) for e in WINDOW_ENDS}

# %%
def panel_scores(end, name):
    act = panel.iloc[end - HORIZON:end]
    pred = pd.DataFrame(panel_runs[end][name], index=act.index)[act.columns]
    return {"state-level WAPE, %": (act - pred).abs().values.sum() / act.values.sum() * 100,
            "bottom-up national WAPE, %": metrics(act.sum(axis=1), pred.sum(axis=1))["WAPE, %"]}

panel_models = ["Seasonal naive", "SARIMA", "Prophet", "Global LightGBM", "Avg(SARIMA | Global LightGBM)"]
panel_valid = pd.DataFrame({n: pd.DataFrame([panel_scores(e, n) for e in VALID_ENDS]).median() for n in panel_models}).T
PANEL_FINAL = panel_valid["state-level WAPE, %"].idxmin()
panel_summary = pd.concat({"validation (median of 8)": panel_valid,
                           "test": pd.DataFrame({n: panel_scores(TEST_END, n) for n in panel_models}).T}, axis=1).round(1)
panel_summary.to_csv(FIG_DIR.parent / "state_results.csv")
print("Chosen on validation for the state level:", PANEL_FINAL)
panel_summary

# %%
act = panel.loc[test.index]
pred = pd.DataFrame(panel_runs[TEST_END][PANEL_FINAL], index=test.index)[act.columns]
per_state = pd.DataFrame({
    "mean orders/day": act.mean(),
    f"WAPE {PANEL_FINAL}, %": (act - pred).abs().sum() / act.sum() * 100,
    "WAPE seasonal naive, %": (act - pd.DataFrame(panel_runs[TEST_END]["Seasonal naive"], index=test.index)).abs().sum() / act.sum() * 100,
}).round(1)
per_state

# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True)
for ax, s in zip(axes.flat, panel.columns):
    ax.plot(panel[s]["2018-06-01":train.index.max()], color=C["muted"], lw=0.9)
    ax.axvspan(test.index.min(), test.index.max(), color="#f0efec", lw=0, zorder=0)
    ax.plot(act[s], color=C["actual"], lw=1.8)
    ax.plot(pred[s], color=C["Ensemble"], lw=1.8)
    ax.set_title(f"{s}: WAPE {per_state.loc[s, f'WAPE {PANEL_FINAL}, %']:.0f}%", loc="left", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
fig.suptitle(f"Test month by state: actual (black) vs {PANEL_FINAL} (purple)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "09_states.png", dpi=150); plt.show()

# %% [markdown]
# ## 11. Conclusions
#
# * **Final model, chosen on validation only: the average of SARIMA and Prophet (with the special
#   events).** On the untouched test month it reaches **WAPE 10.6% (MAPE 10.2%)** against **14.7%**
#   for the seasonal-naive baseline, with MASE 0.79. On the 8 validation windows it has the lowest
#   median WAPE (13.4% vs 20.1% naive) and beats the baseline in 6 of 8.
# * **The result depends on one modelling assumption, and it is stated openly.** If scheduled events
#   (the World Cup) are treated as unknown until they start, validation picks a different model and
#   the test WAPE is 16.1%, worse than naive (section 9.4). Knowing the football calendar is
#   realistic for a planner; knowing about the strike in advance is not, and the model never does.
# * **Seasonal naive is a strong baseline.** The weekly cycle explains most of the variation, so a
#   model has to earn its complexity. LightGBM on one series does not: with lags ≥ 28 days its
#   features fall straight into the World Cup dip.
# * **Intervals: trust SARIMA, not Prophet.** SARIMA's 80% interval covers 82% of validation days;
#   Prophet's covers only 66%, so it is overconfident and would under-staff on busy days.
# * **By state, the picture is mixed and reported as it is.** The average of per-state SARIMA and a
#   global LightGBM was chosen on validation (state-level WAPE 20.8% vs 28.2% naive). On the test
#   month it is still better than naive per state (20.1% vs 23.2%), and clearly so for the small
#   states RS and PR, but per-state Prophet happened to do better on that one month (17.7%).
#
# **Limitations and next steps**
# * 8 validation windows are still a small sample, and three of them are dominated by shocks.
# * The event list was written by hand from the EDA; a production system would take it from a
#   maintained calendar and add promotions and marketing spend as regressors.
# * Prophet's interval could be recalibrated on the validation windows (conformal prediction).
# * Category-level forecasts with reconciliation (e.g. MinT) would be the next step for planning.

# %% [markdown]
# ## Interview notes
#
# * **Why not a random split?** Neighbouring days are correlated, and a random split leaks the
#   future into training, so the model interpolates instead of forecasting.
# * **Why choose on validation windows and not on the test month?** Every choice made by looking at
#   the test (which features, which ensemble) makes the test score optimistic. Here the test month
#   was used once, after all decisions were fixed.
# * **Why lags ≥ 28 in LightGBM?** The horizon is 28 days. For the last forecast day the newest known
#   value is 28 days old. A shorter lag would use values that do not exist at forecast time.
# * **WAPE vs MAPE?** MAPE averages daily percentage errors and explodes on small actuals; WAPE
#   divides the total error by total volume, so it reflects the business cost. MASE compares with
#   the naive forecast and works across series of any size.
# * **Why log for SARIMA?** The weekly swing is proportional to the level (multiplicative). The log
#   makes it additive and stabilises the variance.
