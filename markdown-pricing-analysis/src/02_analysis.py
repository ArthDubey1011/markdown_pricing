"""Step 2: statistical analysis on the tables built by 01_build_tables.py.

Outputs
  outputs/results.json            every number quoted in the report and README
  outputs/tables/*.csv            round summary, models, tiers, Tukey, counterfactual, policy

Run from the repository root:  python src/02_analysis.py
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[1]
OUT, TAB = ROOT / "outputs", ROOT / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
SEED, COGS = 42, 0.40

con = duckdb.connect(str(ROOT / "markdown.duckdb"), read_only=True)
e = con.execute("SELECT * FROM events ORDER BY Product_ID, stage").df()
p = con.execute("SELECT * FROM products ORDER BY Product_ID").df()
R = {"n_products": len(p), "n_events": len(e)}

# ---------------------------------------------------------------- 1. lift by round
rounds = (e.groupby("stage")
          .agg(mean_depth=("markdown", "mean"), incr_lift_mean=("step_lift", "mean"),
               incr_lift_median=("step_lift", "median"), cum_lift=("cum_lift", "mean"),
               breakeven=("breakeven_lift", "mean")) * 100).round(1).reset_index()
rounds.to_csv(TAB / "lift_by_round.csv", index=False)
R["rounds"] = rounds.to_dict("records")
R["lift_mean"], R["lift_median"] = round(e.step_lift.mean() * 100, 1), round(e.step_lift.median() * 100, 1)
R["lift_skew"] = round(float(stats.skew(e.step_lift)), 2)
R["breakeven_share_pct"] = round(float((e.step_lift >= e.breakeven_lift).mean()) * 100, 1)

# ---------------------------------------------------------------- 2. response models
# Lift is proportional to depth, so models have no intercept: lift = depth x slope.
e["segment"] = e.Season + "-" + e.Category
ctrl = " + markdown:C(Promotion_Type) + markdown:C(Brand) + markdown:price_gap_vs_competitor + C(stage)"
cluster = dict(cov_type="cluster", cov_kwds={"groups": e.Product_ID})

def r2(model):
    resid = e.step_lift - model.fittedvalues
    return float(1 - (resid ** 2).sum() / ((e.step_lift - e.step_lift.mean()) ** 2).sum())

m1 = smf.ols("step_lift ~ markdown", data=e).fit()
m2 = smf.ols("step_lift ~ markdown + markdown:Seasonality_Factor" + ctrl, data=e).fit(**cluster)
m3 = smf.ols("step_lift ~ 0 + markdown:C(segment)" + ctrl, data=e).fit(**cluster)
m4 = smf.ols("step_lift ~ 0 + markdown:C(segment) + markdown:Seasonality_Factor" + ctrl, data=e).fit(**cluster)

slopes = m3.params.filter(like="segment")
slopes.index = [i.split("[")[1].rstrip("]") for i in slopes.index]
in_season = sorted(slopes[slopes > 0.5].index)
off_season = sorted(slopes[slopes <= 0.5].index)
R["models"] = {
    "r2": {"M1_depth": round(r2(m1), 3), "M2_x_seasonality": round(r2(m2), 3),
           "M3_x_segment": round(r2(m3), 3), "M4_segment_plus_seasonality": round(r2(m4), 3)},
    "seasonality_coef_M2": round(float(m2.params["markdown:Seasonality_Factor"]), 3),
    "seasonality_coef_M4": round(float(m4.params["markdown:Seasonality_Factor"]), 4),
    "seasonality_p_M4": round(float(m4.pvalues["markdown:Seasonality_Factor"]), 3),
    "channel_p": {k.split("T.")[1].rstrip("]"): round(float(v), 3) for k, v in m3.pvalues.items() if "Promotion_Type" in k},
    "price_gap_p": round(float(m3.pvalues["markdown:price_gap_vs_competitor"]), 3),
    "round_p_min": round(float(m3.pvalues.filter(like="stage").min()), 3),
    "slopes": slopes.round(3).to_dict()}
R["in_season_segments"], R["off_season_segments"] = in_season, off_season
R["slope_in"], R["slope_off"] = round(float(slopes[in_season].mean()), 3), round(float(slopes[off_season].mean()), 3)
R["slope_ratio"] = round(R["slope_in"] / R["slope_off"], 1)
pd.DataFrame({"segment": slopes.index, "slope": slopes.values,
              "season_fit": np.where(slopes.values > 0.5, "in season", "off season")}).to_csv(TAB / "segment_slopes.csv", index=False)

# ---------------------------------------------------------------- 3. elasticity
e["elasticity"] = -e.step_lift / e.markdown
e["in_season"] = e.segment.isin(in_season)
R["elasticity"] = {
    "overall": round(e.elasticity.median(), 2),
    "in_season": round(e.loc[e.in_season, "elasticity"].median(), 2),
    "off_season": round(e.loc[~e.in_season, "elasticity"].median(), 2),
    "by_season": e.groupby("Season").elasticity.median().round(2).to_dict(),
    "by_category": e.groupby("Category").elasticity.median().round(2).to_dict(),
    "by_channel": e.groupby("Promotion_Type").elasticity.median().round(2).to_dict(),
    "by_segment": e.groupby("segment").elasticity.median().round(2).to_dict()}
R["share_products_in_season_pct"] = round(float(e.in_season.mean()) * 100, 1)
R["breakeven_share_by_fit_pct"] = {"in_season": round(float((e[e.in_season].step_lift >= e[e.in_season].breakeven_lift).mean()) * 100, 2),
                                   "off_season": round(float((e[~e.in_season].step_lift >= e[~e.in_season].breakeven_lift).mean()) * 100, 2)}

# ---------------------------------------------------------------- 4. tiers: ANOVA, Kruskal-Wallis, Tukey
# One randomly chosen round per product keeps the groups independent.
rng = np.random.default_rng(SEED)
pick = pd.DataFrame({"Product_ID": p.Product_ID.values, "stage": rng.integers(1, 5, len(p))})
one = e.merge(pick, on=["Product_ID", "stage"])
one["rev_change"] = (1 - one.markdown) * (1 + one.step_lift) - 1
groups = [g.step_lift.values for _, g in one.groupby("discount_band")]
F, pF = stats.f_oneway(*groups)
H, pH = stats.kruskal(*groups)
grand = one.step_lift.mean()
eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((one.step_lift - grand) ** 2).sum()
tk = pairwise_tukeyhsd(one.step_lift * 100, one.discount_band)
tk_rev = pairwise_tukeyhsd(one.rev_change * 100, one.discount_band)
tukey = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
tukey.to_csv(TAB / "tukey_hsd.csv", index=False)
tiers = one.groupby("discount_band").agg(n=("step_lift", "size"), unit_lift=("step_lift", "mean"),
                                         ci95=("step_lift", lambda x: 1.96 * x.std() / np.sqrt(len(x))),
                                         revenue_change=("rev_change", "mean")).reset_index()
tiers[["unit_lift", "ci95", "revenue_change"]] *= 100
tiers.round(2).to_csv(TAB / "tiers.csv", index=False)
R["anova"] = {"F": round(float(F), 1), "p": float(pF), "eta_squared": round(float(eta2), 3),
              "kruskal_H": round(float(H), 1), "kruskal_p": float(pH),
              "tukey_all_significant_lift": bool(all(tk.reject)), "tukey_all_significant_revenue": bool(all(tk_rev.reject)),
              "tiers": tiers.round(2).to_dict("records"),
              "tukey": tukey.assign(meandiff=tukey.meandiff.astype(float).round(2), lower=tukey.lower.astype(float).round(2),
                                    upper=tukey.upper.astype(float).round(2)).to_dict("records")}

# ---------------------------------------------------------------- 5. counterfactual revenue & margin
S = p[[f"Sales_After_M{i}" for i in range(1, 5)]].values
M = p[[f"Markdown_{i}" for i in range(1, 5)]].values
P, S0 = p.Original_Price.values[:, None], p.Historical_Sales.values[:, None]
p["in_season"] = (p.Season + "-" + p.Category).isin(in_season)

def cf(mask, cogs):
    rev_act, rev_cf = (P * (1 - M) * S)[mask].sum(), (4 * P * S0)[mask].sum()
    m_act, m_cf = (P * (1 - M - cogs) * S)[mask].sum(1), (4 * P * (1 - cogs) * S0)[mask].sum(1)
    return {"revenue_pct": round((rev_act / rev_cf - 1) * 100, 1), "margin_pct": round((m_act.sum() / m_cf.sum() - 1) * 100, 1),
            "products_losing_margin_pct": round(float((m_act < m_cf).mean()) * 100, 1)}

allp = np.ones(len(p), bool)
R["counterfactual"] = {f"cogs_{int(c*100)}": cf(allp, c) for c in (0.3, 0.4, 0.5)}
R["counterfactual"]["in_season"] = cf(p.in_season.values, COGS)
R["counterfactual"]["off_season"] = cf(~p.in_season.values, COGS)
vol = (P * S).sum() - (4 * P * S0).sum(); disc = -(P * M * S).sum(); base = (4 * P * S0).sum()
R["waterfall"] = {"revenue": [100, round(vol / base * 100, 1), round(disc / base * 100, 1)],
                  "margin": [100, round(vol / base * 100, 1), round(disc / ((1 - COGS) * base) * 100, 1)]}
pd.DataFrame(R["counterfactual"]).T.to_csv(TAB / "counterfactual.csv")

# ---------------------------------------------------------------- 6. policy simulation
grid = np.round(np.arange(0.10, 0.51, 0.05), 2)
rows, curves = [], {}
for seg, k in slopes.items():
    growth = lambda d: np.cumprod(np.repeat(1 + k * d, 4))
    rev = [float(((1 - d) * growth(d)).sum() / 4 - 1) * 100 for d in grid]
    mar = [float(((1 - d - COGS) * growth(d)).sum() / (4 * (1 - COGS)) - 1) * 100 for d in grid]
    curves[seg] = {"revenue": [round(x, 2) for x in rev], "margin": [round(x, 2) for x in mar]}
    rows.append({"segment": seg, "slope": round(float(k), 3),
                 "margin_best_depth": float(grid[np.argmax(mar)]), "margin_at_best_pct": round(max(mar), 1),
                 "revenue_best_depth": float(grid[np.argmax(rev)]), "revenue_at_best_pct": round(max(rev), 1),
                 "margin_at_revenue_best_pct": round(mar[int(np.argmax(rev))], 1)})
pd.DataFrame(rows).to_csv(TAB / "policy_simulation.csv", index=False)
R["policy"] = {"grid": grid.tolist(), "curves": curves, "by_segment": rows}
R["actual_avg_depth_pct"] = round(float(M.mean()) * 100, 1)

(OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
cf40 = R["counterfactual"]["cogs_40"]
print(f"""Results written to outputs/results.json and outputs/tables/
  median elasticity          {R['elasticity']['overall']}  (in season {R['elasticity']['in_season']}, off season {R['elasticity']['off_season']})
  response slope ratio       {R['slope_ratio']}x in vs off season
  R2  M1 / M2 / M3 / M4      {' / '.join(str(v) for v in R['models']['r2'].values())}
  ANOVA                      F = {R['anova']['F']:,}, eta2 = {R['anova']['eta_squared']}, Tukey all pairs significant: {R['anova']['tukey_all_significant_lift']}
  vs no markdown (COGS 40%)  revenue {cf40['revenue_pct']:+}%, margin {cf40['margin_pct']:+}%""")
