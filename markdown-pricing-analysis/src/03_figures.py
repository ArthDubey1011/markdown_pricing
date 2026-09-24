"""Step 3: every figure used in the README and the LaTeX report.

Writes PNG (for the README) and PDF (vector, for LaTeX) to outputs/figures/.
Run from the repository root:  python src/03_figures.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[1]
F = str(ROOT / "outputs" / "figures") + "/"
Path(F).mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "outputs" / "results.json"))
con = duckdb.connect(str(ROOT / "markdown.duckdb"), read_only=True)
e = con.execute("SELECT * FROM events ORDER BY Product_ID, stage").df()
p = con.execute("SELECT * FROM products ORDER BY Product_ID").df()
rng = np.random.default_rng(42)
pick = pd.DataFrame({"Product_ID": p.Product_ID.values, "stage": rng.integers(1, 5, len(p))})
one = e.merge(pick, on=["Product_ID", "stage"])
e["segment"] = e.Season + "-" + e.Category

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "axes.grid": True, "grid.alpha": .3, "grid.linewidth": .5, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.6, "legend.frameon": False,
    "pdf.fonttype": 42})
NAVY, RED, TEAL, GOLD, GREY = "#1f3a5f", "#c0392b", "#2a7f78", "#c9a227", "#9aa5b1"
FULL, HALF = 6.3, 3.05
def save(fig, name): fig.savefig(F + name + ".pdf"); fig.savefig(F + name + ".png", dpi=300); plt.close(fig)
def pct(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}\\%".replace("\\", "") if False else f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

# F1 data overview
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.5))
data = [p[f"Markdown_{i}"] * 100 for i in range(1, 5)]
bp = axs[0].boxplot(data, widths=.5, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.4))
for b in bp["boxes"]: b.set(facecolor=NAVY, edgecolor=NAVY)
axs[0].set_xticks([1, 2, 3, 4], ["Round 1", "Round 2", "Round 3", "Round 4"]); axs[0].set_ylabel("Discount depth"); pct(axs[0])
axs[0].set_title("(a) Discount depth by markdown round", loc="left")
axs[1].hist(e.step_lift * 100, bins=60, color=TEAL, alpha=.85)
axs[1].axvline(e.step_lift.mean() * 100, color=RED, ls="--", lw=1.1, label=f"mean {e.step_lift.mean()*100:.1f}%")
axs[1].axvline(e.step_lift.median() * 100, color=NAVY, ls=":", lw=1.3, label=f"median {e.step_lift.median()*100:.1f}%")
axs[1].set_xlabel("Incremental sales lift per markdown event"); axs[1].set_ylabel("Events"); pct(axs[1], "x")
axs[1].legend(); axs[1].set_title("(b) Distribution of incremental lift ($n$ = 173,600)", loc="left")
fig.tight_layout(); save(fig, "fig01_data_overview")

# F2 attribution
sv = pd.DataFrame(R["rounds"]).rename(columns={"incr_lift_mean": "incremental_lift_pct", "cum_lift": "naive_cumulative_lift_pct", "mean_depth": "avg_discount_pct"}); x = np.arange(4)
fig, ax = plt.subplots(figsize=(FULL, 2.6))
b1 = ax.bar(x - .19, sv.naive_cumulative_lift_pct, .36, color=GREY, label="Cumulative: vs. pre-markdown baseline")
b2 = ax.bar(x + .19, sv.incremental_lift_pct, .36, color=RED, label="Incremental: vs. immediately preceding period")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=8)
ax.set_xticks(x, [f"Round {s}\n(mean depth {d:.0f}%)" for s, d in zip(sv.stage, sv.avg_discount_pct)])
ax.set_ylabel("Sales lift"); pct(ax); ax.set_ylim(0, 112); ax.legend(loc="upper left")
save(fig, "fig02_attribution")

# F3 lift vs depth by regime
e["regime"] = np.where(e.segment.isin(R["in_season_segments"]), "In season", "Off season")
e["db"] = (e.markdown * 20).round() / 20
g = e.groupby(["regime", "db"]).step_lift.agg(["mean", "sem"]).reset_index()
fig, ax = plt.subplots(figsize=(FULL, 2.6))
for (lab, d), c, m in zip(g.groupby("regime"), [NAVY, GREY], ["o", "s"]):
    ax.errorbar(d.db * 100, d["mean"] * 100, yerr=1.96 * d["sem"] * 100, color=c, marker=m, ms=4, capsize=2, label=f"{lab} (slope {R['slope_in'] if lab=='In season' else R['slope_off']:.2f})")
dd = np.linspace(.1, .5, 60); ax.plot(dd * 100, dd / (1 - dd) * 100, "--", color=RED, lw=1.2, label="Break-even lift $d/(1-d)$")
ax.set_ylim(0, 60); ax.set_xlabel("Discount depth"); ax.set_ylabel("Incremental sales lift"); pct(ax); pct(ax, "x"); ax.legend(loc="upper left")
save(fig, "fig03_lift_by_regime")

# F4 elasticity heatmap
el = e.assign(el=-e.step_lift / e.markdown).groupby(["Season", "Category"]).el.median().unstack().loc[["Spring", "Summer", "Rainy", "Winter"], ["Makeup", "Bodycare", "Skincare"]]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
axs[0].grid(False); axs[0].imshow(el.values, cmap="Blues_r", vmin=-1, vmax=0, aspect="auto")
axs[0].set_xticks(range(3), el.columns); axs[0].set_yticks(range(4), el.index)
for i in range(4):
    for j in range(3):
        v = el.values[i, j]; axs[0].text(j, i, f"${v:.2f}$", ha="center", va="center", color="white" if v < -.5 else "k", fontsize=9)
for sp in axs[0].spines.values(): sp.set_visible(False)
axs[0].set_title("(a) Median elasticity by season and category", loc="left")
d = pd.Series(R["elasticity"]["by_season"]).loc[["Spring", "Summer", "Rainy", "Winter"]]
d2 = pd.Series(R["elasticity"]["by_category"])
lab = [f"Season: {k}" for k in d.index] + [f"Category: {k}" for k in d2.index] + ["All events"]
val = list(d.values) + list(d2.values) + [R["elasticity"]["overall"]]
axs[1].barh(range(len(val))[::-1], val, color=[GREY] * 4 + [TEAL] * 3 + [RED], height=.6)
axs[1].set_yticks(range(len(val))[::-1], lab); axs[1].axvline(-1, color=RED, ls="--", lw=.9); axs[1].set_xlim(-1.05, 0)
axs[1].set_title("(b) Marginal medians", loc="left"); axs[1].set_xlabel("Median price elasticity")
fig.tight_layout(); save(fig, "fig04_elasticity")

# F5 model comparison + segment slopes
MD = R["models"]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
names = ["M1", "M2", "M3", "M4"]
vals = list(MD["r2"].values())
axs[0].bar(range(4), vals, color=[GREY, GOLD, NAVY, NAVY], width=.6)
for i, v in enumerate(vals): axs[0].text(i, v + .02, f"{v:.2f}", ha="center", fontsize=8.5)
axs[0].set_xticks(range(4), names, fontsize=8.5); axs[0].set_ylim(0, 1); axs[0].set_ylabel("$R^2$"); axs[0].set_title("(a) Explained variance", loc="left")
kk = pd.Series(MD["slopes"]).sort_values()
axs[1].barh(range(len(kk)), kk.values, color=[NAVY if v > .5 else GREY for v in kk.values], height=.6)
axs[1].set_yticks(range(len(kk)), [x.replace("-", " / ") for x in kk.index], fontsize=7.5)
axs[1].set_xlabel("Estimated slope $k$ (lift per unit of depth)"); axs[1].set_title("(b) Discount response by segment", loc="left")
fig.tight_layout(); save(fig, "fig05_models")

# F6 tiers
bd = pd.DataFrame(R["anova"]["tiers"]).assign(lift=lambda d: d.unit_lift / 100, lift_ci95=lambda d: d.ci95 / 100, rev_index=lambda d: 1 + d.revenue_change / 100); x = np.arange(4)
fig, ax = plt.subplots(figsize=(HALF, 2.6))
ax.bar(x, bd.lift * 100, .55, yerr=bd.lift_ci95 * 100, capsize=3, color=TEAL, label="Unit lift")
ax.plot(x, (bd.rev_index - 1) * 100, "o-", color=RED, label="Period revenue change")
for i, r in bd.iterrows():
    ax.text(i, r.lift * 100 + 1.6, f"+{r.lift*100:.1f}", ha="center", fontsize=7.5, color=TEAL)
    ax.text(i, (r.rev_index - 1) * 100 - 4.5, f"{(r.rev_index-1)*100:.1f}", ha="center", fontsize=7.5, color=RED)
ax.axhline(0, color="k", lw=.7); ax.set_xticks(x, bd.discount_band); ax.set_xlabel("Discount tier"); ax.set_ylim(-38, 34)
pct(ax); ax.legend(loc="lower left", fontsize=7.5)
save(fig, "fig06_tiers")

# F7 tukey
tk = pairwise_tukeyhsd(one.step_lift * 100, one.discount_band)
t = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0]).astype({"meandiff": float, "lower": float, "upper": float})
t["pair"] = t.group2 + " vs " + t.group1
fig, ax = plt.subplots(figsize=(HALF, 2.6)); yy = np.arange(len(t))
ax.hlines(yy, t.lower, t.upper, color=NAVY, lw=3.5); ax.plot(t.meandiff, yy, "o", color=RED, ms=4)
ax.axvline(0, color="k", ls="--", lw=.7); ax.set_yticks(yy, t.pair); ax.set_xlim(-1, 19)
ax.set_xlabel("Difference in mean lift (pp)")
save(fig, "fig07_tukey")

# F8 break-even
e["db2"] = (e.markdown * 50).round() / 50
q = e.groupby("db2").step_lift.agg(mean="mean", lo=lambda s: s.quantile(.05), hi=lambda s: s.quantile(.95)).reset_index()
dd = np.linspace(.1, .5, 100)
fig, ax = plt.subplots(figsize=(FULL, 2.6))
ax.fill_between(q.db2 * 100, q.lo * 100, q.hi * 100, color=TEAL, alpha=.2, label="Observed lift, 5th-95th percentile")
ax.plot(q.db2 * 100, q["mean"] * 100, color=TEAL, label="Observed lift, mean")
ax.plot(dd * 100, dd / (1 - dd) * 100, color=RED, ls="--", label="Break-even lift $d/(1-d)$")
ax.fill_between(dd * 100, np.interp(dd * 100, q.db2 * 100, q["mean"] * 100), dd / (1 - dd) * 100, color=RED, alpha=.06)
ax.set_xlabel("Discount depth $d$"); ax.set_ylabel("Sales lift"); pct(ax); pct(ax, "x"); ax.legend(loc="upper left")
save(fig, "fig08_breakeven")

# F9 waterfall
wf = R["waterfall"]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
for a, vals, ttl in [(axs[0], wf["revenue"], "(a) Revenue"), (axs[1], wf["margin"], "(b) Contribution margin (COGS = 40%)")]:
    c1, c2, end = vals[0], vals[0] + vals[1], sum(vals)
    a.bar(0, vals[0], color=GREY, width=.6); a.bar(1, vals[1], bottom=c1, color=TEAL, width=.6)
    a.bar(2, vals[2], bottom=c2, color=RED, width=.6); a.bar(3, end, color=NAVY, width=.6)
    for i, (y, s) in enumerate([(c1, f"{vals[0]:.0f}"), (c2, f"+{vals[1]:.1f}"), (c2, f"${vals[2]:.1f}$"), (end, f"{end:.1f}")]):
        a.text(i, y + 3, s, ha="center", fontsize=8.5)
    a.plot([0.3, 0.7], [c1, c1], color="k", lw=.5); a.plot([1.3, 1.7], [c2, c2], color="k", lw=.5)
    a.set_xticks(range(4), ["No markdown", "Volume\neffect", "Discount\ncost", "Actual"]); a.set_title(ttl, loc="left")
axs[0].set_ylabel("Index (no markdown = 100)"); axs[0].set_ylim(0, 178)
fig.tight_layout(); save(fig, "fig09_waterfall")

# F10 policy curves
grid = np.array(R["policy"]["grid"]) * 100; cv = R["policy"]["curves"]
avg = lambda segs, k: np.mean([cv[x][k] for x in segs], axis=0)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
for a, segs, t in [(axs[0], R["in_season_segments"], "(a) In-season segments"), (axs[1], R["off_season_segments"], "(b) Off-season segments")]:
    a.plot(grid, avg(segs, "revenue"), "o-", color=TEAL, ms=3, label="Revenue")
    a.plot(grid, avg(segs, "margin"), "s-", color=NAVY, ms=3, label="Contribution margin")
    a.axhline(0, color="k", lw=.7); a.set_title(t, loc="left"); a.set_xlabel("Uniform discount depth"); pct(a); pct(a, "x")
axs[0].set_ylabel("Change vs. no markdown"); axs[0].legend(loc="lower left")
ri = avg(R["in_season_segments"], "revenue"); mi = avg(R["in_season_segments"], "margin")
axs[0].annotate(f"revenue max: 45%\n(+{ri[7]:.0f}%)", xy=(45, ri[7]), xytext=(29, 45), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.6))
axs[0].annotate(f"margin max: 10%\n(+{mi[0]:.1f}%)", xy=(10, mi[0]), xytext=(14, -30), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.6))
axs[0].set_ylim(-85, 60)
fig.tight_layout(); save(fig, "fig10_policy")


def box(ax, x, y, w, h, fc, ec="none", r=0.08, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def arrow(ax, x0, y0, x1, y1, c="#333", lw=1.1, style="-|>", ms=10, conn="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, connectionstyle=conn))

# ---------------- Pipeline (2 x 3) ----------------
steps = [("Clean", "remove duplicates,\ndrop leaking column", "SQL"),
         ("Reshape", "unpivot rounds,\nLAG for lift", "SQL"),
         ("Elasticity", "regression by\nseason $\\times$ category", "Python"),
         ("Compare tiers", "ANOVA, Tukey HSD,\nKruskal-Wallis", "Python"),
         ("Counterfactual", "revenue and margin\nvs. no markdown", "Python"),
         ("Policy", "simulate depth\nby segment", "Python")]
W, H = 12.6, 4.55
fig, ax = plt.subplots(figsize=(6.3, 6.3 * H / W)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
bw, bh, gx = 3.45, 1.55, 0.8
pos = []
for i in range(6):
    r, c = divmod(i, 3); x = 0.15 + c * (bw + gx); y = 2.8 - r * 2.4; pos.append((x, y))
for i, ((t, s_, tool), (x, y)) in enumerate(zip(steps, pos)):
    col = RED if i == 5 else NAVY
    box(ax, x, y, bw, bh, col, r=0.14)
    ax.add_patch(plt.Circle((x + 0.34, y + bh - 0.34), 0.21, color=GOLD))
    ax.text(x + 0.34, y + bh - 0.35, str(i + 1), ha="center", va="center", fontsize=9, color=NAVY, weight="bold")
    ax.text(x + 0.7, y + bh - 0.36, t, ha="left", va="center", fontsize=9.8, color="white")
    ax.text(x + bw - 0.12, y - 0.2, tool, ha="right", va="center", fontsize=7.3, color="#777", style="italic")
    ax.text(x + 0.3, y + 0.55, s_, ha="left", va="center", fontsize=8.2, color="white", linespacing=1.4)
for i in [0, 1, 3, 4]:
    x, y = pos[i]; arrow(ax, x + bw + 0.08, y + bh / 2, x + bw + gx - 0.08, y + bh / 2, ms=11, lw=1.3)
x3, y3 = pos[2]; x4, y4 = pos[3]
ym = (y3 + y4 + bh) / 2
ax.plot([x3 + bw / 2, x3 + bw / 2, x4 + bw / 2], [y3 - 0.06, ym, ym], color="#333", lw=1.3, solid_joinstyle="round")
arrow(ax, x4 + bw / 2, ym, x4 + bw / 2, y4 + bh + 0.06, ms=11, lw=1.3)
save(fig, "fig00b_pipeline")

# ---------------- Reshape ----------------
W, H = 12.6, 5.2
fig, ax = plt.subplots(figsize=(6.3, 6.3 * H / W)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
box(ax, 0.05, 1.0, 3.45, 3.35, NAVY, r=0.12)
ax.text(1.775, 3.95, "Wide table", ha="center", fontsize=10, color="white", weight="bold")
ax.text(1.775, 3.58, "one row per product", ha="center", fontsize=7.8, color="#cfd8e3", style="italic")
fields = ["ID, category, season", "price, brand, channel", "baseline sales $S_0$",
          "discounts $d_1, \\ldots, d_4$", "sales $S_1, \\ldots, S_4$"]
for k, f in enumerate(fields):
    yy = 3.12 - k * 0.37
    ax.plot([0.3, 3.25], [yy + 0.19, yy + 0.19], color="#3d5a80", lw=.6)
    ax.text(0.35, yy, f, ha="left", va="center", fontsize=7.6, color="white")
ax.text(1.775, 1.17, f"{R['n_products']:,} rows", ha="center", fontsize=8, color=GOLD, weight="bold")
arrow(ax, 3.6, 2.7, 4.55, 2.7, ms=12, lw=1.4)
ax.text(4.07, 2.95, "unpivot", ha="center", fontsize=8, color="#333")
ax.text(4.07, 2.3, "+ LAG", ha="center", fontsize=8, color="#333")
labels = [("Baseline", "$S_0$", GREY), ("Round 1", "$d_1,\\ S_1$", TEAL), ("Round 2", "$d_2,\\ S_2$", TEAL),
          ("Round 3", "$d_3,\\ S_3$", TEAL), ("Round 4", "$d_4,\\ S_4$", TEAL)]
cw, cg, cy, ch = 1.38, 0.2, 2.15, 1.1
xs = []
for i, (t, s_, c) in enumerate(labels):
    x = 4.65 + i * (cw + cg); xs.append(x)
    box(ax, x, cy, cw, ch, c, r=0.1)
    ax.text(x + cw / 2, cy + ch - 0.32, t, ha="center", va="center", fontsize=8.6, color="white", weight="bold")
    ax.text(x + cw / 2, cy + 0.32, s_, ha="center", va="center", fontsize=9.5, color="white")
    if i: arrow(ax, x - cg + 0.02, cy + ch / 2, x - 0.02, cy + ch / 2, ms=8)
mid = xs[0] + (xs[-1] + cw - xs[0]) / 2
ax.text(mid, 3.62, "time  $\\longrightarrow$", ha="center", fontsize=8.5, color="#333")
ax.text(mid, 4.15, "173,600 events  (43,400 products $\\times$ 4 rounds)", ha="center", fontsize=8.5, color=NAVY, weight="bold")
for j in range(1, 5):
    x0 = xs[j - 1] + cw / 2; x1 = xs[j] + cw / 2
    arrow(ax, x1, cy - 0.05, x0, cy - 0.05, c=RED, ms=8, lw=1.1, conn="arc3,rad=-0.55")
    ax.text((x0 + x1) / 2, cy - 0.7, f"$L_{j}$", ha="center", fontsize=9, color=RED)
ax.text(W / 2, 0.55, r"Incremental lift:  $L_j = S_j / S_{j-1} - 1$   (each round vs. the period just before it)",
        ha="center", fontsize=8.5, color=RED)
ax.text(W / 2, 0.12, r"Not used:  cumulative lift $C_j = S_j / S_0 - 1$, which inherits the effect of earlier rounds",
        ha="center", fontsize=7.8, color="#666", style="italic")
save(fig, "fig00a_reshape")
print("Figures written to outputs/figures/")
