# Retail Markdown Pricing & Margin Impact Analysis

**Do discounts pay for themselves?** This project measures what four rounds of markdowns did for 43,400 personal-care and beauty products: how many extra units each discount actually sold, how sensitive demand was to price, and whether the programme made or lost money once product costs are counted.

The problem framing is adapted from Increff's case study on markdown optimization. The data is synthetic, so the findings describe this dataset rather than a real retailer; the method is what carries over.

📄 **[Full report (PDF)](report/Markdown_Pricing_Report.pdf)** · 📓 **[Walkthrough notebook](notebooks/walkthrough.ipynb)** · 🗄️ **[SQL pipeline](sql/markdown_pricing_analysis.sql)**

---

## Key findings

| | |
|---|---|
| **−0.65** | Median price elasticity. Demand is inelastic: a 10% price cut adds about 6.5% more units |
| **5.6×** | How much more strongly in-season products respond to the same discount than off-season ones |
| **+10.7%** | Revenue vs. a no-markdown counterfactual |
| **−22.9%** | Contribution margin vs. the same counterfactual (COGS assumed at 40%) |

1. **The obvious way to measure lift gives the wrong answer.** Sales carry forward from one markdown round to the next, so comparing every round with the original baseline credits later rounds with earlier growth. On that view a 20% markdown looks better than a 35% one (+58% vs +40%). Measured against the period just before each round, it added 12% vs 21%.
2. **Season fit drives the response, not depth, channel or competitor price.** Lift is proportional to depth, and the slope falls into two groups: about 0.85 points of lift per point of discount when a category is in season, about 0.15 when it isn't. The dataset's own "seasonality factor" looks predictive on its own (R² 0.21 → 0.49) but has no effect once season and category are known (p = 0.95). It was a proxy.
3. **Every tier differs, and every step deeper loses more revenue.** One-way ANOVA (F = 3,434, η² = 0.19), a Kruskal–Wallis check and Tukey HSD all agree: each 10-point step adds about 6 points of lift and costs more in revenue, from −7% (10–20% off) to −30% (40–50% off). Only 0.6% of markdown events recovered their own discount.
4. **Revenue went up, margin went down.** Off-season markdowns lost on both counts (revenue −22.5%, margin −44.9%). In season, revenue peaks at 45% off (+32%), but that depth cuts margin by about 40%. Margin is best at 10%.

**Recommendation:** stop off-season markdowns, cap in-season markdowns near 10% (the programme averaged 30%), and judge discounts on incremental units and contribution margin rather than on revenue.

<p align="center">
  <img src="outputs/figures/fig02_attribution.png" width="85%"><br>
  <em>Cumulative vs. incremental lift by markdown round</em>
</p>

<p align="center">
  <img src="outputs/figures/fig03_lift_by_regime.png" width="85%"><br>
  <em>Lift by discount depth for in-season and off-season products, against the lift needed to hold revenue</em>
</p>

<p align="center">
  <img src="outputs/figures/fig09_waterfall.png" width="85%"><br>
  <em>Where the money went over the four rounds (no markdown = 100)</em>
</p>

<p align="center">
  <img src="outputs/figures/fig10_policy.png" width="85%"><br>
  <em>Simulated revenue and margin by discount depth: revenue says go deep, margin says stay shallow</em>
</p>

---

## Approach

<p align="center"><img src="outputs/figures/fig00b_pipeline.png" width="75%"></p>

| Step | What was done | Tools |
|---|---|---|
| 1. Clean | Removed 350 exact duplicate rows; excluded an "Optimal Discount" column that correlates 0.997 with the average markdown (it's derived from the inputs) | SQL |
| 2. Reshape | Unpivoted the four markdown rounds into 173,600 events with `UNION ALL`; `LAG` over each product's periods gives incremental lift | SQL |
| 3. Elasticity | No-intercept OLS of lift on depth interacted with (M1) nothing, (M2) the seasonality factor, (M3) season × category, (M4) both; controls for channel, brand, competitor price gap and round; standard errors clustered by product | statsmodels |
| 4. Compare tiers | One random round per product (so groups are independent); one-way ANOVA, Kruskal–Wallis, Tukey HSD | SciPy, statsmodels |
| 5. Counterfactual | Actual revenue and contribution margin vs. full price at baseline volume; COGS tested at 30/40/50% | SQL, pandas |
| 6. Policy | Simulated a uniform depth (10–50%) for each season × category using its estimated slope | pandas |

<p align="center"><img src="outputs/figures/fig00a_reshape.png" width="80%"></p>

---

## Repository structure

```
markdown-pricing-analysis/
├── data/
│   └── README.md                     how to download the dataset
├── sql/
│   └── markdown_pricing_analysis.sql full SQL pipeline, with expected output under each query
├── src/
│   ├── 01_build_tables.py            runs the SQL in DuckDB -> markdown.duckdb
│   ├── 02_analysis.py                models, elasticity, ANOVA/Tukey, counterfactual, policy -> outputs/
│   └── 03_figures.py                 all figures (PNG + vector PDF) -> outputs/figures/
├── notebooks/
│   └── walkthrough.ipynb             step-by-step walkthrough with outputs
├── outputs/
│   ├── results.json                  every number quoted in the report
│   ├── tables/                       CSV summaries (rounds, tiers, Tukey, counterfactual, policy)
│   └── figures/                      charts and diagrams
├── report/
│   ├── Markdown_Pricing_Report.pdf   full written report
│   └── latex/                        LaTeX source and figures
├── requirements.txt
└── run_all.sh
```

## How to run

```bash
git clone https://github.com/<your-username>/markdown-pricing-analysis.git
cd markdown-pricing-analysis
pip install -r requirements.txt

# 1. download the dataset into data/ (see data/README.md)
# 2. run the pipeline
python src/01_build_tables.py   # SQL: audit, clean, unpivot, lift  (prints each query's result)
python src/02_analysis.py       # statistics -> outputs/results.json, outputs/tables/
python src/03_figures.py        # figures    -> outputs/figures/
```

Or run everything with `bash run_all.sh`. The full pipeline takes under a minute. The SQL file also runs on its own in the DuckDB CLI: `duckdb markdown.duckdb < sql/markdown_pricing_analysis.sql`.

## Data

[Retail Markdown Optimization: Discounts & Sales](https://www.kaggle.com/datasets/arbaaztamboli/retail-markdown-optimization-discounts-and-sales) (Kaggle). A **synthetic** dataset modelled on a French personal-care and beauty retailer: 43,750 rows (43,400 products after de-duplication), three categories, six brands, four seasons, three promotion channels, and a pre-markdown baseline followed by four markdown rounds with sales after each. Product cost is not included.

## Limitations

- **Synthetic data.** The patterns reflect how the data was generated; the sharp two-group structure in the response is an example.
- **No undiscounted control group.** Every product was discounted in every round, so a general trend can't be fully separated from the discount effect. The results are strong associations, not proven causal effects.
- **Assumed cost.** COGS isn't in the data, hence the 30/40/50% sensitivity check.
- **Inventory not modelled.** Markdowns also clear stock that would otherwise be written off, which could justify deeper discounts on aged inventory.
- **No discounts below 10%** in the data, so shallower policies can't be evaluated.

With real transaction data, the next steps would be a randomised holdout of products kept at full price, adding cost and stock age to the margin model, and estimating elasticity per product.

## Tools

SQL (DuckDB) · Python (pandas, NumPy, statsmodels, SciPy, matplotlib) · LaTeX

---

*Author: [Your Name] · Released under the MIT License (code only; the dataset remains under its original Kaggle terms).*
