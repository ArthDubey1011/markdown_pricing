/* =====================================================================
   Retail Markdown Pricing & Margin Impact Analysis
   SQL pipeline (DuckDB)

   Dataset : Retail Markdown Optimization: Discounts & Sales (Kaggle)
   Input   : data/SYNTHETIC Markdown Dataset.csv
   Run     : from the repository root,
             duckdb markdown.duckdb < sql/markdown_pricing_analysis.sql
             (or: python src/01_build_tables.py)

   Sections
     1. Load and audit the raw data
     2. De-duplicate into a clean product table
     3. Unpivot the four markdown rounds (without PIVOT/UNPIVOT)
     4. Incremental lift with LAG
     5. Incremental vs cumulative lift by round        -> report Table 2
     6. Discount tiers and revenue break-even
     7. Season fit and counterfactual revenue / margin -> report Table 6

   Expected outputs are noted under each query.
   ===================================================================== */


/* ---------------------------------------------------------------------
   1. Load and audit the raw data
   --------------------------------------------------------------------- */
CREATE OR REPLACE TABLE raw AS
SELECT * FROM read_csv_auto('data/SYNTHETIC Markdown Dataset.csv', header = true);

-- Row counts, exact duplicates, and the "Optimal Discount" leakage check
SELECT COUNT(*)                                                       AS raw_rows,
       COUNT(DISTINCT Product_ID)                                     AS unique_products,
       COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM raw)) AS duplicate_rows,
       ROUND(CORR("Optimal Discount",
                  (Markdown_1 + Markdown_2 + Markdown_3 + Markdown_4) / 4), 3) AS corr_leak
FROM raw;
-- Expected: 43,750 raw rows | 43,400 products | 350 duplicates | corr 0.997
-- A 0.997 correlation means "Optimal Discount" is derived from the markdowns
-- themselves, so it is excluded from the analysis.


/* ---------------------------------------------------------------------
   2. Clean product table: one row per product
   --------------------------------------------------------------------- */
CREATE OR REPLACE TABLE products AS
SELECT DISTINCT
       Product_ID, Category, Brand, Season, Product_Name,
       Original_Price, Competitor_Price, Seasonality_Factor,
       Markdown_1, Markdown_2, Markdown_3, Markdown_4,
       Historical_Sales, Sales_After_M1, Sales_After_M2, Sales_After_M3, Sales_After_M4,
       Promotion_Type,
       "Customer Ratings" AS Customer_Rating,
       "Return Rate"      AS Return_Rate,
       "Optimal Discount" AS Optimal_Discount,
       (Original_Price - Competitor_Price) / Competitor_Price AS price_gap_vs_competitor
FROM raw;
-- Expected: 43,400 rows


/* ---------------------------------------------------------------------
   3. Unpivot: one row per product per period (baseline = stage 0)
   --------------------------------------------------------------------- */
-- UNION ALL (not UNION): rows are distinct by construction, so UNION's
-- de-duplication and sort would be wasted work.
CREATE OR REPLACE TABLE long_events AS
SELECT Product_ID, 0 AS stage, 0.0 AS markdown, Historical_Sales AS sales FROM products
UNION ALL SELECT Product_ID, 1, Markdown_1, Sales_After_M1 FROM products
UNION ALL SELECT Product_ID, 2, Markdown_2, Sales_After_M2 FROM products
UNION ALL SELECT Product_ID, 3, Markdown_3, Sales_After_M3 FROM products
UNION ALL SELECT Product_ID, 4, Markdown_4, Sales_After_M4 FROM products;
-- Expected: 217,000 rows (43,400 x 5)

-- Sanity check: every product must have exactly 5 periods, or LAG would
-- compare against the wrong period
SELECT COUNT(*) AS products_with_missing_periods
FROM (SELECT Product_ID FROM long_events GROUP BY Product_ID HAVING COUNT(*) <> 5);
-- Expected: 0


/* ---------------------------------------------------------------------
   4. Incremental lift: each round vs the period just before it
   --------------------------------------------------------------------- */
-- The CTE is needed because WHERE runs before window functions: filtering
-- stage > 0 in the same query would remove the baseline before LAG sees it.
-- "* 1.0" avoids integer division on integer sales columns.
CREATE OR REPLACE TABLE events AS
WITH with_prev AS (
    SELECT *,
           LAG(sales)         OVER w AS prev_sales,
           FIRST_VALUE(sales) OVER w AS baseline_sales
    FROM long_events
    WINDOW w AS (PARTITION BY Product_ID ORDER BY stage)
)
SELECT w.Product_ID, w.stage, w.markdown, w.sales, w.prev_sales,
       w.sales * 1.0 / w.prev_sales - 1      AS step_lift,        -- incremental lift
       w.sales * 1.0 / w.baseline_sales - 1  AS cum_lift,         -- cumulative (not used for inference)
       w.markdown / (1 - w.markdown)         AS breakeven_lift,   -- lift needed to hold revenue
       CASE WHEN w.markdown < 0.20 THEN '10-20%'
            WHEN w.markdown < 0.30 THEN '20-30%'
            WHEN w.markdown < 0.40 THEN '30-40%'
            ELSE '40-50%' END                AS discount_band,
       p.Season, p.Category, p.Brand, p.Promotion_Type, p.Seasonality_Factor,
       p.Original_Price, p.price_gap_vs_competitor
FROM with_prev w
JOIN products p USING (Product_ID)
WHERE w.stage > 0;
-- Expected: 173,600 markdown events (43,400 x 4)


/* ---------------------------------------------------------------------
   5. Incremental vs cumulative lift by round  (report Table 2)
   --------------------------------------------------------------------- */
WITH lifts AS (
    SELECT Product_ID, stage, markdown,
           sales * 1.0 / LAG(sales)         OVER w - 1 AS incr_lift,  -- vs previous period
           sales * 1.0 / FIRST_VALUE(sales) OVER w - 1 AS cum_lift,   -- vs original baseline
           markdown / (1 - markdown)                   AS breakeven_lift
    FROM long_events
    WINDOW w AS (PARTITION BY Product_ID ORDER BY stage)
)
SELECT stage                                AS round,
       ROUND(100 * AVG(markdown), 1)        AS mean_depth_pct,
       ROUND(100 * AVG(incr_lift), 1)       AS incr_lift_mean_pct,
       ROUND(100 * MEDIAN(incr_lift), 1)    AS incr_lift_median_pct,
       ROUND(100 * AVG(cum_lift), 1)        AS cum_lift_pct,
       ROUND(100 * AVG(breakeven_lift), 1)  AS breakeven_pct
FROM lifts
WHERE stage > 0
GROUP BY stage
ORDER BY stage;
-- Expected:
-- round | depth | incr mean | incr median | cumulative | break-even
--   1   | 25.1  |   15.0    |    14.0     |    15.0    |   35.3
--   2   | 35.0  |   20.9    |    21.2     |    40.0    |   56.7
--   3   | 20.0  |   11.9    |    11.7     |    58.3    |   25.6
--   4   | 40.0  |   23.9    |    25.5     |   100.8    |   68.3
-- Cumulative lift ranks round 3 (20% off) above round 2 (35% off),
-- incremental lift shows the opposite. All analysis uses incremental lift.


/* ---------------------------------------------------------------------
   6. Discount tiers and revenue break-even (all events)
   --------------------------------------------------------------------- */
SELECT discount_band                                                    AS tier,
       COUNT(*)                                                         AS events,
       ROUND(100 * AVG(step_lift), 1)                                   AS unit_lift_pct,
       ROUND(100 * AVG((1 - markdown) * (1 + step_lift) - 1), 1)        AS period_revenue_change_pct,
       ROUND(100 * AVG(CASE WHEN step_lift >= breakeven_lift THEN 1.0 ELSE 0 END), 2)
                                                                        AS pct_events_breaking_even
FROM events
GROUP BY tier
ORDER BY tier;
-- Deeper tiers sell more units and lose more revenue, and almost no event breaks even.
-- (The ANOVA / Tukey tests in the report use one randomly chosen round per product,
--  run in Python, so their tier figures differ slightly from these all-event averages.)


/* ---------------------------------------------------------------------
   7. Season fit and counterfactual revenue / margin  (report Table 6)
   --------------------------------------------------------------------- */
-- Counterfactual: every product stays at full price and baseline volume.
-- Contribution margin assumes COGS = 40% of original price.
-- ROLLUP returns the in-season group, the off-season group and the total in one pass,
-- COALESCE relabels ROLLUP's NULL total row.
WITH seg AS (                                   -- response slope per season x category
    SELECT Season, Category,
           MEDIAN(step_lift / markdown) AS slope
    FROM events
    GROUP BY Season, Category
),
flagged AS (
    SELECT e.*,
           p.Historical_Sales AS s0,
           CASE WHEN s.slope > 0.5 THEN 'In season' ELSE 'Off season' END AS season_fit
    FROM events e
    JOIN products p USING (Product_ID)
    JOIN seg s ON s.Season = e.Season AND s.Category = e.Category
)
SELECT COALESCE(season_fit, 'All products')                          AS scope,
       ROUND(100 * (SUM(Original_Price * (1 - markdown) * sales)
                    / SUM(Original_Price * s0) - 1), 1)              AS revenue_vs_no_markdown_pct,
       ROUND(100 * (SUM(Original_Price * (1 - markdown - 0.4) * sales)
                    / SUM(Original_Price * 0.6 * s0) - 1), 1)        AS margin_vs_no_markdown_pct
FROM flagged
GROUP BY ROLLUP (season_fit)
ORDER BY scope;
-- Expected:
-- All products : revenue +10.7 | margin -22.9
-- In season    : revenue +21.6 | margin -15.7
-- Off season   : revenue -22.5 | margin -44.9
