# Data

This project uses **Retail Markdown Optimization: Discounts & Sales** from Kaggle:
https://www.kaggle.com/datasets/arbaaztamboli/retail-markdown-optimization-discounts-and-sales

The CSV isn't committed to this repository. To run the pipeline:

1. Download the dataset from the link above (a free Kaggle account is needed).
2. Unzip it and place the file here as:

   ```
   data/SYNTHETIC Markdown Dataset.csv
   ```

3. From the repository root, run `python src/01_build_tables.py`.

With the Kaggle CLI instead:

```bash
kaggle datasets download -d arbaaztamboli/retail-markdown-optimization-discounts-and-sales -p data --unzip
```

**About the data:** synthetic, 43,750 rows × 22 columns (350 are exact duplicates). Each row is a product with its category, brand, season, promotion channel, original and competitor price, a seasonality factor, baseline sales, four markdown levels and the sales after each. There is no product cost column.
