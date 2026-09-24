"""Step 1: run the SQL pipeline in DuckDB.

Builds `markdown.duckdb` in the repository root with three tables:
  products     one row per product (43,400)
  long_events  one row per product per period, baseline included (217,000)
  events       one row per markdown event with incremental lift (173,600)
and prints the result of every query in sql/markdown_pricing_analysis.sql.

Run from the repository root:  python src/01_build_tables.py
"""
import os
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "sql" / "markdown_pricing_analysis.sql"
DB_FILE = ROOT / "markdown.duckdb"
CSV = ROOT / "data" / "SYNTHETIC Markdown Dataset.csv"


def statements(sql: str):
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)        # drop block comments
    for stmt in sql.split(";"):
        if re.sub(r"--.*", "", stmt).strip():
            yield stmt.strip()


def main():
    if not CSV.exists():
        raise SystemExit(f"Dataset not found at {CSV}. See data/README.md for download instructions.")
    os.chdir(ROOT)                                          # SQL uses paths relative to the repo root
    con = duckdb.connect(str(DB_FILE))
    for stmt in statements(SQL_FILE.read_text()):
        result = con.execute(stmt)
        code = re.sub(r"--.*", "", stmt).strip().upper()
        if code.startswith(("SELECT", "WITH")):
            print(result.df().to_string(index=False), end="\n\n")
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ["products", "long_events", "events"]}
    print("Tables built:", counts)
    con.close()


if __name__ == "__main__":
    main()
