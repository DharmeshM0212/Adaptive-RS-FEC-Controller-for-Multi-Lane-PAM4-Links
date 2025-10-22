import pandas as pd
df = pd.read_parquet("results/m5_ctrl_timeline.parquet")
print("blocks uniq:", len(pd.to_numeric(df["block"], errors="coerce").dropna().unique()))
print("first 5 alloc:", df["nsym_alloc"].head().to_list())
print("alloc uniq rows:", len(df["nsym_alloc"].apply(tuple).unique()))
