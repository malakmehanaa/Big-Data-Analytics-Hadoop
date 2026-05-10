import pandas as pd

files = {
    'customers':     r'C:\Users\A\Downloads\customers-10000.csv',
    'leads':         r'C:\Users\A\Downloads\leads-10000.csv',
    'organizations': r'C:\Users\A\Downloads\organizations-10000.csv',
    'people':        r'C:\Users\A\Downloads\people-10000.csv',
    'products':      r'C:\Users\A\Downloads\products-10000.csv',
}
dfs = []
for name, path in files.items():
    df = pd.read_csv(path)
    # Rename all columns except Index with source suffix to avoid duplicates
    df.columns = ['Index'] + [f"{col}_{name}" for col in df.columns[1:]]
    dfs.append(df)

# Horizontal merge on Index
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on='Index', how='outer')

merged.to_csv('merged_horizontal.csv', index=False)
merged.to_excel('merged_horizontal.xlsx', index=False)

print(f"Done! {merged.shape[0]} rows x {merged.shape[1]} columns")
