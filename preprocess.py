import pandas as pd

# Alberta bounding box
LAT_MIN, LAT_MAX = 49, 60
LON_MIN, LON_MAX = -120, -110

files = [
    'data/raw/cwfis_fwi2000sv3.0_ll.csv',
    'data/raw/cwfis_fwi2010sv3.0_ll.csv',
    'data/raw/cwfis_fwi2020sv3.0_ll.csv'
]

# Columns we actually need
keep_cols = ['rep_date', 'name', 'lat', 'lon', 'temp', 'rh', 
             'ws', 'precip', 'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi']

chunks = []
for f in files:
    print(f"Processing {f}...")
    df = pd.read_csv(f, usecols=keep_cols)
    alberta = df[
        (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)
    ]
    chunks.append(alberta)
    print(f"  Alberta records: {len(alberta):,}")

combined = pd.concat(chunks, ignore_index=True)
combined['rep_date'] = pd.to_datetime(combined['rep_date'])
combined = combined.sort_values('rep_date').reset_index(drop=True)

combined.to_csv('data/raw/fwi_alberta_2000_2025.csv', index=False)
print(f"\nDone. Total records: {len(combined):,}")
print(f"Date range: {combined['rep_date'].min()} to {combined['rep_date'].max()}")
df = pd.read_csv('data/raw/fwi_alberta_2000_2025.csv')
print("Total records:", len(df))
print("\nNull counts:")
print(df.isnull().sum())
print("\nNull percentages:")
print((df.isnull().sum() / len(df) * 100).round(1))