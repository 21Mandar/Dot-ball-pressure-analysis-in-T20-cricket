import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "raw_ballbyball_data.csv")
df = pd.read_csv(DATA_PATH)

# Quick checks
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nCompetition breakdown:")
print(df['competition_type'].value_counts())
print(f"\nTotal dismissals: {df['wicket'].sum()}")

print("="*60)
print("DATASET SUFFICIENCY CHECK")
print("="*60)

# 1. Total sample size
print(f"\n1. SAMPLE SIZE:")
print(f"   Total deliveries: {len(df):,}")
print(f"   Required minimum: 50,000 ✓" if len(df) >= 50000 else "   ⚠️ Warning: <50K rows")

# 2. Dismissal events
dismissals = df['dismissal_next_ball'].sum()
print(f"\n2. DISMISSAL EVENTS:")
print(f"   Total dismissals: {dismissals:,}")
print(f"   Required minimum: 1,500 ✓" if dismissals >= 1500 else "   ⚠️ Warning: <1500 dismissals")

# 3. Sample size per dot ball category
print(f"\n3. SAMPLE SIZE BY CONSECUTIVE DOTS:")
dot_counts = df.groupby('consecutive_dots_capped').agg({
    'dismissal_next_ball': ['count', 'sum']
})
dot_counts.columns = ['Total_Balls', 'Dismissals']
print(dot_counts)
print()

# Check if each category has enough data
for dots in range(6):
    total = dot_counts.loc[dots, 'Total_Balls'] if dots in dot_counts.index else 0
    dismissals = dot_counts.loc[dots, 'Dismissals'] if dots in dot_counts.index else 0
    status = "✓" if total >= 1000 and dismissals >= 10 else "⚠️"
    print(f"   {dots} dots: {status} ({total:,} balls, {dismissals} dismissals)")

# 4. Unique batters
print(f"\n4. UNIQUE BATTERS:")
print(f"   Count: {df['batter'].nunique():,}")
print(f"   Required minimum: 500 ✓" if df['batter'].nunique() >= 500 else "   ⚠️ Warning: <500 batters")

# 5. Unique matches
print(f"\n5. UNIQUE MATCHES:")
print(f"   Count: {df['match_id'].nunique():,}")
print(f"   Required minimum: 300 ✓" if df['match_id'].nunique() >= 300 else "   ⚠️ Warning: <300 matches")

# 6. Overall verdict
print(f"\n" + "="*60)
sufficient = (
    len(df) >= 50000 and
    dismissals >= 1500 and
    df['batter'].nunique() >= 500 and
    df['match_id'].nunique() >= 300
)

if sufficient:
    print("✅ VERDICT: Dataset is SUFFICIENT for analysis")
    print("   You have enough data for robust statistical conclusions")
else:
    print("⚠️ VERDICT: Dataset may be MARGINAL")
    print("   Analysis is possible but may need adjustments")

print("="*60)

