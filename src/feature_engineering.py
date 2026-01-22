import pandas as pd
import numpy as np
from tqdm import tqdm

def calc_consecutive_dots(group):
    group = group.sort_values('over').reset_index(drop = True)

    consecutive_dots = []
    current_count = 0

    for idx, row in group.iterrows():
        is_dot = (row['runs_scored'] == 0) and (row['extras'] == 0)

        consecutive_dots.append(current_count)

        if is_dot:
            current_count += 1
        else:
            current_count = 0

    group['consecutive_dots_faced'] = consecutive_dots
    return group

def engineer_features(df):
    print("Starting feature engineering...\n")
    print("Step 1: Filtering dismissal types..")
    pressure_dismissals = [
        'bowled','caught','lbw','stumped','caught and bowled','hit wicket'
        ]

    df['pressure_dismissal'] = df.apply(
        lambda x: 1 if (x['wicket'] == 1 and
                        x['dismissal_kind'] in pressure_dismissals and
                        x['batter'] == x['dismissed_batter']) else 0,
        axis = 1
    )

    print(f" Total dismissals : {df['wicket'].sum():,}")
    print(f" Pressure dismissals : {df['pressure_dismissal'].sum():,}")

    # Calculate consecutive dots for each batter in each innings
    print("\nStep 2: Calculating consecutive dot balls...")
    tqdm.pandas(desc="Processing batters")

    df = df.groupby(['match_id','innings_num','batter']).progress_apply(calc_consecutive_dots).reset_index(drop=True)

    df['consecutive_dots_capped'] =df['consecutive_dots_faced'].apply(
        lambda x: 5 if x >= 5 else x
    )

    #Classifying innings phases
    print("\nStep 3: Classifying match phases...")
    df['match_phase'] = pd.cut(
        df['over'],
        bins = [-0.1,5.9,15.9,20.1],
        labels = ['Powerplay','Middle overs','Death']
    )

    #Balls faced until ow (Cumulative for each batter)
    print("\nStep 4: Calculating the amount of balls faced...")
    df['balls_faced_so_far'] = df.groupby(
        ['match_id','innings_num','batter']
    ).cumcount() + 1

    #Create a target variable : dismissed_nex_ball
    print("\nStep 5: Creating target variable...")
    df = df.sort_values(['match_id','innings_num','over']).reset_index(drop = True)

    df['dismissal_next_ball'] = df.groupby(
        ['match_id','innings_num','batter']
    )['pressure_dismissal'].shift(-1).fillna(0).astype(int)

    #Boundary drought time
    print("\nStep 6: Calculating boundary drought...")

    def calc_boundary_drought(group):
        group = group.sort_values('over').reset_index(drop = True)
        boundary_drought = []
        balls_since_boundary = 0

        for idx, row in group.iterrows():
            boundary_drought.append(balls_since_boundary)

            if row['runs_scored'] in [4,6]:
                balls_since_boundary = 0

            else:
                balls_since_boundary += 1

        group['boundary_drought'] = boundary_drought
        return group

    df = df.groupby(['match_id','innings_num','batter']).apply(
        calc_boundary_drought
    ).reset_index(drop = True)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeatures created:")
    print("  - consecutive_dots_faced")
    print("  - consecutive_dots_capped")
    print("  - match_phase")
    print("  - balls_faced_so_far")
    print("  - dismissal_next_ball")
    print("  - boundary_drought")

    return df

def filter_dataset(df,min_balls = 20):
    print(f"\nFiltering batters with atleast {min_balls} balls faced...")

    batter_counts = df.groupby('batter').size()
    valid_batters = batter_counts[batter_counts >= min_balls].index

    df_filtered = df[df['batter'].isin(valid_batters)].copy()

    print(f"  Before filtering: {df['batter'].nunique():,} batters")
    print(f"  After filtering: {df_filtered['batter'].nunique():,} batters")
    print(f"  Deliveries retained: {len(df_filtered):,} / {len(df):,} "
          f"({len(df_filtered) / len(df) * 100:.1f}%)")

    return df_filtered


def main():
    """Main execution function"""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60 + "\n")

    # Load raw data
    print("Loading raw ball-by-ball data...")
    df = pd.read_csv('/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/DOR_Pressure_Analysis/data/processed/raw_ballbyball_data.csv')
    print(f"Loaded {len(df):,} deliveries\n")

    # Engineer features
    df_engineered = engineer_features(df)

    # Filter dataset
    df_filtered = filter_dataset(df_engineered, min_balls=20)

    # Save
    output_path = '/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/DOR_Pressure_Analysis/data/processed/analysis_ready_data.csv'
    df_filtered.to_csv(output_path, index=False)

    print(f"\n✓ Analysis-ready data saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total deliveries: {len(df_filtered):,}")
    print(f"Unique batters: {df_filtered['batter'].nunique():,}")
    print(f"Unique matches: {df_filtered['match_id'].nunique():,}")
    print(f"\nDot ball distribution:")
    print(df_filtered['consecutive_dots_capped'].value_counts().sort_index())
    print(f"\nDismissal events: {df_filtered['dismissal_next_ball'].sum():,}")


if __name__ == "__main__":
    main()