"""
Transform ball-by-ball data to player-phase-innings aggregates
For Phase-Specific Pressure Response Analysis
"""

import pandas as pd
import numpy as np


def create_player_phase_innings_data(input_path, output_path):
    """
    Aggregate ball-by-ball data to player-phase-innings level
    """

    print("=" * 70)
    print("CREATING PLAYER-PHASE-INNINGS DATASET")
    print("=" * 70)

    # Load data
    print("\nLoading ball-by-ball data...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} deliveries")

    # Filter to standard innings (1 and 2 only)
    print("\nFiltering to standard innings (1 & 2)...")
    df = df[df['innings_num'].isin([1, 2])].copy()
    print(f"  Remaining: {len(df):,} deliveries")

    # Create is_dot and is_boundary flags
    print("\nCreating helper columns...")
    df['is_dot'] = ((df['runs_scored'] == 0) & (df['extras'] == 0)).astype(int)
    df['is_boundary'] = df['runs_scored'].isin([4, 6]).astype(int)

    # Standardize phase names for cleaner output
    phase_mapping = {
        'Powerplay': 'PP',
        'Middle overs': 'MO',
        'Death': 'Death'
    }
    df['phase'] = df['match_phase'].map(phase_mapping)

    # Aggregate to player-phase-innings level
    print("\nAggregating to player-phase-innings level...")

    agg_df = df.groupby(
        ['match_id', 'batter', 'innings_num', 'phase']
    ).agg(
        balls_faced=('batter', 'count'),
        dots_faced=('is_dot', 'sum'),
        runs_scored=('runs_scored', 'sum'),
        boundaries=('is_boundary', 'sum'),
        dismissed_in_phase=('pressure_dismissal', 'max'),
        # Keep some context fields
        batting_team=('batting_team', 'first'),
        competition_type=('competition_type', 'first'),
        date=('date', 'first')
    ).reset_index()

    # Rename for clarity
    agg_df = agg_df.rename(columns={
        'batter': 'player',
        'innings_num': 'innings'
    })

    # Calculate derived metrics
    print("\nCalculating derived metrics...")
    agg_df['dot_density'] = (agg_df['dots_faced'] / agg_df['balls_faced']).round(4)
    agg_df['strike_rate'] = ((agg_df['runs_scored'] / agg_df['balls_faced']) * 100).round(2)
    agg_df['boundary_pct'] = (agg_df['boundaries'] / agg_df['balls_faced']).round(4)

    # Apply minimum ball thresholds per phase
    print("\nApplying minimum ball thresholds...")
    print("  PP: >= 10 balls")
    print("  MO: >= 10 balls")
    print("  Death: >= 5 balls")

    threshold_map = {'PP': 10, 'MO': 10, 'Death': 5}

    filtered_df = agg_df[
        agg_df.apply(lambda row: row['balls_faced'] >= threshold_map[row['phase']], axis=1)
    ].copy()

    print(f"\n  Before filtering: {len(agg_df):,} player-phase-innings records")
    print(f"  After filtering:  {len(filtered_df):,} player-phase-innings records")

    # Reorder columns for clarity
    column_order = [
        'match_id', 'player', 'innings', 'phase', 'batting_team',
        'balls_faced', 'dots_faced', 'dot_density',
        'runs_scored', 'strike_rate',
        'boundaries', 'boundary_pct',
        'dismissed_in_phase',
        'competition_type', 'date'
    ]
    filtered_df = filtered_df[column_order]

    # Save
    filtered_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

    # Summary stats
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTotal records: {len(filtered_df):,}")
    print(f"Unique players: {filtered_df['player'].nunique():,}")
    print(f"Unique matches: {filtered_df['match_id'].nunique():,}")

    print("\nRecords by phase:")
    print(filtered_df['phase'].value_counts().to_string())

    print("\nRecords by innings:")
    print(filtered_df['innings'].value_counts().to_string())

    print("\nDot density distribution:")
    print(filtered_df['dot_density'].describe().round(3).to_string())

    print("\nDismissal rate by phase:")
    print(filtered_df.groupby('phase')['dismissed_in_phase'].mean().round(4).to_string())

    print("\nSample records:")
    print(filtered_df.head(10).to_string())

    return filtered_df


if __name__ == "__main__":
    input_path = 'data/processed/analysis_ready_data.csv'
    output_path = 'data/processed/player_phase_innings.csv'

    df = create_player_phase_innings_data(input_path, output_path)