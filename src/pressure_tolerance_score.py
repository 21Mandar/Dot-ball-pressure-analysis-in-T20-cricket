"""
Calculate Pressure Tolerance Score (PTS) and Behavioral Metrics
Per player, per phase
"""

import pandas as pd
import numpy as np


def calculate_pressure_metrics(input_path, output_path):
    """
    Calculate PTS and behavioral response metrics for each player-phase
    """

    print("=" * 70)
    print("CALCULATING PRESSURE TOLERANCE METRICS")
    print("=" * 70)

    # Load data
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df):,} player-phase-innings records")

    # Phase-specific thresholds (75th percentile)
    thresholds = {
        'PP': 0.57,
        'MO': 0.45,
        'Death': 0.40
    }

    # Classify each record as high or low pressure
    print("\nClassifying pressure situations...")
    df['is_high_pressure'] = df.apply(
        lambda row: 1 if row['dot_density'] > thresholds[row['phase']] else 0,
        axis=1
    )

    # Verify distribution
    print("\nPressure distribution by phase:")
    for phase in ['PP', 'MO', 'Death']:
        phase_df = df[df['phase'] == phase]
        high_pct = phase_df['is_high_pressure'].mean() * 100
        print(f"  {phase}: {high_pct:.1f}% high pressure (threshold: {thresholds[phase]})")

    # Minimum innings requirement per player-phase
    MIN_INNINGS = 10

    print(f"\nAggregating player metrics (minimum {MIN_INNINGS} innings per phase)...")

    # Calculate metrics for each player-phase combination
    results = []

    for phase in ['PP', 'MO', 'Death']:
        phase_df = df[df['phase'] == phase]

        for player in phase_df['player'].unique():
            player_phase_df = phase_df[phase_df['player'] == player]

            # Skip if insufficient innings
            if len(player_phase_df) < MIN_INNINGS:
                continue

            # Split into high and low pressure
            high_pressure = player_phase_df[player_phase_df['is_high_pressure'] == 1]
            low_pressure = player_phase_df[player_phase_df['is_high_pressure'] == 0]

            # Need data in both conditions to calculate PTS
            if len(high_pressure) < 3 or len(low_pressure) < 3:
                continue

            # Calculate metrics
            total_innings = len(player_phase_df)
            high_pressure_innings = len(high_pressure)
            low_pressure_innings = len(low_pressure)

            # Dismissal rates
            high_dismissal_rate = high_pressure['dismissed_in_phase'].mean()
            low_dismissal_rate = low_pressure['dismissed_in_phase'].mean()

            # PTS calculation (with smoothing to avoid division issues)
            # PTS > 1 means MORE likely to get out under pressure
            # PTS < 1 means LESS likely to get out under pressure
            if low_dismissal_rate > 0:
                pts = high_dismissal_rate / low_dismissal_rate
            else:
                pts = np.nan  # Can't calculate if never dismissed in low pressure

            # Strike rate metrics
            high_sr = high_pressure['strike_rate'].mean()
            low_sr = low_pressure['strike_rate'].mean()
            sr_delta = high_sr - low_sr  # Positive = speeds up under pressure

            # Boundary metrics
            high_boundary_pct = high_pressure['boundary_pct'].mean()
            low_boundary_pct = low_pressure['boundary_pct'].mean()
            boundary_delta = high_boundary_pct - low_boundary_pct  # Positive = more aggressive

            # Overall averages
            overall_sr = player_phase_df['strike_rate'].mean()
            overall_dismissal_rate = player_phase_df['dismissed_in_phase'].mean()
            overall_dot_density = player_phase_df['dot_density'].mean()

            results.append({
                'player': player,
                'phase': phase,
                'total_innings': total_innings,
                'high_pressure_innings': high_pressure_innings,
                'low_pressure_innings': low_pressure_innings,
                # Core metrics
                'pts': round(pts, 3) if not np.isnan(pts) else np.nan,
                'high_dismissal_rate': round(high_dismissal_rate, 4),
                'low_dismissal_rate': round(low_dismissal_rate, 4),
                # Behavioral metrics
                'sr_delta': round(sr_delta, 2),
                'high_pressure_sr': round(high_sr, 2),
                'low_pressure_sr': round(low_sr, 2),
                'boundary_delta': round(boundary_delta, 4),
                'high_pressure_boundary_pct': round(high_boundary_pct, 4),
                'low_pressure_boundary_pct': round(low_boundary_pct, 4),
                # Context
                'overall_sr': round(overall_sr, 2),
                'overall_dismissal_rate': round(overall_dismissal_rate, 4),
                'overall_dot_density': round(overall_dot_density, 4)
            })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print(f"\nGenerated metrics for {len(results_df):,} player-phase combinations")
    print(f"Unique players with sufficient data: {results_df['player'].nunique():,}")

    # Save
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("PTS SUMMARY BY PHASE")
    print("=" * 70)

    for phase in ['PP', 'MO', 'Death']:
        phase_results = results_df[results_df['phase'] == phase]
        valid_pts = phase_results['pts'].dropna()

        print(f"\n{phase}:")
        print(f"  Players: {len(phase_results)}")
        print(f"  PTS Mean:   {valid_pts.mean():.3f}")
        print(f"  PTS Median: {valid_pts.median():.3f}")
        print(f"  PTS Std:    {valid_pts.std():.3f}")
        print(f"  PTS Range:  {valid_pts.min():.3f} to {valid_pts.max():.3f}")

    print("\n" + "=" * 70)
    print("BEHAVIORAL RESPONSE SUMMARY")
    print("=" * 70)

    for phase in ['PP', 'MO', 'Death']:
        phase_results = results_df[results_df['phase'] == phase]

        print(f"\n{phase}:")
        print(f"  Avg SR Delta: {phase_results['sr_delta'].mean():+.2f}")
        print(f"  Avg Boundary Delta: {phase_results['boundary_delta'].mean():+.4f}")

        # Count behavioral patterns
        speedup = (phase_results['sr_delta'] > 5).sum()
        slowdown = (phase_results['sr_delta'] < -5).sum()
        neutral = len(phase_results) - speedup - slowdown

        print(f"  Speed up under pressure (SR +5):  {speedup} ({speedup / len(phase_results) * 100:.1f}%)")
        print(f"  Slow down under pressure (SR -5): {slowdown} ({slowdown / len(phase_results) * 100:.1f}%)")
        print(f"  Neutral response:                 {neutral} ({neutral / len(phase_results) * 100:.1f}%)")

    return results_df


if __name__ == "__main__":
    input_path = 'data/processed/player_phase_innings.csv'
    output_path = 'data/processed/player_pressure_metrics.csv'

    df = calculate_pressure_metrics(input_path, output_path)

    # Show some interesting players
    print("\n" + "=" * 70)
    print("SAMPLE: HIGH PTS PLAYERS (Struggle Under Pressure)")
    print("=" * 70)
    print(df.nlargest(10, 'pts')[['player', 'phase', 'pts', 'sr_delta', 'total_innings']].to_string(index=False))

    print("\n" + "=" * 70)
    print("SAMPLE: LOW PTS PLAYERS (Thrive Under Pressure)")
    print("=" * 70)
    print(df.nsmallest(10, 'pts')[['player', 'phase', 'pts', 'sr_delta', 'total_innings']].to_string(index=False))