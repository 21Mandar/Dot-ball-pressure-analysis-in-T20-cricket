"""
Visualizations for Pressure Tolerance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
Path("results/figures").mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data():
    """Load the pressure metrics data"""
    df = pd.read_csv('data/processed/player_pressure_metrics.csv')
    return df


# =============================================================================
# VISUALIZATION 1: PTS Distribution by Phase
# =============================================================================
def plot_pts_distribution(df):
    """Box plot and violin plot of PTS by phase"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter for reliable data
    reliable = df[(df['high_pressure_innings'] >= 5) & (df['pts'] > 0) & (df['pts'] < 10)]

    # Box plot
    ax1 = axes[0]
    phase_order = ['PP', 'MO', 'Death']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    box = reliable.boxplot(column='pts', by='phase', ax=ax1,
                           positions=[1, 2, 3], widths=0.6, patch_artist=True)

    ax1.set_xticklabels(['PP', 'MO', 'Death'])
    ax1.set_xlabel('Match Phase', fontsize=12)
    ax1.set_ylabel('Pressure Tolerance Score (PTS)', fontsize=12)
    ax1.set_title('PTS Distribution by Phase', fontsize=14, fontweight='bold')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='PTS = 1.0 (No Effect)')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='PTS = 2.0 (Vulnerable)')
    ax1.legend(loc='upper right')
    plt.suptitle('')

    # Violin plot
    ax2 = axes[1]
    sns.violinplot(data=reliable, x='phase', y='pts', order=phase_order,
                   palette=colors, ax=ax2)
    ax2.set_xlabel('Match Phase', fontsize=12)
    ax2.set_ylabel('Pressure Tolerance Score (PTS)', fontsize=12)
    ax2.set_title('PTS Density by Phase', fontsize=14, fontweight='bold')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/01_pts_distribution_by_phase.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/01_pts_distribution_by_phase.png")
    plt.close()


# =============================================================================
# VISUALIZATION 2: SR Delta vs PTS Scatter
# =============================================================================
def plot_sr_delta_vs_pts(df):
    """Scatter plot showing relationship between SR change and dismissal risk"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    reliable = df[(df['high_pressure_innings'] >= 5) & (df['pts'] > 0) & (df['pts'] < 10)]

    phases = ['PP', 'MO', 'Death']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for idx, (phase, color) in enumerate(zip(phases, colors)):
        ax = axes[idx]
        phase_df = reliable[reliable['phase'] == phase]

        ax.scatter(phase_df['sr_delta'], phase_df['pts'],
                   alpha=0.5, c=color, s=50, edgecolors='white', linewidth=0.5)

        # Add quadrant lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

        # Quadrant labels
        ax.text(0.05, 0.95, 'Speeds Up +\nSurvives', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', color='green', fontweight='bold')
        ax.text(0.95, 0.95, 'Slows Down +\nSurvives', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', ha='right', color='blue', fontweight='bold')
        ax.text(0.05, 0.05, 'Speeds Up +\nGets Out', transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', color='orange', fontweight='bold')
        ax.text(0.95, 0.05, 'Slows Down +\nGets Out', transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', ha='right', color='red', fontweight='bold')

        ax.set_xlabel('Strike Rate Delta (Under Pressure)', fontsize=11)
        ax.set_ylabel('PTS (Higher = More Vulnerable)', fontsize=11)
        ax.set_title(f'{phase}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 6)

    plt.suptitle('Behavioral Response vs Dismissal Risk Under Pressure',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/02_sr_delta_vs_pts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/02_sr_delta_vs_pts.png")
    plt.close()


# =============================================================================
# VISUALIZATION 3: Star Players Heatmap
# =============================================================================
def plot_star_players_heatmap(df):
    """Heatmap of PTS for star players across phases"""

    star_players = [
        'V Kohli', 'RG Sharma', 'AB de Villiers', 'MS Dhoni', 'JC Buttler',
        'KL Rahul', 'DA Warner', 'CH Gayle', 'SK Raina', 'S Dhawan',
        'HH Pandya', 'RA Jadeja', 'GJ Maxwell', 'KS Williamson',
        'Babar Azam', 'Mohammad Rizwan', 'Q de Kock', 'SPD Smith',
        'RR Pant', 'SV Samson', 'F du Plessis', 'JM Bairstow'
    ]

    stars_df = df[df['player'].isin(star_players)]

    # Pivot for heatmap
    pivot_df = stars_df.pivot(index='player', columns='phase', values='pts')
    pivot_df = pivot_df.reindex(columns=['PP', 'MO', 'Death'])

    # Sort by average PTS
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg', ascending=True)
    pivot_df = pivot_df.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=1.5, vmin=0, vmax=5, linewidths=0.5,
                cbar_kws={'label': 'PTS (Higher = More Vulnerable)'}, ax=ax)

    ax.set_xlabel('Match Phase', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    ax.set_title('Pressure Tolerance Score by Player and Phase\n(Green = Resistant, Red = Vulnerable)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/03_star_players_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/03_star_players_heatmap.png")
    plt.close()


# =============================================================================
# VISUALIZATION 4: Player Profile Cards (Top Examples)
# =============================================================================
def plot_player_profiles(df):
    """Create profile visualizations for selected players"""

    # Select interesting contrasting players
    profiles = {
        'JC Buttler': 'Elite PP, Vulnerable Death',
        'V Kohli': 'Anchor - PP Vulnerable',
        'GJ Maxwell': 'Death Specialist',
        'MS Dhoni': 'The Finisher'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (player, description) in enumerate(profiles.items()):
        ax = axes[idx]
        player_df = df[df['player'] == player].copy()

        if len(player_df) == 0:
            continue

        phases = ['PP', 'MO', 'Death']
        x = np.arange(len(phases))
        width = 0.35

        # Get PTS values (fill NaN with 0)
        pts_values = []
        sr_delta_values = []
        for phase in phases:
            phase_data = player_df[player_df['phase'] == phase]
            if len(phase_data) > 0:
                pts_values.append(phase_data['pts'].values[0])
                sr_delta_values.append(phase_data['sr_delta'].values[0])
            else:
                pts_values.append(np.nan)
                sr_delta_values.append(np.nan)

        # Bar colors based on PTS
        colors = ['#2ecc71' if p < 1.0 else '#e74c3c' if p > 2.0 else '#f39c12'
                  for p in pts_values]

        bars = ax.bar(x, pts_values, width, color=colors, edgecolor='black', linewidth=1.2)

        # Add PTS values on bars
        for bar, val in zip(bars, pts_values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add SR delta as text below
        for i, (phase, sr) in enumerate(zip(phases, sr_delta_values)):
            if not np.isnan(sr):
                ax.text(i, -0.3, f'SR Δ: {sr:+.0f}', ha='center', va='top', fontsize=9)

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Pressure Effect')
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel('PTS', fontsize=11)
        ax.set_title(f'{player}\n{description}', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.5, max([p for p in pts_values if not np.isnan(p)]) + 1)

        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Resistant (< 1.0)'),
            Patch(facecolor='#f39c12', label='Normal (1.0-2.0)'),
            Patch(facecolor='#e74c3c', label='Vulnerable (> 2.0)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle('Player Pressure Profiles', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/04_player_profiles.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/04_player_profiles.png")
    plt.close()


# =============================================================================
# VISUALIZATION 5: Dismissal Rate Comparison
# =============================================================================
def plot_dismissal_rate_comparison(df):
    """Compare high vs low pressure dismissal rates"""

    fig, ax = plt.subplots(figsize=(12, 8))

    reliable = df[(df['high_pressure_innings'] >= 5) & (df['low_pressure_innings'] >= 5)]

    phases = ['PP', 'MO', 'Death']
    x = np.arange(len(phases))
    width = 0.35

    high_rates = []
    low_rates = []

    for phase in phases:
        phase_df = reliable[reliable['phase'] == phase]
        high_rates.append(phase_df['high_dismissal_rate'].mean() * 100)
        low_rates.append(phase_df['low_dismissal_rate'].mean() * 100)

    bars1 = ax.bar(x - width / 2, low_rates, width, label='Low Pressure',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width / 2, high_rates, width, label='High Pressure',
                   color='#e74c3c', edgecolor='black')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Match Phase', fontsize=12)
    ax.set_ylabel('Dismissal Rate (%)', fontsize=12)
    ax.set_title('Dismissal Rates: High vs Low Dot Density Situations',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(high_rates) + 15)

    # Add percentage increase annotations
    for i, (low, high) in enumerate(zip(low_rates, high_rates)):
        increase = ((high - low) / low) * 100
        ax.annotate(f'+{increase:.0f}%', xy=(i + width / 2, high + 5),
                    fontsize=10, color='red', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals05_dismissal_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/05_dismissal_rate_comparison.png")
    plt.close()


# =============================================================================
# VISUALIZATION 6: Pressure Response Categories
# =============================================================================
def plot_pressure_categories(df):
    """Pie charts showing distribution of pressure response types"""

    reliable = df[(df['high_pressure_innings'] >= 5) & (df['pts'] > 0)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    phases = ['PP', 'MO', 'Death']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for idx, phase in enumerate(phases):
        ax = axes[idx]
        phase_df = reliable[reliable['phase'] == phase]

        resistant = (phase_df['pts'] < 1.0).sum()
        normal = ((phase_df['pts'] >= 1.0) & (phase_df['pts'] <= 2.0)).sum()
        vulnerable = (phase_df['pts'] > 2.0).sum()

        sizes = [resistant, normal, vulnerable]
        labels = [f'Resistant\n({resistant})', f'Normal\n({normal})', f'Vulnerable\n({vulnerable})']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90,
                                          explode=(0.05, 0, 0.05))

        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(f'{phase}', fontsize=14, fontweight='bold')

    plt.suptitle('Player Distribution by Pressure Response Category',
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/06_pressure_categories.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/06_pressure_categories.png")
    plt.close()


# =============================================================================
# VISUALIZATION 7: Top 10 Most/Least Vulnerable (by phase)
# =============================================================================
def plot_top_vulnerable(df):
    """Horizontal bar charts of most and least vulnerable players"""

    reliable = df[(df['high_pressure_innings'] >= 5) &
                  (df['pts'] > 0) &
                  (df['total_innings'] >= 20)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    phases = ['PP', 'MO', 'Death']

    for idx, phase in enumerate(phases):
        phase_df = reliable[reliable['phase'] == phase].copy()

        # Top vulnerable (high PTS)
        ax_top = axes[0, idx]
        top_vulnerable = phase_df.nlargest(10, 'pts')

        colors_top = ['#e74c3c' if p > 2.5 else '#f39c12' for p in top_vulnerable['pts']]
        ax_top.barh(top_vulnerable['player'], top_vulnerable['pts'], color=colors_top, edgecolor='black')
        ax_top.set_xlabel('PTS', fontsize=11)
        ax_top.set_title(f'{phase}: Most Vulnerable', fontsize=12, fontweight='bold')
        ax_top.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        ax_top.invert_yaxis()

        # Top resistant (low PTS)
        ax_bottom = axes[1, idx]
        top_resistant = phase_df.nsmallest(10, 'pts')

        colors_bottom = ['#2ecc71' if p < 1.0 else '#f39c12' for p in top_resistant['pts']]
        ax_bottom.barh(top_resistant['player'], top_resistant['pts'], color=colors_bottom, edgecolor='black')
        ax_bottom.set_xlabel('PTS', fontsize=11)
        ax_bottom.set_title(f'{phase}: Most Resistant', fontsize=12, fontweight='bold')
        ax_bottom.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        ax_bottom.invert_yaxis()

    plt.suptitle('Top 10 Most Vulnerable vs Most Resistant Players\n(Min 20 innings, 5+ high-pressure situations)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/07_top_vulnerable_resistant.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/07_top_vulnerable_resistant.png")
    plt.close()


# =============================================================================
# VISUALIZATION 8: Summary Dashboard
# =============================================================================
def plot_summary_dashboard(df):
    """Single summary dashboard with key findings"""

    reliable = df[(df['high_pressure_innings'] >= 5) & (df['pts'] > 0) & (df['pts'] < 10)]

    fig = plt.figure(figsize=(20, 12))

    # Grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. PTS by Phase (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    phases = ['PP', 'MO', 'Death']
    data_to_plot = [reliable[reliable['phase'] == p]['pts'].values for p in phases]
    bp = ax1.boxplot(data_to_plot, labels=phases, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylabel('PTS')
    ax1.set_title('PTS Distribution by Phase', fontweight='bold')

    # 2. Average dismissal rate comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.35
    high_rates = [reliable[reliable['phase'] == p]['high_dismissal_rate'].mean() * 100 for p in phases]
    low_rates = [reliable[reliable['phase'] == p]['low_dismissal_rate'].mean() * 100 for p in phases]
    ax2.bar(x - width / 2, low_rates, width, label='Low Pressure', color='#3498db')
    ax2.bar(x + width / 2, high_rates, width, label='High Pressure', color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases)
    ax2.set_ylabel('Dismissal Rate (%)')
    ax2.set_title('Dismissal Rate Comparison', fontweight='bold')
    ax2.legend()

    # 3. Player category distribution
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['Resistant\n(PTS<1)', 'Normal\n(1-2)', 'Vulnerable\n(PTS>2)']
    counts = [
        (reliable['pts'] < 1.0).sum(),
        ((reliable['pts'] >= 1.0) & (reliable['pts'] <= 2.0)).sum(),
        (reliable['pts'] > 2.0).sum()
    ]
    ax3.bar(categories, counts, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax3.set_ylabel('Number of Player-Phase Records')
    ax3.set_title('Pressure Response Distribution', fontweight='bold')
    for i, v in enumerate(counts):
        ax3.text(i, v + 5, str(v), ha='center', fontweight='bold')

    # 4. SR Delta distribution
    ax4 = fig.add_subplot(gs[1, 0])
    for phase, color in zip(phases, colors):
        phase_df = reliable[reliable['phase'] == phase]
        ax4.hist(phase_df['sr_delta'], bins=30, alpha=0.5, label=phase, color=color)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Strike Rate Delta')
    ax4.set_ylabel('Frequency')
    ax4.set_title('SR Change Under Pressure', fontweight='bold')
    ax4.legend()

    # 5. Key stats text box
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')

    stats_text = """
    KEY FINDINGS
    ════════════════════════════════

    Average PTS by Phase:
    • Powerplay:    2.11 (2x more likely to get out)
    • Middle Overs: 1.73 
    • Death:        1.89

    Behavioral Response:
    • 99% of players slow down under pressure
    • Avg SR drop: -48 to -65 points
    • Death overs show biggest SR decline

    Player Distribution:
    • 13% Pressure Resistant (PTS < 1.0)
    • 52% Normal Response (PTS 1.0-2.0)
    • 35% Pressure Vulnerable (PTS > 2.0)

    Notable Findings:
    • Anchor batters (Kohli, Rahul) show
      4x+ vulnerability in Powerplay
    • Finishers (Maxwell, Raina) show
      resilience at Death (PTS < 1.0)
    """

    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. Top players comparison
    ax6 = fig.add_subplot(gs[1, 2])

    star_players = ['V Kohli', 'JC Buttler', 'MS Dhoni', 'GJ Maxwell']
    star_df = df[df['player'].isin(star_players)]

    for player in star_players:
        player_data = star_df[star_df['player'] == player]
        pts_vals = []
        for phase in phases:
            phase_row = player_data[player_data['phase'] == phase]
            if len(phase_row) > 0:
                pts_vals.append(phase_row['pts'].values[0])
            else:
                pts_vals.append(np.nan)
        ax6.plot(phases, pts_vals, marker='o', linewidth=2, markersize=8, label=player)

    ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax6.set_ylabel('PTS')
    ax6.set_title('Star Players Comparison', fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.set_ylim(0, 5)

    plt.suptitle('DOT BALL PRESSURE ANALYSIS: SUMMARY DASHBOARD',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('results/pressure_analysis_visuals/08_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/08_summary_dashboard.png")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("GENERATING PRESSURE ANALYSIS VISUALIZATIONS")
    print("=" * 70)

    df = load_data()
    print(f"\nLoaded {len(df):,} player-phase records")

    print("\nGenerating visualizations...\n")

    plot_pts_distribution(df)
    plot_sr_delta_vs_pts(df)
    plot_star_players_heatmap(df)
    plot_player_profiles(df)
    plot_dismissal_rate_comparison(df)
    plot_pressure_categories(df)
    plot_top_vulnerable(df)
    plot_summary_dashboard(df)

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  01_pts_distribution_by_phase.png")
    print("  02_sr_delta_vs_pts.png")
    print("  03_star_players_heatmap.png")
    print("  04_player_profiles.png")
    print("  05_dismissal_rate_comparison.png")
    print("  06_pressure_categories.png")
    print("  07_top_vulnerable_resistant.png")
    print("  08_summary_dashboard.png")


if __name__ == "__main__":
    main()