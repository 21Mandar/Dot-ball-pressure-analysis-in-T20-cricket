"""
Player Archetype Clustering - REFINED LABELS
Better archetype naming based on cluster characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

# Create output directory
Path("results/pressure_analysis_visuals").mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_and_prepare_data():
    """Load pressure metrics and prepare for clustering"""

    df = pd.read_csv('data/processed/player_pressure_metrics.csv')

    # Filter for reliable data
    reliable = df[
        (df['high_pressure_innings'] >= 5) &
        (df['low_pressure_innings'] >= 5) &
        (df['pts'] > 0) &
        (df['pts'] < 10) &
        (df['total_innings'] >= 15)
    ].copy()

    return reliable


def create_clustering_features(df):
    """Create feature matrix for clustering"""

    features = df[[
        'pts',
        'sr_delta',
        'boundary_delta',
        'overall_dot_density',
        'overall_sr'
    ]].copy()

    features = features.dropna()
    player_info = df.loc[features.index, ['player', 'phase', 'total_innings']].copy()

    return features, player_info


def assign_refined_archetype_labels(cluster_centers_scaled, cluster_centers_original, feature_names):
    """
    Assign meaningful, UNIQUE labels to clusters based on their characteristics
    Uses both scaled (for relative comparison) and original values (for interpretation)
    """

    centers_scaled = pd.DataFrame(cluster_centers_scaled, columns=feature_names)
    centers_original = pd.DataFrame(cluster_centers_original, columns=feature_names)

    # Calculate relative rankings for each feature
    pts_rank = centers_scaled['pts'].rank()
    sr_delta_rank = centers_scaled['sr_delta'].rank()  # Higher = less negative = better
    boundary_delta_rank = centers_scaled['boundary_delta'].rank()
    dot_density_rank = centers_scaled['overall_dot_density'].rank()
    overall_sr_rank = centers_scaled['overall_sr'].rank()

    labels = []
    used_labels = set()

    # Define all possible archetypes with priority scoring
    archetype_definitions = [
        {
            'name': 'Pressure Vulnerable',
            'color': '#e74c3c',
            'description': 'High dismissal risk under pressure, significant SR drop',
            'condition': lambda i: pts_rank[i] >= 4 and sr_delta_rank[i] <= 2
        },
        {
            'name': 'Composed Accumulator',
            'color': '#2ecc71',
            'description': 'Survives pressure well, absorbs dots, patient approach',
            'condition': lambda i: pts_rank[i] <= 2 and dot_density_rank[i] >= 4
        },
        {
            'name': 'Explosive Aggressor',
            'color': '#3498db',
            'description': 'High overall SR, attacks regardless of pressure',
            'condition': lambda i: overall_sr_rank[i] >= 4 and boundary_delta_rank[i] >= 3
        },
        {
            'name': 'Controlled Performer',
            'color': '#f39c12',
            'description': 'Balanced response, moderate across all metrics',
            'condition': lambda i: 2 <= pts_rank[i] <= 4 and 2 <= sr_delta_rank[i] <= 4
        },
        {
            'name': 'Pressure Collapser',
            'color': '#9b59b6',
            'description': 'Biggest SR drop under pressure, reduces boundaries significantly',
            'condition': lambda i: sr_delta_rank[i] == 1 and boundary_delta_rank[i] <= 2
        },
        {
            'name': 'Risk Taker',
            'color': '#1abc9c',
            'description': 'Increases aggression under pressure, high risk approach',
            'condition': lambda i: boundary_delta_rank[i] >= 4 or sr_delta_rank[i] >= 4
        },
        {
            'name': 'Steady Anchor',
            'color': '#34495e',
            'description': 'Consistent approach, neither excels nor struggles under pressure',
            'condition': lambda i: True  # Fallback
        }
    ]

    # Assign labels ensuring uniqueness
    for cluster_id in range(len(centers_scaled)):
        assigned = False

        for archetype in archetype_definitions:
            if archetype['name'] not in used_labels and archetype['condition'](cluster_id):
                labels.append({
                    'cluster': cluster_id,
                    'label': archetype['name'],
                    'color': archetype['color'],
                    'description': archetype['description'],
                    'pts_mean': centers_original['pts'][cluster_id],
                    'sr_delta_mean': centers_original['sr_delta'][cluster_id],
                    'boundary_delta_mean': centers_original['boundary_delta'][cluster_id],
                    'dot_density_mean': centers_original['overall_dot_density'][cluster_id],
                    'overall_sr_mean': centers_original['overall_sr'][cluster_id]
                })
                used_labels.add(archetype['name'])
                assigned = True
                break

        # Fallback if no condition matched
        if not assigned:
            fallback_name = f'Type {cluster_id + 1}'
            labels.append({
                'cluster': cluster_id,
                'label': fallback_name,
                'color': '#95a5a6',
                'description': 'Unique pressure response pattern',
                'pts_mean': centers_original['pts'][cluster_id],
                'sr_delta_mean': centers_original['sr_delta'][cluster_id],
                'boundary_delta_mean': centers_original['boundary_delta'][cluster_id],
                'dot_density_mean': centers_original['overall_dot_density'][cluster_id],
                'overall_sr_mean': centers_original['overall_sr'][cluster_id]
            })

    return pd.DataFrame(labels)


def perform_clustering(features, n_clusters=5):
    """Perform K-means clustering"""

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # Get cluster centers in original scale
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

    return clusters, features_scaled, features_pca, kmeans, scaler, pca, centers_original


def plot_cluster_scatter_refined(features_pca, clusters, player_info, cluster_labels):
    """2D scatter plot with refined labels"""

    fig, ax = plt.subplots(figsize=(14, 10))

    label_map = dict(zip(cluster_labels['cluster'], cluster_labels['label']))
    color_map = dict(zip(cluster_labels['cluster'], cluster_labels['color']))

    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id
        label = label_map.get(cluster_id, f'Cluster {cluster_id}')
        color = color_map.get(cluster_id, '#cccccc')

        ax.scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=80,
            edgecolors='white',
            linewidth=0.5
        )

    # Annotate notable players
    notable_players = [
        'V Kohli', 'JC Buttler', 'MS Dhoni', 'GJ Maxwell', 'AB de Villiers',
        'RG Sharma', 'KL Rahul', 'DA Warner', 'CH Gayle', 'SK Raina'
    ]

    player_info_reset = player_info.reset_index(drop=True)

    for i, row in player_info_reset.iterrows():
        if row['player'] in notable_players and i < len(features_pca):
            ax.annotate(
                f"{row['player']}\n({row['phase']})",
                (features_pca[i, 0], features_pca[i, 1]),
                fontsize=8,
                alpha=0.8,
                ha='center'
            )

    ax.set_xlabel('Principal Component 1\n(← Patient / Aggressive →)', fontsize=12)
    ax.set_ylabel('Principal Component 2\n(← Low Risk / High Risk →)', fontsize=12)
    ax.set_title('Player Pressure Response Archetypes\n(K-Means Clustering with PCA)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Archetype', loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/10_archetype_clusters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/10_archetype_clusters.png")
    plt.close()


def plot_cluster_profiles_refined(features, clusters, cluster_labels, feature_names):
    """Radar/bar chart with refined labels"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['cluster'] = clusters

    label_map = dict(zip(cluster_labels['cluster'], cluster_labels['label']))
    color_map = dict(zip(cluster_labels['cluster'], cluster_labels['color']))

    display_names = ['PTS', 'SR Delta', 'Boundary Δ', 'Dot Density', 'Overall SR']

    for cluster_id in range(min(5, len(cluster_labels))):
        ax = axes[cluster_id]
        cluster_data = features_df[features_df['cluster'] == cluster_id]

        if len(cluster_data) == 0:
            continue

        means = cluster_data[feature_names].mean()
        color = color_map.get(cluster_id, '#cccccc')

        bars = ax.bar(range(len(feature_names)), means.values, color=color,
                      edgecolor='black', alpha=0.7)

        # Add value labels on bars
        for bar, val in zip(bars, means.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean Value (Scaled)', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

        label = label_map.get(cluster_id, f'Cluster {cluster_id}')
        count = len(cluster_data)
        ax.set_title(f'{label}\n(n={count})', fontsize=12, fontweight='bold', color=color)

    # Legend/explanation in last subplot
    axes[5].axis('off')

    explanation_text = """
    ARCHETYPE DEFINITIONS
    ═══════════════════════════════════════
    
    🔴 Pressure Vulnerable
       High dismissal risk, big SR drops
       
    🟢 Composed Accumulator  
       Survives well, patient approach
       
    🔵 Explosive Aggressor
       High SR, attacks under pressure
       
    🟡 Controlled Performer
       Balanced, moderate response
       
    🟣 Pressure Collapser
       Biggest SR/boundary drops
       
    ═══════════════════════════════════════
    
    Features (Scaled):
    • PTS: Pressure Tolerance Score
    • SR Delta: Strike rate change
    • Boundary Δ: Boundary % change  
    • Dot Density: Natural dot tendency
    • Overall SR: Base strike rate
    """

    axes[5].text(0.05, 0.95, explanation_text, transform=axes[5].transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Cluster Profiles: Mean Feature Values by Archetype',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/11_cluster_profiles.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/11_cluster_profiles.png")
    plt.close()


def plot_archetype_by_phase_refined(df_with_clusters, cluster_labels):
    """Archetype distribution by phase with refined labels"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    label_map = dict(zip(cluster_labels['cluster'], cluster_labels['label']))
    color_map = dict(zip(cluster_labels['cluster'], cluster_labels['color']))

    phases = ['PP', 'MO', 'Death']

    for idx, phase in enumerate(phases):
        ax = axes[idx]
        phase_df = df_with_clusters[df_with_clusters['phase'] == phase]

        if len(phase_df) == 0:
            continue

        cluster_counts = phase_df['cluster'].value_counts().sort_index()

        labels = [label_map.get(c, f'Cluster {c}') for c in cluster_counts.index]
        colors = [color_map.get(c, '#cccccc') for c in cluster_counts.index]

        wedges, texts, autotexts = ax.pie(
            cluster_counts.values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.02] * len(cluster_counts)
        )

        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        ax.set_title(f'{phase}\n(n={len(phase_df)})', fontsize=14, fontweight='bold')

    plt.suptitle('Archetype Distribution by Match Phase', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/12_archetype_by_phase.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/12_archetype_by_phase.png")
    plt.close()


def plot_star_players_archetypes_refined(df_with_clusters, cluster_labels):
    """Star players archetype grid with refined labels"""

    star_players = [
        'V Kohli', 'RG Sharma', 'AB de Villiers', 'MS Dhoni', 'JC Buttler',
        'KL Rahul', 'DA Warner', 'CH Gayle', 'SK Raina', 'S Dhawan',
        'HH Pandya', 'GJ Maxwell', 'KS Williamson', 'Babar Azam'
    ]

    label_map = dict(zip(cluster_labels['cluster'], cluster_labels['label']))
    color_map = dict(zip(cluster_labels['cluster'], cluster_labels['color']))

    stars_df = df_with_clusters[df_with_clusters['player'].isin(star_players)].copy()
    stars_df['archetype'] = stars_df['cluster'].map(label_map)
    stars_df['arch_color'] = stars_df['cluster'].map(color_map)

    # Pivot for visualization
    pivot = stars_df.pivot_table(
        index='player',
        columns='phase',
        values='archetype',
        aggfunc='first'
    )
    pivot = pivot.reindex(columns=['PP', 'MO', 'Death'])

    pivot_colors = stars_df.pivot_table(
        index='player',
        columns='phase',
        values='arch_color',
        aggfunc='first'
    )
    pivot_colors = pivot_colors.reindex(columns=['PP', 'MO', 'Death'])

    # Sort by player name
    pivot = pivot.sort_index()
    pivot_colors = pivot_colors.reindex(pivot.index)

    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = range(len(pivot))
    x_pos = range(len(pivot.columns))

    for i, player in enumerate(pivot.index):
        for j, phase in enumerate(pivot.columns):
            archetype = pivot.loc[player, phase]
            if pd.notna(archetype):
                color = pivot_colors.loc[player, phase] if pd.notna(pivot_colors.loc[player, phase]) else '#cccccc'
                rect = plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                                      facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.add_patch(rect)

                # Shorten label for display
                short_label = archetype.split()[0][:8]
                ax.text(j, i, short_label, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white' if color in ['#e74c3c', '#9b59b6', '#34495e'] else 'black')

    ax.set_xlim(-0.5, len(pivot.columns) - 0.5)
    ax.set_ylim(-0.5, len(pivot) - 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Powerplay', 'Middle Overs', 'Death'], fontsize=12, fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel('Match Phase', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    ax.set_title('Star Players: Pressure Response Archetype by Phase',
                 fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    unique_archetypes = cluster_labels[['label', 'color']].drop_duplicates()
    legend_elements = [Patch(facecolor=row['color'], edgecolor='black', label=row['label'])
                       for _, row in unique_archetypes.iterrows()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              title='Archetype', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/13_star_players_archetypes.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/13_star_players_archetypes.png")
    plt.close()

    return stars_df[['player', 'phase', 'archetype', 'pts', 'sr_delta']]


def plot_archetype_summary_table(cluster_labels):
    """Create a summary table visualization of archetypes"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Create table data
    table_data = []
    for _, row in cluster_labels.iterrows():
        table_data.append([
            row['label'],
            f"{row['pts_mean']:.2f}",
            f"{row['sr_delta_mean']:.1f}",
            f"{row['boundary_delta_mean']:.3f}",
            f"{row['overall_sr_mean']:.1f}",
            row['description']
        ])

    columns = ['Archetype', 'Avg PTS', 'Avg SR Δ', 'Avg Bndry Δ', 'Avg SR', 'Description']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.08, 0.08, 0.1, 0.08, 0.35]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color archetype cells
    for i, row in cluster_labels.iterrows():
        table[(i + 1, 0)].set_facecolor(row['color'])
        table[(i + 1, 0)].set_text_props(color='white' if row['color'] in ['#e74c3c', '#9b59b6', '#34495e'] else 'black',
                                          fontweight='bold')

    ax.set_title('Archetype Definitions and Characteristics', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('results/pressure_analysis_visuals/14_archetype_summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/pressure_analysis_visuals/14_archetype_summary_table.png")
    plt.close()


def main():
    print("=" * 70)
    print("PLAYER ARCHETYPE CLUSTERING - REFINED LABELS")
    print("=" * 70)

    # Load data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data()
    print(f"  Loaded {len(df)} reliable player-phase records")

    # Create features
    print("\nCreating clustering features...")
    features, player_info = create_clustering_features(df)
    feature_names = ['pts', 'sr_delta', 'boundary_delta', 'overall_dot_density', 'overall_sr']
    print(f"  Feature matrix shape: {features.shape}")

    # Perform clustering
    n_clusters = 5
    print(f"\nPerforming K-means clustering (K={n_clusters})...")
    clusters, features_scaled, features_pca, kmeans, scaler, pca, centers_original = perform_clustering(features, n_clusters)

    # Assign refined labels
    print("\nAssigning refined archetype labels...")
    cluster_labels = assign_refined_archetype_labels(
        kmeans.cluster_centers_,
        centers_original,
        feature_names
    )

    print("\nArchetype Definitions:")
    print("-" * 70)
    for _, row in cluster_labels.iterrows():
        print(f"  {row['label']:22} | PTS: {row['pts_mean']:5.2f} | SR Δ: {row['sr_delta_mean']:+6.1f} | {row['description']}")

    # Add clusters to dataframe
    df_clustered = df.loc[features.index].copy()
    df_clustered['cluster'] = clusters

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_cluster_scatter_refined(features_pca, clusters, player_info, cluster_labels)
    plot_cluster_profiles_refined(features_scaled, clusters, cluster_labels, feature_names)
    plot_archetype_by_phase_refined(df_clustered, cluster_labels)
    star_archetypes = plot_star_players_archetypes_refined(df_clustered, cluster_labels)
    plot_archetype_summary_table(cluster_labels)

    # Save clustered data
    label_map = dict(zip(cluster_labels['cluster'], cluster_labels['label']))
    color_map = dict(zip(cluster_labels['cluster'], cluster_labels['color']))
    df_clustered['archetype'] = df_clustered['cluster'].map(label_map)
    df_clustered['archetype_color'] = df_clustered['cluster'].map(color_map)
    df_clustered.to_csv('data/processed/player_archetypes.csv', index=False)
    print("\n✓ Saved: data/processed/player_archetypes.csv")

    # Print summary
    print("\n" + "=" * 70)
    print("ARCHETYPE DISTRIBUTION SUMMARY")
    print("=" * 70)

    print("\nOverall distribution:")
    print(df_clustered['archetype'].value_counts().to_string())

    print("\nBy phase:")
    phase_arch = df_clustered.groupby(['phase', 'archetype']).size().unstack(fill_value=0)
    print(phase_arch.to_string())

    print("\n" + "=" * 70)
    print("STAR PLAYERS ARCHETYPES")
    print("=" * 70)
    print(star_archetypes.sort_values(['player', 'phase']).to_string(index=False))

    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  10_archetype_clusters.png")
    print("  11_cluster_profiles.png")
    print("  12_archetype_by_phase.png")
    print("  13_star_players_archetypes.png")
    print("  14_archetype_summary_table.png (NEW)")
    print("  data/processed/player_archetypes.csv")


if __name__ == "__main__":
    main()