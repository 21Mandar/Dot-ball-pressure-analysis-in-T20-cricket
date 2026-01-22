"""
Interactive Player Pressure Profile Lookup Tool
Streamlit-based dashboard for exploring player pressure response data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Cricket Pressure Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .archetype-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        metrics_df = pd.read_csv('data/processed/player_pressure_metrics.csv')
        archetypes_df = pd.read_csv('data/processed/player_archetypes.csv')
        phase_innings_df = pd.read_csv('data/processed/player_phase_innings.csv')
        return metrics_df, archetypes_df, phase_innings_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None


def get_archetype_color(archetype):
    """Return color for each archetype"""
    colors = {
        'Steady Anchor': '#34495e',
        'Composed Accumulator': '#2ecc71',
        'Pressure Collapser': '#9b59b6',
        'Explosive Aggressor': '#3498db',
        'Pressure Vulnerable': '#e74c3c',
        'Controlled Performer': '#f39c12',
        'Risk Taker': '#1abc9c'
    }
    return colors.get(archetype, '#95a5a6')


def get_pts_category(pts):
    """Categorize PTS value"""
    if pts < 1.0:
        return "Pressure Resistant", "#2ecc71"
    elif pts <= 2.0:
        return "Normal Response", "#f39c12"
    else:
        return "Pressure Vulnerable", "#e74c3c"


def create_pts_gauge(pts, phase):
    """Create a gauge chart for PTS"""
    category, color = get_pts_category(pts)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pts,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': phase, 'font': {'size': 16, 'color': '#333'}},
        number={'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "#666"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 1], 'color': '#d5f5e3'},
                {'range': [1, 2], 'color': '#fef9e7'},
                {'range': [2, 5], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': pts
            }
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333"}
    )

    return fig


def create_radar_chart(player_data):
    """Create radar chart for player metrics"""
    phases = ['PP', 'MO', 'Death']

    # Prepare data
    pts_values = []
    sr_delta_values = []

    for phase in phases:
        phase_row = player_data[player_data['phase'] == phase]
        if len(phase_row) > 0:
            pts_values.append(phase_row['pts'].values[0])
            # Normalize SR delta (make positive for visualization)
            sr_delta_values.append(abs(phase_row['sr_delta'].values[0]))
        else:
            pts_values.append(0)
            sr_delta_values.append(0)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=pts_values + [pts_values[0]],
        theta=phases + [phases[0]],
        fill='toself',
        name='PTS',
        line_color='#e74c3c',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(pts_values) * 1.2 if max(pts_values) > 0 else 5]
            )
        ),
        showlegend=True,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_comparison_bar(player_data, all_data):
    """Create bar chart comparing player to average"""
    phases = ['PP', 'MO', 'Death']

    player_pts = []
    avg_pts = []

    for phase in phases:
        # Player value
        phase_row = player_data[player_data['phase'] == phase]
        if len(phase_row) > 0:
            player_pts.append(phase_row['pts'].values[0])
        else:
            player_pts.append(0)

        # Average value
        phase_avg = all_data[all_data['phase'] == phase]['pts'].mean()
        avg_pts.append(phase_avg)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Player',
        x=phases,
        y=player_pts,
        marker_color='#3498db'
    ))

    fig.add_trace(go.Bar(
        name='League Average',
        x=phases,
        y=avg_pts,
        marker_color='#bdc3c7'
    ))

    fig.update_layout(
        barmode='group',
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        yaxis_title='PTS',
        xaxis_title='Phase'
    )

    # Add reference line at PTS = 1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="No Pressure Effect")

    return fig


def display_archetype_badge(archetype):
    """Display archetype as a colored badge"""
    color = get_archetype_color(archetype)

    # Determine text color based on background
    dark_backgrounds = ['#34495e', '#9b59b6', '#e74c3c']
    text_color = 'white' if color in dark_backgrounds else 'black'

    st.markdown(
        f"""<span style="background-color: {color}; color: {text_color}; 
        padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; 
        font-size: 1rem;">{archetype}</span>""",
        unsafe_allow_html=True
    )


def main():
    # Header
    st.markdown('<p class="main-header">🏏 Cricket Pressure Analysis Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore how players respond to dot ball pressure across match phases</p>',
                unsafe_allow_html=True)

    # Load data
    metrics_df, archetypes_df, phase_innings_df = load_data()

    if metrics_df is None:
        st.error("Unable to load data. Please ensure all data files are in the correct location.")
        st.stop()

    # Sidebar
    st.sidebar.title("🔍 Player Search")

    # Get list of players with sufficient data
    players_with_data = metrics_df[metrics_df['total_innings'] >= 10]['player'].unique()
    players_sorted = sorted(players_with_data)

    # Search functionality
    search_term = st.sidebar.text_input("Search player name:", "")

    if search_term:
        filtered_players = [p for p in players_sorted if search_term.lower() in p.lower()]
    else:
        filtered_players = players_sorted

    if len(filtered_players) == 0:
        st.sidebar.warning("No players found matching your search.")
        selected_player = None
    else:
        selected_player = st.sidebar.selectbox(
            "Select a player:",
            filtered_players,
            index=0 if len(filtered_players) > 0 else None
        )

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About PTS")
    st.sidebar.markdown("""
    **Pressure Tolerance Score (PTS)** measures how a player's dismissal probability changes under dot ball pressure.

    - **PTS < 1.0**: Pressure Resistant
    - **PTS 1.0-2.0**: Normal Response  
    - **PTS > 2.0**: Pressure Vulnerable
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Archetypes")
    st.sidebar.markdown("""
    - 🟢 **Composed Accumulator**: Patient, survives pressure
    - 🔵 **Explosive Aggressor**: High SR, attacks under pressure
    - 🟣 **Pressure Collapser**: Big SR drop under pressure
    - 🔴 **Pressure Vulnerable**: High dismissal risk
    - ⚫ **Steady Anchor**: Consistent but struggles under pressure
    """)

    # Main content
    if selected_player:
        # Get player data
        player_metrics = metrics_df[metrics_df['player'] == selected_player]
        player_archetypes = archetypes_df[
            archetypes_df['player'] == selected_player] if archetypes_df is not None else None

        if len(player_metrics) == 0:
            st.warning(f"No detailed metrics available for {selected_player}")
            st.stop()

        # Player header
        st.markdown(f"## {selected_player}")

        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)

        total_innings = player_metrics['total_innings'].sum()
        avg_pts = player_metrics['pts'].mean()
        avg_sr_delta = player_metrics['sr_delta'].mean()
        phases_available = len(player_metrics)

        with col1:
            st.metric("Total Phase-Innings", f"{total_innings}")
        with col2:
            st.metric("Average PTS", f"{avg_pts:.2f}")
        with col3:
            st.metric("Avg SR Delta", f"{avg_sr_delta:+.1f}")
        with col4:
            st.metric("Phases Analyzed", f"{phases_available}/3")

        st.markdown("---")

        # Phase-wise breakdown
        st.markdown("###  Phase-wise Pressure Response")

        phase_cols = st.columns(3)
        phases = ['PP', 'MO', 'Death']
        phase_names = {'PP': 'Powerplay', 'MO': 'Middle Overs', 'Death': 'Death Overs'}

        for idx, phase in enumerate(phases):
            with phase_cols[idx]:
                phase_data = player_metrics[player_metrics['phase'] == phase]

                if len(phase_data) > 0:
                    pts = phase_data['pts'].values[0]
                    sr_delta = phase_data['sr_delta'].values[0]
                    innings = phase_data['total_innings'].values[0]
                    high_dism = phase_data['high_dismissal_rate'].values[0] * 100
                    low_dism = phase_data['low_dismissal_rate'].values[0] * 100

                    # Get archetype for this phase
                    if player_archetypes is not None:
                        arch_data = player_archetypes[player_archetypes['phase'] == phase]
                        if len(arch_data) > 0 and 'archetype' in arch_data.columns:
                            archetype = arch_data['archetype'].values[0]
                        else:
                            archetype = "N/A"
                    else:
                        archetype = "N/A"

                    st.markdown(f"#### {phase_names[phase]}")

                    # PTS Gauge
                    fig = create_pts_gauge(pts, "")
                    st.plotly_chart(fig, use_container_width=True)

                    # Category
                    category, cat_color = get_pts_category(pts)
                    st.markdown(f"**Status:** <span style='color:{cat_color}; font-weight:bold;'>{category}</span>",
                                unsafe_allow_html=True)

                    # Archetype badge
                    if archetype != "N/A":
                        st.markdown("**Archetype:**")
                        display_archetype_badge(archetype)

                    # Metrics
                    st.markdown(f"""
                    - **Innings:** {innings}
                    - **SR Delta:** {sr_delta:+.1f}
                    - **High Pressure Dism:** {high_dism:.1f}%
                    - **Low Pressure Dism:** {low_dism:.1f}%
                    """)
                else:
                    st.markdown(f"#### {phase_names[phase]}")
                    st.info("Insufficient data for this phase")

        st.markdown("---")

        # Comparison charts
        st.markdown("### Visual Statistics")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### PTS vs League Average")
            comparison_fig = create_comparison_bar(player_metrics, metrics_df)
            st.plotly_chart(comparison_fig, use_container_width=True)

        with viz_col2:
            st.markdown("#### Pressure Profile Radar")
            radar_fig = create_radar_chart(player_metrics)
            st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown("---")

        # Detailed metrics table
        st.markdown("### Metrics")

        display_cols = ['phase', 'pts', 'sr_delta', 'boundary_delta',
                        'high_dismissal_rate', 'low_dismissal_rate',
                        'total_innings', 'high_pressure_innings']

        display_df = player_metrics[display_cols].copy()
        display_df.columns = ['Phase', 'PTS', 'SR Delta', 'Boundary Delta',
                              'High Pressure Dism Rate', 'Low Pressure Dism Rate',
                              'Total Innings', 'High Pressure Innings']

        # Format percentages
        display_df['High Pressure Dism Rate'] = (display_df['High Pressure Dism Rate'] * 100).round(1).astype(str) + '%'
        display_df['Low Pressure Dism Rate'] = (display_df['Low Pressure Dism Rate'] * 100).round(1).astype(str) + '%'
        display_df['PTS'] = display_df['PTS'].round(2)
        display_df['SR Delta'] = display_df['SR Delta'].round(1)
        display_df['Boundary Delta'] = display_df['Boundary Delta'].round(3)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("---")
        st.markdown("### Insight")

        # Generate automatic interpretation
        interpretations = []

        for _, row in player_metrics.iterrows():
            phase = row['phase']
            pts = row['pts']
            sr_delta = row['sr_delta']

            phase_name = phase_names.get(phase, phase)

            if pts < 1.0:
                interpretations.append(
                    f"✅ **{phase_name}**: Excellent pressure resistance (PTS: {pts:.2f}). This player is *less* likely to get out when dot pressure builds.")
            elif pts <= 2.0:
                interpretations.append(
                    f"⚠️ **{phase_name}**: Normal pressure response (PTS: {pts:.2f}). Dismissal risk increases moderately under pressure.")
            else:
                interpretations.append(
                    f"🔴 **{phase_name}**: High pressure vulnerability (PTS: {pts:.2f}). This player is {pts:.1f}x more likely to get out when facing dot pressure.")

            if sr_delta < -60:
                interpretations.append(f"   - Significant scoring slowdown under pressure (SR drop: {sr_delta:.0f})")

        for interp in interpretations:
            st.markdown(interp)

    else:
        # No player selected - show overview
        st.markdown("### 👈 Select a player from the sidebar to begin")

        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")

        overview_col1, overview_col2, overview_col3 = st.columns(3)

        with overview_col1:
            st.metric("Total Players", f"{metrics_df['player'].nunique():,}")
        with overview_col2:
            st.metric("Total Records", f"{len(metrics_df):,}")
        with overview_col3:
            st.metric("Avg PTS", f"{metrics_df['pts'].mean():.2f}")

        # Show top pressure resistant and vulnerable players
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Most Pressure Resistant (Low PTS)")
            reliable = metrics_df[(metrics_df['high_pressure_innings'] >= 5) & (metrics_df['pts'] > 0)]
            top_resistant = reliable.nsmallest(10, 'pts')[['player', 'phase', 'pts', 'total_innings']]
            st.dataframe(top_resistant, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### ⚠️ Most Pressure Vulnerable (High PTS)")
            top_vulnerable = reliable.nlargest(10, 'pts')[['player', 'phase', 'pts', 'total_innings']]
            st.dataframe(top_vulnerable, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()