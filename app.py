import io
import re
from pathlib import Path
from typing import List, Optional, Set

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D


DATA_PATH = Path(__file__).resolve().parent / "data" / "working_dataset.csv"
TOP_LEAGUES = ["Bundesliga", "La Liga", "Premier League", "Ligue 1", "Serie A"]
LOGO_PATHS = [
    Path(__file__).resolve().parent / "assests" / "2023_scoreadvise_transparent.png",
    Path(__file__).resolve().parent / "assets" / "2023_scoreadvise_transparent.png",
]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def metric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    skip_cols = {"ID", "Age", "Born", "Matches Played", "Starts", "Min", "90s", "Shots on Target %", "Goals / Shot", "Goals / Shot on Target",
                 "Average Shot Distance", "Pass Compl. Ration", "Tkl%"}
    return [col for col in numeric_cols if col not in skip_cols]


def parse_positions(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [pos.strip() for pos in str(value).split(",") if pos.strip()]


def split_positions(series: pd.Series) -> List[str]:
    positions: Set[str] = set()
    for value in series.dropna().unique():
        positions.update(parse_positions(value))
    return sorted(positions)


def positions_for_player(df: pd.DataFrame, player_name: str) -> List[str]:
    positions: Set[str] = set()
    for value in df.loc[df["Player"] == player_name, "Position"].dropna().unique():
        positions.update(parse_positions(value))
    return sorted(positions)


def filter_by_positions(df: pd.DataFrame, selected_positions: List[str]) -> pd.DataFrame:
    if not selected_positions:
        return df
    return df[
        df["Position"].apply(
            lambda value: any(pos in parse_positions(value) for pos in selected_positions)
        )
    ]


def slugify(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    return clean.strip("_") or "plot"


def plot_scatter_near(
    df: pd.DataFrame,
    player_name: str,
    x_col: str,
    y_col: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    top_n: int = 20,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    background = "black"
    text_color = "white"
    score_green = "#d0ff01"
    outlier_x_color = "#ff4d4d"
    outlier_y_color = "#00bfff"

    ax_scatter = fig.add_subplot(111)
    fig.patch.set_facecolor(background)
    ax_scatter.patch.set_facecolor(background)

    df_fill = df[df["90s"] > 0].copy()
    if df_fill.empty:
        raise ValueError("No data after filtering.")

    df_fill["_x_per90"] = df_fill[x_col] / df_fill["90s"]
    df_fill["_y_per90"] = df_fill[y_col] / df_fill["90s"]

    ax_scatter.scatter(
        df_fill["_x_per90"],
        df_fill["_y_per90"],
        alpha=0.45,
        c="white",
        s=30,
        zorder=1,
    )

    df_player = df_fill[df_fill["Player"] == player_name]
    if df_player.empty:
        raise ValueError(f"Player '{player_name}' not found in the filtered data.")

    player_x = df_player["_x_per90"].iloc[0]
    player_y = df_player["_y_per90"].iloc[0]
    top_x = df_fill.nlargest(top_n, "_x_per90")
    top_y = df_fill.nlargest(top_n, "_y_per90")
    x_top_min = float(top_x["_x_per90"].min()) if not top_x.empty else None
    y_top_min = float(top_y["_y_per90"].min()) if not top_y.empty else None

    plotted_players: Set[str] = set()

    def label_text(row: pd.Series) -> str:
        last_name = row.get("Last Name")
        if isinstance(last_name, str) and last_name:
            return last_name
        player = row.get("Player")
        if isinstance(player, str) and player:
            return player.split()[-1]
        return ""

    for _, row in top_x.iterrows():
        name = row["Player"]
        if name in plotted_players or name == player_name:
            continue
        xv, yv = row["_x_per90"], row["_y_per90"]
        ax_scatter.scatter(
            xv,
            yv,
            s=120,
            edgecolors="black",
            linewidth=1,
            c=outlier_x_color,
            marker="o",
            zorder=5,
            alpha=0.95,
        )
        ax_scatter.annotate(
            label_text(row),
            (xv, yv),
            textcoords="offset points",
            xytext=(6, 4),
            ha="left",
            fontsize=9,
            color=outlier_x_color,
            weight="bold",
        )
        plotted_players.add(name)

    ax_scatter.scatter(
        player_x,
        player_y,
        c=score_green,
        s=220,
        edgecolors="black",
        linewidth=1.4,
        zorder=10,
    )
    ax_scatter.annotate(
        player_name,
        (player_x, player_y),
        textcoords="offset points",
        xytext=(10, 8),
        ha="left",
        fontsize=11,
        color=score_green,
        weight="bold",
        zorder=10,
    )

    for _, row in top_y.iterrows():
        name = row["Player"]
        if name in plotted_players or name == player_name:
            continue
        xv, yv = row["_x_per90"], row["_y_per90"]
        ax_scatter.scatter(
            xv,
            yv,
            s=120,
            edgecolors="black",
            linewidth=1,
            c=outlier_y_color,
            marker="o",
            zorder=4,
            alpha=0.95,
        )
        ax_scatter.annotate(
            label_text(row),
            (xv, yv),
            textcoords="offset points",
            xytext=(6, -6),
            ha="left",
            fontsize=9,
            color=outlier_y_color,
            weight="bold",
        )
        plotted_players.add(name)

    x_label = x_label or x_col
    y_label = y_label or y_col
    ax_scatter.set_xlabel(f"{x_label} per 90", fontsize=12, weight="bold", color=text_color)
    ax_scatter.set_ylabel(f"{y_label} per 90", fontsize=12, weight="bold", color=text_color)

    avg_x = df_fill[x_col].sum() / df_fill["90s"].sum()
    avg_y = df_fill[y_col].sum() / df_fill["90s"].sum()
    ax_scatter.axvline(avg_x, color="white", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
    ax_scatter.axhline(avg_y, color="white", linestyle="--", linewidth=1, alpha=0.6, zorder=2)

    if x_top_min is not None:
        ax_scatter.axvline(
            x_top_min,
            color=outlier_x_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            zorder=2,
        )
    if y_top_min is not None:
        ax_scatter.axhline(
            y_top_min,
            color=outlier_y_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            zorder=2,
        )

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Top {top_n} in {x_label} / 90",
            markerfacecolor=outlier_x_color,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Top {top_n} in {y_label} / 90",
            markerfacecolor=outlier_y_color,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=player_name,
            markerfacecolor=score_green,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=8,
        ),
        Line2D([0], [0], color="white", lw=1, ls="--", label="League average"),
    ]
    ax_scatter.legend(
        handles=legend_elems,
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor="white",
    )

    ax_scatter.grid(visible=True, color="white", linestyle="-.", linewidth=0.5, alpha=0.25)
    ax_scatter.tick_params(axis="both", colors=text_color)

    fig.subplots_adjust(bottom=0.12)
    fig.text(
        0.5,
        0.02,
        "created by scoreadvise | https://x.com/scoreadviseYT",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        alpha=0.5,
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="Per-90 Scatter Explorer", layout="wide")
    st.markdown(
        """
        <style>
        :root {
          color-scheme: dark;
          --score-green: #d0ff01;
          --primary-color: #d0ff01;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
          background-color: #000;
          color: #fff;
        }
        header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
          background-color: #000;
        }
        section.main, [data-testid="stAppViewContainer"] > .main {
          background-color: #000;
        }
        [data-testid="stSidebar"] {
          background-color: #000;
          color: #fff;
          border-right: 1px solid #222;
        }
        [data-testid="stMarkdownContainer"] {
          color: #fff;
        }
        h1, h2, h3, h4, h5 {
          color: var(--score-green);
        }
        input, textarea, select {
          color: #fff !important;
          background-color: #111 !important;
        }
        [data-baseweb="select"] > div {
          background-color: #111 !important;
          color: #fff !important;
        }
        [data-baseweb="tag"] {
          background-color: #111 !important;
          color: #fff !important;
        }
        [data-baseweb="slider"] .rc-slider-track {
          background-color: var(--score-green);
        }
        [data-baseweb="slider"] .rc-slider-rail {
          background-color: #222;
        }
        [data-baseweb="slider"] .rc-slider-handle {
          border-color: var(--score-green);
          background-color: #000;
        }
        .stSlider [data-baseweb="slider"] {
          color: var(--score-green);
        }
        .stSlider [data-testid="stSliderValue"],
        .stSlider [data-testid="stSliderThumbValue"] {
          color: var(--score-green) !important;
        }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {
          color: #fff;
        }
        [data-testid="stDownloadButton"] > button {
          background-color: #000 !important;
          color: #fff !important;
          border: 1px solid #fff !important;
          min-height: 2.5rem;
          padding: 0.25rem 0.70rem;
          line-height: 1.1;
          width: 100%;
        }
        [data-testid="stDownloadButton"] > button:hover {
          background-color: #111 !important;
          color: #fff !important;
          border-color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    header_left, header_right = st.columns([6, 1])
    with header_left:
        st.title("Per-90 Scatter Explorer")
    with header_right:
        logo_path = next((path for path in LOGO_PATHS if path.exists()), None)
        if logo_path:
            st.image(str(logo_path), use_column_width=True)

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    df = load_data(DATA_PATH)
    seasons = sorted(df["Season"].dropna().unique().tolist())
    leagues = sorted(df["League"].dropna().unique().tolist())
    metrics = metric_columns(df)

    default_season = seasons[-1] if seasons else None

    with st.sidebar:
        st.header("Filters")
        selected_seasons = st.multiselect(
            "Season",
            seasons,
            default=[default_season] if default_season else [],
        )
        default_leagues = [league for league in TOP_LEAGUES if league in leagues]
        if not default_leagues:
            default_leagues = leagues
        selected_leagues = st.multiselect("League", leagues, default=default_leagues)
        min_90s = st.slider("Minimum 90s", 0.0, 30.0, 5.0, 0.5)

    df_base = df.copy()
    if selected_seasons:
        df_base = df_base[df_base["Season"].isin(selected_seasons)]
    if selected_leagues:
        df_base = df_base[df_base["League"].isin(selected_leagues)]
    df_base = df_base[df_base["90s"] >= min_90s]

    if df_base.empty:
        st.warning("No players match the current filters.")
        return

    player_list = sorted(df_base["Player"].dropna().unique().tolist())
    if not player_list:
        st.warning("No players available for selection.")
        return

    position_options = split_positions(df_base["Position"])
    if not position_options:
        st.warning("No positions available for selection.")
        return

    position_map = {player: positions_for_player(df_base, player) for player in player_list}
    st.session_state["position_map"] = position_map

    if "selected_player" not in st.session_state or st.session_state.selected_player not in player_list:
        st.session_state.selected_player = player_list[0]

    default_positions = position_map.get(st.session_state.selected_player, position_options)
    if "selected_positions" not in st.session_state:
        st.session_state.selected_positions = default_positions
    else:
        valid_positions = [
            pos for pos in st.session_state.selected_positions if pos in position_options
        ]
        if not valid_positions:
            st.session_state.selected_positions = default_positions
        else:
            st.session_state.selected_positions = valid_positions

    def sync_positions() -> None:
        player = st.session_state.get("selected_player")
        new_positions = st.session_state.get("position_map", {}).get(player, [])
        st.session_state.selected_positions = new_positions

    st.markdown("Highlight player")
    player_col, download_col = st.columns([4, 1])
    with player_col:
        player_name = st.selectbox(
            "Highlight player",
            player_list,
            key="selected_player",
            on_change=sync_positions,
            label_visibility="collapsed",
        )
    download_placeholder = download_col.empty()

    with st.sidebar:
        selected_positions = st.multiselect(
            "Position",
            position_options,
            key="selected_positions",
        )

        st.header("Scatter settings")
        x_default = metrics.index("Touches") if "Touches" in metrics else 0
        y_default = metrics.index("Prog. Carries") if "Prog. Carries" in metrics else 1
        x_col = st.selectbox("X-axis metric", metrics, index=x_default)
        y_col = st.selectbox("Y-axis metric", metrics, index=y_default)
        top_n = st.slider("Top-N outliers", 5, 50, 20, 1)

    df_filtered = filter_by_positions(df_base, selected_positions)

    if df_filtered.empty:
        st.warning("No players match the current filters.")
        return

    if player_name not in df_filtered["Player"].values:
        st.warning("Selected player is not included with the current position filter.")
        return

    try:
        fig = plot_scatter_near(
            df_filtered,
            player_name,
            x_col,
            y_col,
            x_label=x_col,
            y_label=y_col,
            top_n=top_n,
        )
        st.pyplot(fig, use_container_width=True)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=200,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        buffer.seek(0)
        filename = f"{slugify(player_name)}_{slugify(x_col)}_vs_{slugify(y_col)}.png"
        download_placeholder.download_button(
            "Download PNG",
            buffer,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )
    except ValueError as exc:
        st.warning(str(exc))

if __name__ == "__main__":
    main()
