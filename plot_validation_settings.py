"""Compare model settings on the validation years."""

import argparse
from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def setting_label(model, settings):
    if model == "XGBoost":
        return f"Depth {settings['max_depth']} / {settings['n_estimators']:,} trees"
    if model == "Ridge":
        return f"Penalty {settings['alpha']:,}"
    return f"{settings['hidden_size']} units / {settings['epochs']} epochs"


def plot_validation_settings(validation, destination):
    """Plot mean validation MAE from the notebook table or its CSV export."""
    scores = (
        validation.assign(settings=validation["settings"].map(str))
        .groupby(["model", "settings"], sort=False)["mae_bps"]
        .mean()
    )
    baseline = scores.loc["No change"].iat[0]
    trials = {
        name: scores.loc[name].rename_axis("settings").reset_index()
        for name in ("XGBoost", "Ridge")
    }
    deltas = [table["mae_bps"] - baseline for table in trials.values()]
    left = min(0, *(delta.min() for delta in deltas))
    right = max(0, *(delta.max() for delta in deltas))
    span = max(right - left, 0.1)
    years = ", ".join(str(year) for year in sorted(validation["target_year"].unique()))

    height = 2.7 + 0.36 * max(map(len, trials.values()))
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(11, height))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.25, top=0.67, wspace=0.52)
    fig.text(0.06, 0.92, "Which settings had the lowest validation error?",
             fontsize=20, weight="bold", color="#18283a")
    fig.text(0.06, 0.85, f"Validation years {years}. Lower average error is better.",
             fontsize=11, color="#46566a")

    for ax, (name, table) in zip(axes, trials.items()):
        labels = [setting_label(name, literal_eval(value)) for value in table["settings"]]
        best = table["mae_bps"].idxmin()
        delta = table["mae_bps"] - baseline
        ax.axvline(0, color="#64748b", linewidth=1.1, linestyle="--")
        ax.grid(axis="x", color="#e5eaf0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(name, loc="left", fontsize=14, weight="bold", color="#18283a")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(left - 0.08 * span, right + 0.24 * span)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c5cdd8")
        ax.tick_params(axis="y", length=0, pad=9, labelcolor="#24364b")
        ax.tick_params(axis="x", colors="#46566a")
        for index, (difference, mae) in enumerate(zip(delta, table["mae_bps"])):
            color = "#0f6d69" if index == best else "#64748b"
            ax.scatter(difference, index, color=color, s=66 if index == best else 43, zorder=3)
            ax.annotate(f"{mae:.2f}", (difference, index), xytext=(7, 0),
                        textcoords="offset points", va="center", color="#24364b",
                        fontsize=9, weight="bold" if index == best else "normal")

    fig.supxlabel("Difference from no-change MAE (basis points)", y=0.12, fontsize=11)
    fig.text(0.06, 0.035,
             f"Dashed line = no-change baseline ({baseline:.2f} bps). "
             "Dot labels show actual MAE. Green marks the best setting for each model.",
             fontsize=9, color="#46566a")
    fig.savefig(destination, dpi=200, facecolor="white")
    return fig


def plot_validation_by_model(validation, directory):
    """Save a chart of yearly and mean errors for each model's candidate settings."""
    annual = (
        validation.assign(settings=validation["settings"].map(str))
        .set_index(["model", "settings", "target_year"])["mae_bps"]
        .unstack("target_year", sort=False)
        .sort_index(axis=1)
    )
    baseline = annual.loc[["No change"]]
    all_means = annual.mean(axis=1)
    smallest, largest = all_means.min(), all_means.max()
    span = max(largest - smallest, 0.1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    for name in ("XGBoost", "Ridge", "LSTM"):
        candidates = annual.loc[[name]]
        selected = all_means.loc[name].idxmin()
        rows = pd.concat([candidates, baseline])
        means = rows.mean(axis=1)
        average_column = len(rows.columns)
        positions = range(len(rows))
        baseline_row = len(rows) - 1
        fig, (year_ax, mean_ax) = plt.subplots(
            1, 2, sharey=True, figsize=(10.8, 2.5 + 0.42 * len(rows)),
            gridspec_kw={"width_ratios": [1.30, 0.90]},
        )
        fig.subplots_adjust(left=0.215, right=0.955, top=0.66, bottom=0.21, wspace=0.23)
        fig.text(0.055, 0.90, f"{name} settings on validation data",
                 fontsize=22, weight="bold", color="#18283a")
        fig.text(0.055, 0.82, "Mean absolute return error in basis points. Lower is better.",
                 fontsize=11, color="#46566a")

        labels = [setting_label(name, literal_eval(value)) for value in candidates.index.get_level_values("settings")]
        year_ax.set_yticks(positions, [*labels, "No change benchmark"])
        year_ax.set_ylim(len(rows) - 0.5, -0.5)
        year_ax.set_xlim(-0.5, average_column + 0.5)
        year_ax.set_xticks(
            range(average_column + 1),
            [*(str(year) for year in rows.columns), f"{average_column}-year average"],
            fontsize=9,
        )
        year_ax.axvline(average_column - 0.5, color="#c5d4de", linewidth=1)
        year_ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=7)
        year_ax.set_title("Validation scores", loc="left", pad=26, fontsize=12, weight="bold")
        year_ax.tick_params(axis="y", length=0, pad=10, labelcolor="#24364b")
        mean_ax.set_title("Average across years", loc="left", pad=14, fontsize=12, weight="bold")
        mean_ax.tick_params(axis="y", left=False, labelleft=False)
        mean_ax.tick_params(axis="x", colors="#46566a", labelsize=9)
        mean_ax.set_xlim(smallest - 0.11 * span, largest + 0.11 * span)
        mean_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        mean_ax.grid(axis="x", color="#dee5ec")
        mean_ax.set_axisbelow(True)
        mean_ax.axvline(means.iloc[baseline_row], linestyle="--", linewidth=1, color="#64748b")
        for ax in (year_ax, mean_ax):
            ax.spines[["top", "left", "right", "bottom"]].set_visible(False)

        for position, (key, scores) in enumerate(rows.iterrows()):
            winner = key == (name, selected)
            is_baseline = position == baseline_row
            shade = "#e8eef6" if is_baseline else ("#f3f7fa" if position % 2 == 0 else "white")
            for ax in (year_ax, mean_ax):
                ax.axhspan(position - 0.49, position + 0.49, color=shade, linewidth=0, zorder=-1)
            for column, score in enumerate(scores):
                year_ax.text(column, position, f"{score:.2f}", ha="center", va="center",
                             color="#24364b", fontsize=10)

            color = "#0f6d69" if winner else "#64748b"
            value = means.loc[key]
            year_ax.text(average_column, position, f"{value:.2f}", ha="center", va="center",
                         color="#0f6d69" if winner else "#24364b",
                         weight="bold" if winner else "normal", fontsize=10)
            mean_ax.scatter(value, position, s=55 if winner else 39, color=color,
                            marker="D" if is_baseline else "o", zorder=3)
            if winner:
                label = year_ax.get_yticklabels()[position]
                label.set_color(color)
                label.set_weight("bold")

        fig.text(0.055, 0.052,
                 "Each year has equal weight. The lowest average within the model determines the selected setting.",
                 fontsize=9, color="#46566a")
        path = directory / f"validation_{name.lower()}_settings.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="?", default=Path("outputs/validation_scores.csv"))
    parser.add_argument("--by-model", action="store_true", help="Save a yearly comparison for each model.")
    args = parser.parse_args()
    validation = pd.read_csv(args.scores)
    if args.by_model:
        print(*plot_validation_by_model(validation, args.scores.parent), sep="\n")
    else:
        destination = args.scores.with_name("validation_settings.png")
        figure = plot_validation_settings(validation, destination)
        plt.close(figure)
        print(destination)
