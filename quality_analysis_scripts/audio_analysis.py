"""Analyze audio quality from the Trusted Zone CSV results"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "quality_reports").is_dir():
    PROJECT_DIR = SCRIPT_DIR
else:
    PROJECT_DIR = SCRIPT_DIR.parent

# Read existing Trusted Zone CSVs and save plots in the same report folder
REPORT_DIR = PROJECT_DIR / "quality_reports"
PLOT_DIR = REPORT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def numeric_column(df, column):
    """Read a numeric CSV column treating invalid values as missing"""
    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, column):
    """Read True/False values correctly after loading them from CSV"""
    return df[column].astype(str).str.lower().eq("true")


def save_plot(filename):
    """Save the current chart in the plots folder and free its memory"""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


# Plot recording duration, silence and warnings
def plot_audio_quality(audio):
    """Analyze duration, silence and volume."""

    valid = audio[
        audio["status"].isin(["trusted", "already_trusted"])
    ].copy()

    for column in (
        "duration_seconds",
        "rms",
        "peak",
        "silence_ratio",
        "clipping_ratio"
    ):
        valid[column] = numeric_column(
            valid, column
        )

    valid["mostly_silent_flag"] = boolean_column(
        valid, "mostly_silent"
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        valid["duration_seconds"].dropna(),
        bins=20,
        edgecolor="black"
    )
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of recordings")
    plt.title("Audio Duration Distribution")
    save_plot("audio_durations.png")

    plt.figure(figsize=(9, 5))
    plt.hist(
        valid["silence_ratio"].dropna(),
        bins=20,
        edgecolor="black"
    )
    plt.axvline(
        0.90,
        color="red",
        linestyle="--",
        label="Mostly silent threshold"
    )
    plt.xlabel("Fraction of low-amplitude samples")
    plt.ylabel("Number of recordings")
    plt.title("Audio Silence Ratio Distribution")
    plt.legend()
    save_plot("audio_silence_ratios.png")

    flagged_by_species = (
        valid.groupby("species")["mostly_silent_flag"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))
    flagged_by_species.plot(kind="bar")
    plt.xlabel("Bird species")
    plt.ylabel("Flagged recordings")
    plt.title("Mostly Silent Recordings by Species")
    plt.xticks(rotation=70, ha="right")
    save_plot("audio_silence_by_species.png")

    return {
        "mean_duration": valid["duration_seconds"].mean(),
        "median_duration": valid["duration_seconds"].median(),
        "mean_rms": valid["rms"].mean(),
        "mean_silence_ratio": valid["silence_ratio"].mean(),
        "mostly_silent_count": int(
            valid["mostly_silent_flag"].sum()
        )
    }


# Export silent-recording warnings grouped by species
def analyze_audio_warnings(audio):
    """Investigate mostly silent recordings by species"""

    valid = audio.loc[
        audio["status"].isin(["trusted", "already_trusted"])
    ].copy()

    valid["mostly_silent_flag"] = boolean_column(
        valid, "mostly_silent"
    )

    for column in (
        "duration_seconds",
        "silence_ratio",
        "rms",
        "peak"
    ):
        valid[column] = numeric_column(valid, column)

    summary = valid.groupby("species").agg(
        recordings=("key", "count"),
        flagged=("mostly_silent_flag", "sum"),
        mean_silence_ratio=("silence_ratio", "mean"),
        median_silence_ratio=("silence_ratio", "median"),
        mean_duration_seconds=("duration_seconds", "mean"),
        mean_rms=("rms", "mean")
    ).reset_index()

    summary["flagged_percentage"] = (
        100 * summary["flagged"] / summary["recordings"]
    ).round(2)

    summary.to_csv(
        REPORT_DIR / "audio_warning_summary.csv",
        index=False
    )

    flagged = valid.loc[
        valid["mostly_silent_flag"],
        [
            "key", "species", "duration_seconds",
            "silence_ratio", "rms", "peak"
        ]
    ].sort_values(
        "silence_ratio", ascending=False
    )

    flagged.to_csv(
        REPORT_DIR / "flagged_audio.csv",
        index=False
    )

    print("\n========== AUDIO WARNINGS ==========")
    print(f"Mostly silent recordings: {len(flagged)}")
    print(summary.to_string(index=False))

    return summary, flagged


def main():
    # Read the audio validation results produced by trust_audio.py
    audio = pd.read_csv(REPORT_DIR / "audio_quality.csv")

    # Keep the original duration, silence and warning analyses
    stats = plot_audio_quality(audio)
    warning_summary, flagged = analyze_audio_warnings(audio)

    # Count only accepted recordings when reporting silence warnings
    accepted = audio["status"].isin(["trusted", "already_trusted"])
    results = {
        "inspected": len(audio),
        "accepted": int(accepted.sum()),
        "excluded": int(audio["status"].eq("excluded").sum()),
        "processing_errors": int(audio["status"].eq("error").sum()),
        "mostly_silent_warnings": len(flagged),
    }

    # Save an audio summary
    lines = ["AUDIO QUALITY ANALYSIS", "=" * 30]
    lines.extend(f"{name}: {value}" for name, value in results.items())
    lines.extend(f"{name}: {value:.2f}" for name, value in stats.items())
    path = REPORT_DIR / "audio_analysis_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Audio analysis saved to:", path)


if __name__ == "__main__":
    main()
