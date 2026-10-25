"""Run all three quality analyses and produce one combined Markdown report

This file coordinates the separate image, text and audio modules.
"""

from pathlib import Path

# Use the same plotting backend as the individual analysis scripts
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from image_analysis import plot_image_quality
from text_analysis import plot_text_quality, analyze_text_changes
from audio_analysis import plot_audio_quality, analyze_audio_warnings

# Find quality_reports in the project root
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "quality_reports").is_dir():
    PROJECT_DIR = SCRIPT_DIR
else:
    PROJECT_DIR = SCRIPT_DIR.parent

REPORT_DIR = PROJECT_DIR / "quality_reports"
PLOT_DIR = REPORT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_REPORT = REPORT_DIR / "image_quality.csv"
TEXT_REPORT = REPORT_DIR / "text_quality.csv"
AUDIO_REPORT = REPORT_DIR / "audio_quality.csv"


def load_reports():
    """Read the three quality CSVs created by the Trusted Zone scripts"""
    images = pd.read_csv(IMAGE_REPORT)
    texts = pd.read_csv(TEXT_REPORT)
    audio = pd.read_csv(AUDIO_REPORT)
    return images, texts, audio


def numeric_column(df, column):
    """Convert a CSV column to numbers - invalid entries become missing."""
    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, column):
    """Read True/False values consistently from CSV"""
    return df[column].astype(str).str.lower().eq("true")


def save_plot(filename):
    """Save a plot to quality_reports/plots and close it"""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def markdown_table(df):
    """Create a Markdown table"""
    def cell(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def plot_species_coverage(images, texts, audio):
    """Compare modality coverage across bird species"""

    image_counts = images[
        images["status"].isin(["trusted", "already_trusted"])
    ]["species"].value_counts()

    text_counts = texts[
        texts["status"].isin(["trusted", "cleaned", "already_trusted"])
    ]["species"].value_counts()

    audio_counts = audio[
        audio["status"].isin(["trusted", "already_trusted"])
    ]["species"].value_counts()

    species = sorted(
        set(image_counts.index)
        | set(text_counts.index)
        | set(audio_counts.index)
    )

    coverage = pd.DataFrame({
        "species": species,
        "images": [image_counts.get(s, 0) for s in species],
        "texts": [text_counts.get(s, 0) for s in species],
        "audio": [audio_counts.get(s, 0) for s in species]
    })

    coverage.to_csv(
        REPORT_DIR / "species_coverage.csv",
        index=False
    )

    x = np.arange(len(species))
    width = 0.26

    plt.figure(figsize=(15, 6))
    plt.bar(
        x - width,
        coverage["images"],
        width,
        label="Images"
    )
    plt.bar(
        x,
        coverage["texts"],
        width,
        label="Text"
    )
    plt.bar(
        x + width,
        coverage["audio"],
        width,
        label="Audio"
    )

    plt.xticks(
        x,
        [s.split(".")[0] for s in species],
        rotation=45
    )
    plt.xlabel("Bird species ID")
    plt.ylabel("Number of files")
    plt.title("Multimodal Coverage by Bird Species")
    plt.legend()

    save_plot("species_coverage.png")

    return coverage

def plot_quality_outcomes(images, texts, audio):
    """Compare accepted and excluded files by modality"""

    datasets = {
        "Images": images,
        "Text": texts,
        "Audio": audio
    }

    labels = []
    accepted = []
    excluded = []

    accepted_statuses = {
        "trusted",
        "already_trusted",
        "cleaned"
    }

    for name, df in datasets.items():
        labels.append(name)

        accepted.append(
            int(df["status"].isin(accepted_statuses).sum())
        )

        excluded.append(
            int((df["status"] == "excluded").sum())
        )

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, accepted, label="Accepted")
    plt.bar(
        x,
        excluded,
        bottom=accepted,
        label="Excluded"
    )
    plt.xticks(x, labels)
    plt.ylabel("Number of files")
    plt.title("Trusted Zone Quality Outcomes")
    plt.legend()

    save_plot("quality_outcomes.png")

def write_summary(images, texts, audio,
                  image_stats, text_stats,
                  audio_stats, coverage):
    """Save a summary of the analysis"""

    summary = []

    summary.append("TRUSTED ZONE QUALITY ANALYSIS")
    summary.append("=" * 40)

    summary.append("\nDATASET OVERVIEW")
    summary.append(f"Images inspected: {len(images)}")
    summary.append(f"Texts inspected: {len(texts)}")
    summary.append(f"Audio inspected: {len(audio)}")
    summary.append(
        f"Species represented: {len(coverage)}"
    )

    summary.append("\nIMAGE QUALITY")
    for key, value in image_stats.items():
        summary.append(f"{key}: {value:.2f}")

    summary.append("\nTEXT QUALITY")
    for key, value in text_stats.items():
        summary.append(f"{key}: {value:.2f}")

    summary.append("\nAUDIO QUALITY")
    for key, value in audio_stats.items():
        summary.append(f"{key}: {value:.2f}")

    summary.append("\nSPECIES COVERAGE")
    summary.append(coverage.to_string(index=False))

    summary_path = REPORT_DIR / "quality_summary.txt"

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n".join(summary))
    print(f"\nSummary saved to: {summary_path}")

def analyze_completeness(coverage):
    """Identify species missing one or more modalities"""

    result = coverage.copy()

    for modality in ("images", "texts", "audio"):
        result[f"has_{modality}"] = result[modality] > 0

    result["modalities_present"] = (
        result[
            ["has_images", "has_texts", "has_audio"]
        ].sum(axis=1)
    )

    result["complete"] = result["modalities_present"] == 3

    result.to_csv(
        REPORT_DIR / "modality_completeness.csv",
        index=False
    )

    complete = int(result["complete"].sum())
    total = len(result)
    incomplete = result.loc[
        ~result["complete"], "species"
    ].tolist()

    print("\n========== MODALITY COMPLETENESS ==========")
    print(f"Species with all 3 modalities: {complete}/{total}")
    print(f"Species missing a modality: {total - complete}")

    for species in incomplete:
        print("Missing modality:", species)

    return result

def analyze_before_after(images, texts, audio):
    """Summarize processing decisions across modalities"""

    datasets = {
        "Images": images,
        "Text": texts,
        "Audio": audio
    }

    rows = []

    for modality, df in datasets.items():
        statuses = df["status"].astype(str).str.lower()

        accepted = int(
            statuses.isin(
                ["trusted", "already_trusted", "cleaned"]
            ).sum()
        )

        excluded = int((statuses == "excluded").sum())
        errors = int((statuses == "error").sum())

        cleaned = (
            int(
                boolean_column(
                    df.loc[
                        statuses.isin(
                            ["trusted", "cleaned", "already_trusted"]
                        )
                    ],
                    "changed"
                ).sum()
            )
            if modality == "Text"
            else 0
        )

        if modality == "Audio":
            flagged = int(
                boolean_column(
                    df.loc[statuses.isin(["trusted", "already_trusted"])],
                    "mostly_silent"
                ).sum()
            )
        elif modality == "Text":
            flagged = int(
                boolean_column(
                    df.loc[
                        statuses.isin(["trusted", "cleaned", "already_trusted"])
                    ],
                    "short_text"
                ).sum()
            )
        else:
            flagged = 0

        rows.append({
            "modality": modality,
            "formatted_files_inspected": len(df),
            "trusted_files_accepted": accepted,
            "excluded_files": excluded,
            "processing_errors": errors,
            "cleaned_files": cleaned,
            "flagged_files": flagged,
            "acceptance_percentage": round(
                100 * accepted / len(df), 2
            ) if len(df) else 0
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        REPORT_DIR / "before_after_quality.csv",
        index=False
    )

    print("\n========== BEFORE / AFTER ==========")
    print(result.to_string(index=False))

    return result

def write_final_report(
    completeness,
    audio_summary,
    flagged_audio,
    before_after,
    text_changes
):
    """Generate a report based on measured results."""

    total_species = len(completeness)
    complete_species = int(completeness["complete"].sum())

    missing_audio = completeness.loc[
        ~completeness["has_audio"], "species"
    ].tolist()

    audio_row = before_after.loc[
        before_after["modality"] == "Audio"
    ].iloc[0]

    text_row = before_after.loc[
        before_after["modality"] == "Text"
    ].iloc[0]

    total_original_chars = 0
    total_removed_chars = 0

    for _, row in text_changes.iterrows():
        total_removed_chars += (
            row["characters_removed"] * row["files"]
        )

    # Use the original character total from the text quality report.
    texts = pd.read_csv(TEXT_REPORT)

    total_original_chars = numeric_column(
        texts, "original_chars"
    ).sum()

    cleaning_percentage = (
        100 * total_removed_chars / total_original_chars
        if total_original_chars else 0
    )

    lines = [
        "# Data Quality Assessment",
        "",
        "## 1. Scope",
        "",
        "The analysis evaluates image, text and audio files "
        "processed from the Formatted Zone into the Trusted Zone.",
        "",
        "## 2. Integrity and acceptance",
        "",
        markdown_table(before_after),
        ""
        "## 3. Multimodal completeness",
        "",
        f"{complete_species} of {total_species} species "
        "have at least one file in each modality.",
        "",
        "Species without audio:",
        ""
    ]

    if missing_audio:
        lines.extend(
            f"- {species}" for species in missing_audio
        )
    else:
        lines.append("- None")

    lines.extend([
        ""
        "## 4. Audio quality",
        "",
        f"{len(flagged_audio)} of "
        f"{int(audio_row['trusted_files_accepted'])} accepted "
        "recordings were flagged as mostly silent.",
        "",
        "These recordings were retained because low amplitude "
        "does not necessarily imply unusable bird vocalizations.",
        "",
        "### Warnings by species",
        "",
        markdown_table(audio_summary),
        "",
        "## 5. Text cleaning",
        "",
        f"{int(text_row['cleaned_files'])} text files "
        "were modified during cleaning.",
        "",
        f"Net characters removed: {int(total_removed_chars)} "
        f"({cleaning_percentage:.3f}% of original characters).",
        "",
        "### Net character changes per file",
        "",
        markdown_table(text_changes),
        "",
        "A one-character reduction is consistent with removal "
        "of a trailing newline, but this should be verified "
        "against original file contents.",
        "",
        "## 6. Remaining limitations",
        "",
        "- Audio is missing for some bird species.",
        "- Quiet audio may contain useful vocalizations.",
        ""
    ])

    report_path = REPORT_DIR / "final_quality_report.md"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"\nFinal report saved to: {report_path}")


def main():
    """Coordinate the three independent analyses and build the final report"""
    # These CSVs must already exist from trust_images/text/audio.py
    images, texts, audio = load_reports()

    # Each modality's analysis has its own py file
    # These calls also save the modality-specific PNG plots
    image_stats = plot_image_quality(images)
    text_stats = plot_text_quality(texts)
    audio_stats = plot_audio_quality(audio)

    # Keep the original cross-modality charts and coverage calculations here
    coverage = plot_species_coverage(images, texts, audio)
    plot_quality_outcomes(images, texts, audio)
    write_summary(images, texts, audio,
                  image_stats, text_stats, audio_stats, coverage)

    # Detailed analyses to their appropriate modality
    completeness = analyze_completeness(coverage)
    audio_summary, flagged_audio = analyze_audio_warnings(audio)
    before_after = analyze_before_after(images, texts, audio)
    text_changes = analyze_text_changes(texts)

    # Produce one Markdown report
    write_final_report(completeness, audio_summary, flagged_audio,
                       before_after, text_changes)

    print("\nQuality analysis completed.")
    print("Markdown report:", REPORT_DIR / "final_quality_report.md")
    print("Plots folder:", PLOT_DIR)


if __name__ == "__main__":
    main()
