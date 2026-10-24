
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DIR = Path("quality_reports")
PLOT_DIR = REPORT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_REPORT = REPORT_DIR / "image_quality.csv"
TEXT_REPORT = REPORT_DIR / "text_quality.csv"
AUDIO_REPORT = REPORT_DIR / "audio_quality.csv"


def load_reports():
    """Load the quality reports produced by Trusted Zone."""

    images = pd.read_csv(IMAGE_REPORT)
    texts = pd.read_csv(TEXT_REPORT)
    audio = pd.read_csv(AUDIO_REPORT)

    return images, texts, audio


def numeric_column(df, column):
    """Convert a report column into numeric values."""
    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, column):
    """Read boolean values consistently from CSV."""
    return (
        df[column]
        .astype(str)
        .str.lower()
        .eq("true")
    )


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def plot_species_coverage(images, texts, audio):
    """Compare modality coverage across bird species."""

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


def plot_image_quality(images):
    """Analyze image dimensions and aspect ratios."""

    valid = images[
        images["status"].isin(["trusted", "already_trusted"])
    ].copy()

    for column in (
        "width",
        "height",
        "aspect_ratio",
        "size_bytes"
    ):
        valid[column] = numeric_column(valid, column)

    plt.figure(figsize=(9, 5))
    plt.hist(
        valid["width"].dropna(),
        bins=25,
        edgecolor="black"
    )
    plt.xlabel("Image width (pixels)")
    plt.ylabel("Number of images")
    plt.title("Distribution of Image Widths")
    save_plot("image_widths.png")

    plt.figure(figsize=(9, 5))
    plt.hist(
        valid["aspect_ratio"].dropna(),
        bins=25,
        edgecolor="black"
    )
    plt.xlabel("Width / height")
    plt.ylabel("Number of images")
    plt.title("Image Aspect Ratio Distribution")
    save_plot("image_aspect_ratios.png")

    return {
        "mean_width": valid["width"].mean(),
        "mean_height": valid["height"].mean(),
        "mean_aspect_ratio": valid["aspect_ratio"].mean(),
        "mean_size_kb": valid["size_bytes"].mean() / 1024
    }


def plot_text_quality(texts):
    """Analyze text lengths and cleaning effects."""

    accepted = texts[
        texts["status"].isin(["trusted", "cleaned", "already_trusted"])
    ].copy()

    for column in (
        "original_chars",
        "cleaned_chars",
        "original_words",
        "cleaned_words"
    ):
        accepted[column] = numeric_column(
            accepted, column
        )

    plt.figure(figsize=(9, 5))
    plt.hist(
        accepted["cleaned_words"].dropna(),
        bins=25,
        edgecolor="black"
    )
    plt.xlabel("Words per description")
    plt.ylabel("Number of descriptions")
    plt.title("Text Description Lengths")
    save_plot("text_word_counts.png")

    original_total = accepted["original_chars"].sum()
    cleaned_total = accepted["cleaned_chars"].sum()

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Before cleaning", "After cleaning"],
        [original_total, cleaned_total]
    )
    plt.ylabel("Total characters")
    plt.title("Effect of Text Cleaning")
    save_plot("text_cleaning_effect.png")

    return {
        "mean_words": accepted["cleaned_words"].mean(),
        "median_words": accepted["cleaned_words"].median(),
        "original_chars": original_total,
        "cleaned_chars": cleaned_total,
        "removed_chars": original_total - cleaned_total
    }


def plot_audio_quality(audio):
    """Analyze duration, silence, and volume."""

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


def plot_quality_outcomes(images, texts, audio):
    """Compare accepted and excluded files by modality."""

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
    """Save a readable summary of the analysis."""

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
    """Identify species missing one or more modalities."""

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


def analyze_audio_warnings(audio):
    """Investigate mostly silent recordings by species."""

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


def analyze_before_after(images, texts, audio):
    """Summarize processing decisions across modalities."""

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


def analyze_text_changes(texts):
    """Measure exactly how much each text file changed."""

    accepted = texts.loc[
        texts["status"].isin(["trusted", "cleaned", "already_trusted"])
    ].copy()

    accepted["original_chars"] = numeric_column(
        accepted, "original_chars"
    )

    accepted["cleaned_chars"] = numeric_column(
        accepted, "cleaned_chars"
    )

    accepted["characters_removed"] = (
        accepted["original_chars"]
        - accepted["cleaned_chars"]
    )

    distribution = (
        accepted["characters_removed"]
        .value_counts()
        .sort_index()
        .rename_axis("characters_removed")
        .reset_index(name="files")
    )

    distribution.to_csv(
        REPORT_DIR / "text_change_distribution.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        distribution["characters_removed"].astype(str),
        distribution["files"]
    )
    plt.xlabel("Net characters removed per file")
    plt.ylabel("Number of text files")
    plt.title("Distribution of Text Cleaning Changes")

    save_plot("text_change_distribution.png")

    print("\n========== TEXT CLEANING ==========")
    print(distribution.to_string(index=False))

    return distribution


def write_final_report(
    completeness,
    audio_summary,
    flagged_audio,
    before_after,
    text_changes
):
    """Generate a concise report based on measured results."""

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

    # Use the already-known original character total
    # from the text quality report.
    texts = pd.read_csv(TEXT_REPORT)

    total_original_chars = numeric_column(
        texts, "original_chars"
    ).sum()

    cleaning_percentage = (
        100 * total_removed_chars / total_original_chars
        if total_original_chars else 0
    )

    lines = [
        "# Trusted Zone — Data Quality Assessment",
        "",
        "## 1. Scope",
        "",
        "The analysis evaluates image, text and audio files "
        "processed from the Formatted Zone into the Trusted Zone.",
        "",
        "## 2. Integrity and acceptance",
        "",
        before_after.to_markdown(index=False),
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
        audio_summary.to_markdown(index=False),
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
        text_changes.to_markdown(index=False),
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
    images, texts, audio = load_reports()

    # Existing quality analysis
    coverage = plot_species_coverage(
        images, texts, audio
    )

    image_stats = plot_image_quality(images)
    text_stats = plot_text_quality(texts)
    audio_stats = plot_audio_quality(audio)

    plot_quality_outcomes(images, texts, audio)

    write_summary(
        images, texts, audio,
        image_stats, text_stats,
        audio_stats, coverage
    )

    # New detailed analysis
    completeness = analyze_completeness(coverage)

    audio_summary, flagged_audio = (
        analyze_audio_warnings(audio)
    )

    before_after = analyze_before_after(
        images, texts, audio
    )

    text_changes = analyze_text_changes(texts)

    write_final_report(
        completeness,
        audio_summary,
        flagged_audio,
        before_after,
        text_changes
    )

    print("\nQuality analysis completed.")
    print(f"Reports saved in: {REPORT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()


