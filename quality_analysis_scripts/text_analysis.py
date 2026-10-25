"""Analyze text quality from the Trusted Zone CSV results"""

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
    """Read a numeric CSV column, treating invalid values as missing."""
    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, column):
    """Read True/False values correctly after loading them from CSV."""
    return df[column].astype(str).str.lower().eq("true")


def save_plot(filename):
    """Save the current chart in the plots folder and free its memory"""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


# Measure description lengths and the effect of text cleaning
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


# Count how many characters were removed from each accepted text
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


def main():
    # Read the text validation and cleaning results from trust_text.py.
    texts = pd.read_csv(REPORT_DIR / "text_quality.csv")

    # Keep the original text-length and cleaning plots
    stats = plot_text_quality(texts)
    changes = analyze_text_changes(texts)

    # Include already-trusted texts - they may also have been cleaned
    accepted = texts["status"].isin(["trusted", "cleaned", "already_trusted"])
    cleaned = boolean_column(texts.loc[accepted], "changed")
    short = boolean_column(texts.loc[accepted], "short_text")

    # Summarize text quality
    results = {
        "inspected": len(texts),
        "accepted": int(accepted.sum()),
        "excluded": int(texts["status"].eq("excluded").sum()),
        "cleaned": int(cleaned.sum()),
        "short_text_warnings": int(short.sum()),
    }
    lines = ["TEXT QUALITY ANALYSIS", "=" * 30]
    lines.extend(f"{name}: {value}" for name, value in results.items())
    lines.extend(f"{name}: {value:.2f}" for name, value in stats.items())
    path = REPORT_DIR / "text_analysis_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Text analysis saved to:", path)


if __name__ == "__main__":
    main()
