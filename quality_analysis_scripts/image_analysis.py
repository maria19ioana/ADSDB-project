"""Analyze image quality from the Trusted Zone CSV results"""

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
    """Read a numeric CSV column, treating invalid values as missing"""
    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, column):
    """Read True/False values correctly after loading them from CSV"""
    return df[column].astype(str).str.lower().eq("true")


def save_plot(filename):
    """Save the current chart in the plots folder and free its memory"""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


# Create plots and calculate image dimensions and aspect ratios
def plot_image_quality(images):
    """Analyze image dimensions and aspect ratios"""

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


def main():
    # Read the image validation results produced by trust_images.py
    images = pd.read_csv(REPORT_DIR / "image_quality.csv")

    # Reuse the original image statistics and two image plots
    stats = plot_image_quality(images)

    # Count accepted, excluded and other processing outcomes
    accepted = images["status"].isin(["trusted", "already_trusted"])
    excluded = images["status"].eq("excluded")
    results = {
        "inspected": len(images),
        "accepted": int(accepted.sum()),
        "excluded": int(excluded.sum()),
        "other": int((~(accepted | excluded)).sum()),
    }

    # Save a small summary for the image modality
    lines = ["IMAGE QUALITY ANALYSIS", "=" * 30]
    lines.extend(f"{name}: {value}" for name, value in results.items())
    lines.extend(f"{name}: {value:.2f}" for name, value in stats.items())
    path = REPORT_DIR / "image_analysis_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Image analysis saved to:", path)


if __name__ == "__main__":
    main()
