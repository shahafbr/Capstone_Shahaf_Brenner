import os

from src.thesis_ml.visualization import (
    build_cv_dataframe,
    plot_cv_boxplot,
    plot_cv_line,
    plot_cv_summary,
)


def main():
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    df = build_cv_dataframe("results")

    print("Rows per model:")
    print(df.groupby("model").size())
    print("\nModels found:", df["model"].unique())

    for metric in ["rmse", "mae", "r2"]:
        plot_cv_boxplot(df, metric, os.path.join(output_dir, f"boxplot_{metric}.png"))
        plot_cv_line(df, metric, os.path.join(output_dir, f"line_{metric}.png"))
        plot_cv_summary(df, metric, os.path.join(output_dir, f"summary_{metric}.png"))

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()