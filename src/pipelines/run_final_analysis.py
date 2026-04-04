from src.thesis_ml.final_analysis import run_all_final_analysis


def main():
    result = run_all_final_analysis(
        results_root="results",
        output_dir="results/final_analysis",
        top_n=10,
    )

    print("Final analysis completed.")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()