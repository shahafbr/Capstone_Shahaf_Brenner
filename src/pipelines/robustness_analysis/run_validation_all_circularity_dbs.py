from src.thesis_ml.validation_runner import run_validation_models


SCENARIOS = [
    ("no_correlation", r"data\DB\no_correlation\synthetic_full_5000.csv"),
    ("low_noise", r"data\DB\low_noise\synthetic_full_5000.csv"),
    ("high_noise", r"data\DB\high_noise\synthetic_full_5000.csv"),
    ("alpha_plus_10", r"data\DB\alpha_plus_10\synthetic_full_5000.csv"),
    ("alpha_minus_10", r"data\DB\alpha_minus_10\synthetic_full_5000.csv"),
    ("drop_Task_completion", r"data\DB\drop_Task_completion\synthetic_full_5000.csv"),
    ("drop_After_hours_communication", r"data\DB\drop_After_hours_communication\synthetic_full_5000.csv"),
    ("drop_Break_frequency", r"data\DB\drop_Break_frequency\synthetic_full_5000.csv"),
    ("drop_Work_time_VS_clock_in_out", r"data\DB\drop_Work_time_VS_clock_in_out\synthetic_full_5000.csv"),
    ("weight_resample_01", r"data\DB\weight_resample_01\synthetic_full_5000.csv"),
    ("weight_resample_02", r"data\DB\weight_resample_02\synthetic_full_5000.csv"),
    ("weight_resample_03", r"data\DB\weight_resample_03\synthetic_full_5000.csv"),
    ("weight_resample_04", r"data\DB\weight_resample_04\synthetic_full_5000.csv"),
    ("weight_resample_05", r"data\DB\weight_resample_05\synthetic_full_5000.csv"),
]


def main():
    for scenario_name, data_path in SCENARIOS:
        print(f"\nRunning validation models for: {scenario_name}")
        results = run_validation_models(
            data_path=data_path,
            scenario_name=scenario_name,
        )
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
