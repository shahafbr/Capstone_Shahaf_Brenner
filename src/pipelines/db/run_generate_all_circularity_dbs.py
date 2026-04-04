from src.synthetic_db.synthetic_scenarios import (
    build_no_correlation_scenario,
    build_low_noise_scenario,
    build_high_noise_scenario,
    build_alpha_plus_10_scenario,
    build_alpha_minus_10_scenario,
    build_feature_drop_scenario,
    build_weight_resample_scenario,
)
from src.synthetic_db.synthetic_generator import generate_and_save_scenario


def main():
    scenarios = [
        build_no_correlation_scenario(),
        build_low_noise_scenario(),
        build_high_noise_scenario(),
        build_alpha_plus_10_scenario(),
        build_alpha_minus_10_scenario(),
        build_feature_drop_scenario("Task_completion"),
        build_feature_drop_scenario("After_hours_communication"),
        build_feature_drop_scenario("Break_frequency"),
        build_feature_drop_scenario("Work_time_VS_clock_in_out"),
        build_weight_resample_scenario(1),
        build_weight_resample_scenario(2),
        build_weight_resample_scenario(3),
        build_weight_resample_scenario(4),
        build_weight_resample_scenario(5),
    ]

    for cfg in scenarios:
        result = generate_and_save_scenario(cfg)
        print(cfg.scenario_name, result)


if __name__ == "__main__":
    main()