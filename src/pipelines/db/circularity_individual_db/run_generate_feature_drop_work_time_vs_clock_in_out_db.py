from src.synthetic_db.synthetic_scenarios import build_feature_drop_scenario
from src.synthetic_db.synthetic_generator import generate_and_save_scenario


def main():
    cfg = build_feature_drop_scenario("Work_time_VS_clock_in_out")
    result = generate_and_save_scenario(cfg)
    print(result)


if __name__ == "__main__":
    main()