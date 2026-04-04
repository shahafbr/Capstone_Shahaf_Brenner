from src.synthetic_db.synthetic_scenarios import build_weight_resample_scenario
from src.synthetic_db.synthetic_generator import generate_and_save_scenario


def main():
    cfg = build_weight_resample_scenario(4)
    result = generate_and_save_scenario(cfg)
    print(result)


if __name__ == "__main__":
    main()