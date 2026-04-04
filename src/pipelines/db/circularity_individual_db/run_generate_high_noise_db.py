from src.synthetic_db.synthetic_scenarios import build_high_noise_scenario
from src.synthetic_db.synthetic_generator import generate_and_save_scenario


def main():
    cfg = build_high_noise_scenario()
    result = generate_and_save_scenario(cfg)
    print(result)


if __name__ == "__main__":
    main()