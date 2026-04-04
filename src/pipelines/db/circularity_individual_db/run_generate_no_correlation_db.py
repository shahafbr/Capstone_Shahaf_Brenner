from src.synthetic_db.synthetic_scenarios import build_no_correlation_scenario
from src.synthetic_db.synthetic_generator import generate_and_save_scenario


def main():
    cfg = build_no_correlation_scenario()
    result = generate_and_save_scenario(cfg)
    print(result)


if __name__ == "__main__":
    main()