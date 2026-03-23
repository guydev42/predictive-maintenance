"""Generate synthetic sensor reading data and save to CSV."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_sensor_data

DATA_DIR = os.path.dirname(__file__)


def main():
    df = generate_sensor_data(n_readings=15000, n_machines=50, failure_rate=0.08,
                              random_state=42)

    out_path = os.path.join(DATA_DIR, "sensor_readings.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} sensor readings to {out_path}")
    print(f"Machines: {df['machine_id'].nunique()}")
    print(f"Failure rate: {df['failure_within_7days'].mean():.4f} "
          f"({df['failure_within_7days'].sum()} failures)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
