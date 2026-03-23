"""
Synthetic industrial sensor data generator and loader.
Produces 15K sensor readings from 50 machines with ~8% failure rate within 7 days.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_sensor_data(n_readings=15000, n_machines=50, failure_rate=0.08,
                         random_state=42):
    """
    Generate synthetic sensor readings for predictive maintenance.

    Features:
        machine_id               - identifier for each machine (1-50)
        temperature              - operating temperature (Celsius)
        vibration                - vibration intensity (mm/s)
        pressure                 - system pressure (bar)
        rpm                      - rotational speed
        power_consumption        - power draw (kW)
        operating_hours          - total hours the machine has run
        age_months               - machine age in months
        maintenance_history_count - number of past maintenance events
        rolling_mean_temp_24h    - 24h rolling average temperature
        rolling_std_vibration_24h - 24h rolling standard deviation of vibration
        temp_pressure_ratio      - temperature / pressure interaction

    Target:
        failure_within_7days - binary (0 = no failure, 1 = failure expected)

    Returns:
        pd.DataFrame with all features and target
    """
    rng = np.random.RandomState(random_state)

    n_fail = int(n_readings * failure_rate)
    n_normal = n_readings - n_fail

    machine_ids_normal = rng.randint(1, n_machines + 1, size=n_normal)
    machine_ids_fail = rng.randint(1, n_machines + 1, size=n_fail)

    # Assign stable machine properties based on machine_id
    machine_ages = {}
    machine_hours = {}
    machine_maint_counts = {}
    for mid in range(1, n_machines + 1):
        machine_ages[mid] = rng.randint(6, 120)
        machine_hours[mid] = rng.randint(500, 50000)
        machine_maint_counts[mid] = rng.randint(0, 20)

    # --- Normal readings ---
    normal = _generate_readings(
        rng, n_normal, machine_ids_normal, machine_ages, machine_hours,
        machine_maint_counts, is_failure=False
    )
    normal["failure_within_7days"] = 0

    # --- Pre-failure readings ---
    failure = _generate_readings(
        rng, n_fail, machine_ids_fail, machine_ages, machine_hours,
        machine_maint_counts, is_failure=True
    )
    failure["failure_within_7days"] = 1

    df = pd.concat([
        pd.DataFrame(normal),
        pd.DataFrame(failure),
    ], ignore_index=True)

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Compute rolling / interaction features
    df["rolling_mean_temp_24h"] = df.groupby("machine_id")["temperature"].transform(
        lambda x: x.rolling(window=min(24, len(x)), min_periods=1).mean()
    ).round(2)

    df["rolling_std_vibration_24h"] = df.groupby("machine_id")["vibration"].transform(
        lambda x: x.rolling(window=min(24, len(x)), min_periods=1).std().fillna(0)
    ).round(4)

    df["temp_pressure_ratio"] = (df["temperature"] / df["pressure"].clip(lower=0.5)).round(4)

    return df


def _generate_readings(rng, n, machine_ids, machine_ages, machine_hours,
                       machine_maint_counts, is_failure):
    """Generate sensor readings for normal or pre-failure conditions."""
    if is_failure:
        # Pre-failure: elevated temperature, vibration, lower pressure
        temperature = rng.normal(loc=92, scale=12, size=n).clip(50, 150)
        vibration = rng.lognormal(mean=1.8, sigma=0.6, size=n).clip(1, 30)
        pressure = rng.normal(loc=3.5, scale=1.2, size=n).clip(0.5, 10)
        rpm = rng.normal(loc=1600, scale=350, size=n).clip(200, 3500)
        power_consumption = rng.normal(loc=85, scale=18, size=n).clip(10, 200)
    else:
        # Normal operation
        temperature = rng.normal(loc=68, scale=8, size=n).clip(30, 120)
        vibration = rng.lognormal(mean=0.8, sigma=0.5, size=n).clip(0.2, 15)
        pressure = rng.normal(loc=5.5, scale=0.8, size=n).clip(1, 10)
        rpm = rng.normal(loc=1450, scale=200, size=n).clip(400, 3000)
        power_consumption = rng.normal(loc=55, scale=12, size=n).clip(10, 150)

    age_months = np.array([machine_ages[mid] for mid in machine_ids])
    operating_hours = np.array([
        machine_hours[mid] + rng.randint(0, 500)
        for mid in machine_ids
    ])
    maintenance_history_count = np.array([
        machine_maint_counts[mid] for mid in machine_ids
    ])

    return {
        "machine_id": machine_ids,
        "temperature": temperature.round(2),
        "vibration": vibration.round(3),
        "pressure": pressure.round(2),
        "rpm": rpm.round(1),
        "power_consumption": power_consumption.round(2),
        "operating_hours": operating_hours,
        "age_months": age_months,
        "maintenance_history_count": maintenance_history_count,
    }


def load_and_prepare(filepath="data/sensor_readings.csv", test_size=0.2,
                     random_state=42):
    """
    Load sensor data and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath)

    feature_cols = [c for c in df.columns
                    if c not in ("failure_within_7days", "machine_id")]
    X = df[feature_cols].values.astype(float)
    y = df["failure_within_7days"].values
    feature_names = list(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {X_train.shape[1]}")
    print(f"Failure rate (train): {y_train.mean():.4f}")
    print(f"Failure rate (test):  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    df = generate_sensor_data()
    print(f"Generated {len(df)} sensor readings")
    print(f"Machines: {df['machine_id'].nunique()}")
    print(f"Failure rate: {df['failure_within_7days'].mean():.4f}")
    print(f"\nFeature summary:\n{df.describe().round(2)}")
