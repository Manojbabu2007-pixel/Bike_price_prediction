"""Lab 5 - schema validation of the raw Bike Details data (pandera)."""
import os

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column


def get_bike_schema():
    """Strict schema for BIKE DETAILS.csv (7 columns)."""
    return pa.DataFrameSchema(
        {
            "name": Column(str, coerce=True),
            "selling_price": Column(int, Check.gt(0)),
            "year": Column(int, Check.in_range(1980, 2026)),
            "seller_type": Column(str, Check.isin(["Individual", "Dealer"]), coerce=True),
            "owner": Column(
                str,
                Check.isin(["1st owner", "2nd owner", "3rd owner", "4th owner"]),
                coerce=True,
            ),
            "km_driven": Column(int, Check.ge(0)),
            # 435 of 1,061 values are legitimately missing in the raw data;
            # the pipeline imputes them, so missing is allowed but negatives are not.
            "ex_showroom_price": Column(float, Check.ge(0.0), nullable=True),
        },
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        print(f"[WARN] {n_dupes} duplicate rows found (they are dropped during preprocessing).")

    try:
        get_bike_schema().validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    data_path = "data/raw/BIKE DETAILS.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        raise SystemExit(1)
    ok = validate_schema(pd.read_csv(data_path), output_report_name="baseline_validation.csv")
    raise SystemExit(0 if ok else 1)
