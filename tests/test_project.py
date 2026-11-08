import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from car_utils import RAW_FEATURES, load_data  # noqa: E402


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(ROOT / "models" / "car_price_model.joblib")


def test_artifact_contains_everything_the_app_needs(artifact):
    expected = {"model", "model_name", "band", "metrics_holdout", "metrics_table", "reference",
                "options", "trend_age", "trend_dist", "feature_importance"}
    assert expected <= set(artifact)


def test_load_data_strips_column_whitespace():
    df = load_data(ROOT / "data" / "Cap_Test_Data_2025.csv")
    assert "Distance" in df.columns and "Distance " not in df.columns
    assert set(RAW_FEATURES) <= set(df.columns)


def test_predictions_match_saved_submission(artifact):
    test = load_data(ROOT / "data" / "Cap_Test_Data_2025.csv")
    saved = pd.read_csv(ROOT / "outputs" / "submission.csv")
    preds = artifact["model"].predict(test.drop(columns=["ID"]))
    assert (test["ID"].values == saved["ID"].values).all()
    np.testing.assert_allclose(np.round(preds, 2), saved["Price"].values, atol=0.01)


def test_preprocessor_repairs_impossible_values(artifact):
    prep = artifact["model"].regressor_.named_steps["prep"]
    row = pd.DataFrame([{
        "Maker": "skoda", "model": "octavia", "Location": "Pune", "Distance": 9_899_800,
        "Owner Type": "First", "manufacture_year": 2020, "Age of car": 5,
        "engine_displacement": 32000, "engine_power": 900, "body_type": np.nan,
        "Vroom Audit Rating": 6, "transmission": "man", "door_count": 4, "seat_count": 5,
        "fuel_type": "diesel"}])
    out = prep.transform(row)
    assert out["engine_displacement"].iloc[0] <= 6500
    assert out["engine_power"].iloc[0] <= 500
    assert out["distance"].iloc[0] <= 500_000
    assert not out.isna().any().any()
    assert (out.dtypes == "float64").all()


def test_dashboard_starts_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Estimate", "Batch predictions", "Market trends", "Model details"]
