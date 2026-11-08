"""
Used Car Price Estimator (India) - Streamlit application.

Run from the project root:
    streamlit run app.py

Sections
    Estimate           single-vehicle valuation with price ranges and sensitivity charts
    Batch predictions  upload a CSV, receive predictions, download the result
    Market trends      median price by age and distance for selected makers
    Model details      hold-out accuracy, model comparison, feature importance
"""
from __future__ import annotations

import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# car_utils must be importable: the saved model references its preprocessing class.
from car_utils import CURRENT_YEAR, RAW_FEATURES, load_data

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
SAMPLE_INPUT_PATH = ROOT / "data" / "Cap_Test_Data_2025.csv"

# --------------------------------------------------------------------------- #
# Visual style: restrained palette, system fonts, no decoration
# --------------------------------------------------------------------------- #
BUTTER = "#FFF2BA"       # page background (Buttermilk)
MIDNIGHT = "#0F3C65"     # headings, sidebar, main data series (Midnight Blue)
BLUE_MID = "#4F7DAA"     # mid tint of the blue, for supporting data
BLUE_PALE = "#A9C2DA"    # pale tint of the blue, for wide ranges and secondary bars
PANEL = "#FBE99C"        # slightly deeper buttermilk for panels
LINE = "#E6D58E"         # hairlines and chart grid
INK = "#0A2B49"          # body text
MUTED = "#4A6580"        # secondary text
GRID = "rgba(15, 60, 101, 0.07)"   # page background grid
# One colour per maker: blues from the Midnight Blue family and golds from the Buttermilk family.
SERIES = ["#0F3C65", "#B8860B", "#4F7DAA", "#8C6D0A", "#2F5F8F", "#C9A227", "#7FA0C0", "#4A3B05"]
HEAD_FONT = "Georgia, 'Times New Roman', serif"
BODY_FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

st.set_page_config(page_title="Used Car Price Estimator", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ font-family: {BODY_FONT};
        background-image: linear-gradient({GRID} 1px, transparent 1px), linear-gradient(90deg, {GRID} 1px, transparent 1px);
        background-size: 28px 28px; }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}

    /* Headings and key figures: serif, in the primary colour */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMetricValue"] {{ font-family: {HEAD_FONT}; color: {MIDNIGHT}; letter-spacing: 0; }}
    .stApp h1 {{ font-size: 2rem; font-weight: 700; padding-bottom: 0.2rem; }}
    .stApp h2, .stApp h3 {{ font-weight: 700; }}
    .stApp [data-testid="stSidebar"] h1, .stApp [data-testid="stSidebar"] h2,
    .stApp [data-testid="stSidebar"] h3 {{ color: {BUTTER}; font-size: 1.05rem; }}
    .stApp [data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
    .stApp [data-testid="stCaptionContainer"] {{ color: {MUTED}; opacity: 1; }}
    .stApp code {{ color: {MIDNIGHT}; background: {PANEL}; }}

    /* Metric panels: quiet by default, respond to the pointer */
    .stApp [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid rgba(15, 60, 101, 0.25); border-radius: 8px; padding: 0.9rem 1.1rem;
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease; }}
    .stApp [data-testid="stMetric"]:hover {{
        border-color: {MIDNIGHT}; box-shadow: 0 4px 14px rgba(15, 60, 101, 0.18); transform: translateY(-2px); }}

    /* Tables and charts sit on a solid surface so the grid stays in the page margins */
    .stApp [data-testid="stTable"], .stApp [data-testid="stVegaLiteChart"], .stApp [data-testid="stArrowVegaLiteChart"] {{
        background: {BUTTER}; border: 1px solid rgba(15, 60, 101, 0.15); border-radius: 8px; padding: 6px 8px; }}

    /* Buttons and tabs */
    .stApp .stDownloadButton button, .stApp .stButton button {{
        border: 1px solid {MIDNIGHT}; color: {MIDNIGHT}; background: transparent; border-radius: 6px;
        transition: background .2s ease, color .2s ease; }}
    .stApp .stDownloadButton button:hover, .stApp .stButton button:hover {{
        background: {MIDNIGHT}; color: {BUTTER}; border-color: {MIDNIGHT}; }}
    .stApp .stTabs [data-baseweb="tab"] {{ font-weight: 600; transition: color .2s ease; }}
    .stApp .stTabs [data-baseweb="tab"]:hover {{ color: {BLUE_MID}; }}

    /* Notices in the blue of the palette */
    .stApp [data-testid="stAlertContainer"] {{ background: #E4EDF6; color: {INK}; border: 1px solid {BLUE_MID}; }}

    /* A gentle fade when a tab is opened */
    @keyframes riseIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
    .stApp [data-baseweb="tab-panel"] {{ animation: riseIn .45s ease-out; }}
    @media (prefers-reduced-motion: reduce) {{
        .stApp * {{ animation: none !important; transition: none !important; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model")
def load_artifact():
    """Load the trained model once and share it across sessions."""
    return joblib.load(MODEL_PATH)


def inr(x: float) -> str:
    """Short rupee format using lakh / crore."""
    x = float(x)
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def inr_exact(x: float) -> str:
    """Full rupee amount with Indian digit grouping (e.g. ₹8,43,200)."""
    n = int(round(float(x)))
    digits = str(abs(n))
    if len(digits) > 3:
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", digits[:-3])
        digits = f"{head},{digits[-3:]}"
    return f"{'-' if n < 0 else ''}₹{digits}"


def label(text: str) -> str:
    """Display name: short names (bmw, q3, tt, x5) in capitals, others in title case."""
    return text.upper() if len(text) <= 3 else text.title()


def build_row(maker, model, location, distance, owner, age, disp, power, rating,
              transmission, doors, seats, fuel, body_type) -> pd.DataFrame:
    """Turn the form inputs into a one-row frame with the raw training schema."""
    return pd.DataFrame([{
        "Maker": maker, "model": model, "Location": location, "Distance": distance,
        "Owner Type": owner, "manufacture_year": CURRENT_YEAR - age, "Age of car": age,
        "engine_displacement": disp, "engine_power": power, "body_type": body_type,
        "Vroom Audit Rating": rating, "transmission": transmission,
        "door_count": doors, "seat_count": seats, "fuel_type": fuel,
    }])


def index_of(options: list, value, default: int = 0) -> int:
    """Index of `value` in `options`, or `default` when it is not offered."""
    try:
        return options.index(value)
    except ValueError:
        return default


def style_chart(chart: alt.Chart, height: int = 280) -> alt.Chart:
    """Apply one consistent, understated look to every chart."""
    return (
        chart.properties(height=height)
        .configure(background="transparent", font=BODY_FONT)
        .configure_view(stroke=None)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domainColor=LINE,
                        tickColor=LINE, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                          orient="top", direction="horizontal", title=None, columns=4)
    )


def show(chart: alt.Chart) -> None:
    st.altair_chart(chart, theme=None, width="stretch")


FEATURE_LABELS = {
    "distance": "Distance driven", "log_distance": "Distance driven (log scale)",
    "km_per_year": "Kilometres per year", "age": "Age of car",
    "power_per_litre": "Power per litre of displacement", "engine_power": "Engine power",
    "engine_displacement": "Engine displacement", "audit_rating": "Audit rating",
    "owner_ordinal": "Owner type", "is_automatic": "Automatic transmission",
    "is_diesel": "Diesel fuel", "door_count": "Door count", "seat_count": "Seat count",
    "distance_missing": "Distance missing", "power_missing": "Power missing",
    "disp_invalid_or_missing": "Displacement invalid or missing", "has_body_type": "Special body type",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    prefix, _, rest = name.partition("_")
    group = {"mm": "Model", "maker": "Maker", "loc": "Location"}.get(prefix)
    if group:
        return f"{group}: {label(rest.replace('_', ' '))}"
    return name.replace("_", " ").capitalize()


# --------------------------------------------------------------------------- #
# Model and sidebar inputs
# --------------------------------------------------------------------------- #
try:
    art = load_artifact()
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH.relative_to(ROOT)}`. "
             "Run the notebook in `notebooks/` to train and save the model.")
    st.stop()

ref, opts, model = art["reference"], art["options"], art["model"]

st.sidebar.header("Vehicle details")

st.sidebar.subheader("Identification")
maker = st.sidebar.selectbox("Maker", opts["makers"], format_func=label)
models_for_maker = sorted(v["model"] for v in ref.values() if v["maker"] == maker)
model_name = st.sidebar.selectbox("Model", models_for_maker, format_func=label)
spec = ref[f"{maker}_{model_name}"]
location = st.sidebar.selectbox("Location", opts["locations"])

st.sidebar.subheader("Specification")
fuel = st.sidebar.radio("Fuel type", spec["fuels"], format_func=str.title, horizontal=True)
transmission = st.sidebar.radio(
    "Transmission", spec["transmissions"],
    format_func=lambda t: "Automatic" if t == "auto" else "Manual", horizontal=True)

st.sidebar.subheader("Usage and condition")
age = st.sidebar.slider("Age of car (years)", 0, 30, 5)
distance = st.sidebar.slider("Distance driven (km)", 0, 300_000, 60_000, step=1_000)
owner = st.sidebar.selectbox("Owner type", opts["owners"])
rating = st.sidebar.slider("Audit rating", 4, 8, 7)

with st.sidebar.expander("Engine and body (defaults come from the selected model)"):
    disp = st.number_input("Engine displacement (cc)", 600, 6500, int(spec["displacement"]), step=10)
    power = st.number_input("Engine power (kW)", 20, 500, int(spec["power"]), step=1)
    door_options, seat_options = [2, 3, 4, 5], [2, 4, 5, 7]
    doors = st.selectbox("Doors", door_options, index=index_of(door_options, int(spec["doors"]), 2))
    seats = st.selectbox("Seats", seat_options, index=index_of(seat_options, int(spec["seats"]), 2))
    body = st.selectbox("Body type", ["standard", "compact", "van"], format_func=str.title)
body_type = np.nan if body == "standard" else body

st.sidebar.subheader("Price range")
band_pct = st.sidebar.slider("Fair range (± %)", 1, 20, 5,
                             help="Symmetric band placed around the point estimate.")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
metrics = art["metrics_holdout"]
st.title("Used Car Price Estimator")
st.caption(f"Indian pre-owned car market. Model: {art['model_name']}. "
           f"Hold-out R² {metrics['R2']:.3f}; mean absolute error {inr(metrics['MAE'])}.")

tab_est, tab_batch, tab_trend, tab_info = st.tabs(
    ["Estimate", "Batch predictions", "Market trends", "Model details"])

car = dict(maker=maker, model=model_name, location=location, owner=owner, disp=disp, power=power,
           rating=rating, transmission=transmission, doors=doors, seats=seats, fuel=fuel,
           body_type=body_type)


def predict_grid(variants: list[dict]) -> np.ndarray:
    """Predict variants of the current car (used by the sensitivity charts)."""
    base = dict(car, distance=distance, age=age)
    rows = [build_row(**{**base, **v}) for v in variants]
    return model.predict(pd.concat(rows, ignore_index=True))


# --------------------------------------------------------------------------- #
# Estimate
# --------------------------------------------------------------------------- #
with tab_est:
    price = float(model.predict(build_row(**car, distance=distance, age=age))[0])
    lo_sym, hi_sym = price * (1 - band_pct / 100), price * (1 + band_pct / 100)
    lo_emp, hi_emp = price * art["band"]["low_ratio"], price * art["band"]["high_ratio"]

    st.subheader(f"{label(maker)} {label(model_name)}, {age} years old, {distance:,} km, {location}")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Estimated price", inr(price), delta=inr_exact(price), delta_color="off", delta_arrow="off")
    with right:
        st.table(pd.DataFrame({
            "Range": [f"Fair range (± {band_pct}%)", "Empirical range (80%)"],
            "Lower": [inr_exact(lo_sym), inr_exact(lo_emp)],
            "Upper": [inr_exact(hi_sym), inr_exact(hi_emp)],
            "Basis": ["Symmetric band around the estimate",
                      "10th to 90th percentile of errors on unseen hold-out cars"],
        }).set_index("Range"))

    ranges = pd.DataFrame({
        "Range": [f"Fair range (± {band_pct}%)", "Empirical range (80%)"],
        "Low": [lo_sym / 1e5, lo_emp / 1e5], "High": [hi_sym / 1e5, hi_emp / 1e5],
    })
    bars = alt.Chart(ranges).mark_bar(height=16).encode(
        x=alt.X("Low:Q", title="Price (₹ lakh)", scale=alt.Scale(zero=False)),
        x2="High:Q",
        y=alt.Y("Range:N", title=None, sort=list(ranges["Range"])),
        color=alt.Color("Range:N", scale=alt.Scale(domain=list(ranges["Range"]), range=[BLUE_MID, BLUE_PALE]), legend=None),
        tooltip=["Range", alt.Tooltip("Low:Q", format=",.2f"), alt.Tooltip("High:Q", format=",.2f")],
    )
    marker = alt.Chart(pd.DataFrame({"Estimate": [price / 1e5]})).mark_rule(color=MIDNIGHT, strokeWidth=3).encode(
        x="Estimate:Q", tooltip=[alt.Tooltip("Estimate:Q", format=",.2f", title="Estimate (₹ lakh)")])
    show(style_chart(bars + marker, height=110))
    st.caption("The vertical line marks the point estimate.")

    st.subheader("Sensitivity of the estimate")
    st.caption("Each chart changes one input and holds the others at the values chosen in the sidebar.")

    ages = np.arange(0, 21)
    age_df = pd.DataFrame({"Age (years)": ages,
                           "Price (₹ lakh)": predict_grid([{"age": int(a)} for a in ages]) / 1e5})
    dists = np.arange(0, 300_001, 15_000)
    dist_df = pd.DataFrame({"Distance (thousand km)": dists / 1000,
                            "Price (₹ lakh)": predict_grid([{"distance": int(d)} for d in dists]) / 1e5})

    c1, c2 = st.columns(2, gap="large")
    c1.markdown("**Price by age of car**")
    c1.line_chart(age_df, x="Age (years)", y="Price (₹ lakh)", color=MIDNIGHT, height=260)
    c1.caption(f"Selected car: {age} years old, estimated at {inr(price)}.")
    c2.markdown("**Price by distance driven**")
    c2.line_chart(dist_df, x="Distance (thousand km)", y="Price (₹ lakh)", color=MIDNIGHT, height=260)
    c2.caption(f"Selected car: {distance:,} km driven, estimated at {inr(price)}.")

    st.caption("In this dataset, location, owner type and audit rating have very little effect on price.")

# --------------------------------------------------------------------------- #
# Batch predictions
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Predictions from a CSV file")
    st.write("Upload a file containing these columns: " + ", ".join(f"`{c}`" for c in RAW_FEATURES) +
             ". An `ID` column is optional and is carried through to the output.")

    if SAMPLE_INPUT_PATH.exists():
        sample = pd.read_csv(SAMPLE_INPUT_PATH).head(25)
        st.download_button("Download a sample input file", sample.to_csv(index=False).encode(),
                           file_name="sample_input.csv", mime="text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = load_data(up)
            missing_cols = [c for c in RAW_FEATURES if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                with st.spinner("Predicting"):
                    preds = model.predict(df[RAW_FEATURES])
                out = pd.DataFrame({"ID": df["ID"] if "ID" in df else np.arange(1, len(df) + 1),
                                    "Price": np.round(preds, 2)})
                st.success(f"Predictions generated for {len(out):,} vehicles.")

                s1, s2, s3 = st.columns(3)
                s1.metric("Mean price", inr(out["Price"].mean()))
                s2.metric("Median price", inr(out["Price"].median()))
                s3.metric("Highest price", inr(out["Price"].max()))

                st.dataframe(out.head(100), hide_index=True, width="stretch")
                st.caption("The first 100 rows are shown. The download contains all rows.")
                st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   file_name="predictions.csv", mime="text/csv")

                st.markdown("**Distribution of predicted prices**")
                hist = alt.Chart(pd.DataFrame({"Price (₹ lakh)": out["Price"] / 1e5})).mark_bar(color=BLUE_MID).encode(
                    x=alt.X("Price (₹ lakh):Q", bin=alt.Bin(maxbins=30)),
                    y=alt.Y("count():Q", title="Number of vehicles"))
                show(style_chart(hist, height=240))
        except Exception as exc:  # keep the app running on malformed uploads
            st.error(f"The file could not be processed: {exc}")

# --------------------------------------------------------------------------- #
# Market trends
# --------------------------------------------------------------------------- #
with tab_trend:
    st.subheader("Median price in the training data")
    default_picks = [maker, "bmw", "toyota"] if maker not in ("bmw", "toyota") else [maker, "skoda"]
    picks = st.multiselect("Makers to compare", opts["makers"], default=default_picks, format_func=label)
    if picks:
        colour = {m: SERIES[i % len(SERIES)] for i, m in enumerate(opts["makers"])}
        ordered = sorted(picks)                      # pivot() orders the columns alphabetically
        colours = [colour[m] for m in ordered]
        ta = art["trend_age"][art["trend_age"]["Maker"].isin(picks)].copy()
        td = art["trend_dist"][art["trend_dist"]["Maker"].isin(picks)].copy()
        ta = ta.pivot(index="Age", columns="Maker", values="MedianPrice") / 1e5
        td["Distance (thousand km)"] = td["dist_mid_km"].astype(float) / 1000
        td = td.pivot(index="Distance (thousand km)", columns="Maker", values="MedianPrice") / 1e5
        ta, td = ta[ordered].rename(columns=label), td[ordered].rename(columns=label)

        t1, t2 = st.columns(2, gap="large")
        t1.markdown("**By age of car (years)**")
        t1.line_chart(ta, color=colours, height=300, y_label="Median price (₹ lakh)")
        t2.markdown("**By distance driven (thousand km, binned)**")
        t2.line_chart(td, color=colours, height=300, y_label="Median price (₹ lakh)")
        st.caption("Ages above 20 years are grouped into 20. Distance points are bin midpoints.")
    else:
        st.info("Select at least one maker.")

# --------------------------------------------------------------------------- #
# Model details
# --------------------------------------------------------------------------- #
with tab_info:
    st.subheader("Accuracy on a 20% hold-out sample")
    mt = art["metrics_table"].copy()
    table = pd.DataFrame({
        "Model": mt["Model"],
        "RMSE": mt["RMSE"].map(inr_exact),
        "MAE": mt["MAE"].map(inr_exact),
        "R²": mt["R2"].map("{:.4f}".format),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("RMSE: root mean squared error. MAE: mean absolute error. Both are in rupees. "
               f"The deployed model, {art['model_name']}, has the lowest RMSE.")

    mae_chart = alt.Chart(mt.assign(**{"MAE (₹ lakh)": mt["MAE"] / 1e5})).mark_bar(height=18).encode(
        x=alt.X("MAE (₹ lakh):Q"), y=alt.Y("Model:N", sort="x", title=None),
        color=alt.condition(alt.datum.Model == art["model_name"], alt.value(MIDNIGHT), alt.value(BLUE_PALE)),
        tooltip=["Model", alt.Tooltip("MAE (₹ lakh):Q", format=",.2f")])
    st.markdown("**Mean absolute error by model (lower is better; deployed model highlighted)**")
    show(style_chart(mae_chart, height=200))

    st.subheader("Most influential inputs")
    fi = art["feature_importance"].copy()
    fi["Input"] = fi["feature"].map(feature_label)
    fi["Share of splits"] = fi["importance"] / fi["importance"].sum()
    top = fi.head(12)
    fi_chart = alt.Chart(top).mark_bar(color=MIDNIGHT, height=16).encode(
        x=alt.X("Share of splits:Q", axis=alt.Axis(format="%")),
        y=alt.Y("Input:N", sort="-x", title=None),
        tooltip=["Input", alt.Tooltip("Share of splits:Q", format=".1%")])
    show(style_chart(fi_chart, height=300))
    st.caption("Importance is the share of tree splits that use each input. Split counts overstate noisy "
               "numeric inputs; the notebook includes a permutation-importance check.")

    st.subheader("Limitations")
    st.write("The data set appears partly synthetic, so estimates are indicative and are not a formal "
             "appraisal. Predictions for very old vehicles, very high mileage or rarely listed models "
             "(for example Maserati) are less reliable.")
