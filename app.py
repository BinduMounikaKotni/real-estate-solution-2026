import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../../Models/model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def validate_inputs(sqft, year_built, property_age, year_sold):
    errors = []

    if sqft <= 0:
        errors.append("Square footage must be greater than 0.")

    if year_built > year_sold:
        errors.append("Year built cannot be greater than year sold.")

    if property_age < 0:
        errors.append("Property age cannot be negative.")

    return errors


def build_input_dataframe(
    beds,
    baths,
    sqft,
    year_built,
    lot_size,
    basement,
    popular,
    recession,
    property_age,
    property_type_condo,
    insurance,
    property_tax,
    year_sold
):
    return pd.DataFrame({
        "beds": [beds],
        "baths": [baths],
        "sqft": [sqft],
        "year_built": [year_built],
        "lot_size": [lot_size],
        "basement": [basement],
        "popular": [popular],
        "recession": [recession],
        "property_age": [property_age],
        "property_type_Condo": [property_type_condo],
        "insurance": [insurance],
        "property_tax": [property_tax],
        "year_sold": [year_sold]
    })


# ---------------- LOAD MODEL ---------------- #
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ---------------- SIDEBAR ---------------- #
st.sidebar.title("About Project")
st.sidebar.info(
    """
    **Real Estate Price Predictor**
    
    This app predicts house prices using a trained machine learning model.
    
    **Built with:**
    - Python
    - Streamlit
    - Pandas
    - Pickle / Scikit-learn
    """
)

if hasattr(model, "__class__"):
    st.sidebar.success(f"Model loaded: {model.__class__.__name__}")


# ---------------- HEADER ---------------- #
st.title("🏠 Real Estate Price Predictor")
st.markdown(
    "Enter the property details below to estimate the **predicted selling price**."
)

st.divider()

# ---------------- INPUT FORM ---------------- #
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Property Details")
        beds = st.number_input("Beds", min_value=0, max_value=10, value=3, help="Number of bedrooms")
        baths = st.number_input("Baths", min_value=0, max_value=10, value=2, help="Number of bathrooms")
        sqft = st.number_input("Square Footage", min_value=200, max_value=10000, value=1500)
        lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=200000, value=3000)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)

    with col2:
        st.subheader("Area & Structure")
        basement = st.selectbox("Basement Available", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        popular = st.selectbox("Popular Area", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        recession = st.selectbox("Recession Year", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        property_type_condo = st.selectbox("Property Type", [0, 1], format_func=lambda x: "Condo" if x == 1 else "Non-Condo")
        property_age = st.number_input("Property Age", min_value=0, max_value=200, value=20)

    with col3:
        st.subheader("Financial Details")
        insurance = st.number_input("Insurance", min_value=0, value=1200)
        property_tax = st.number_input("Property Tax", min_value=0, value=2500)
        year_sold = st.number_input("Year Sold", min_value=1800, max_value=2025, value=2020)

        st.markdown("### Quick Summary")
        st.metric("Bedrooms", beds)
        st.metric("Bathrooms", baths)
        st.metric("Area", f"{sqft} sqft")

    submitted = st.form_submit_button("Predict Price")


# ---------------- PREDICTION ---------------- #
if submitted:
    errors = validate_inputs(sqft, year_built, property_age, year_sold)

    if errors:
        for err in errors:
            st.warning(err)
    else:
        try:
            input_data = build_input_dataframe(
                beds,
                baths,
                sqft,
                year_built,
                lot_size,
                basement,
                popular,
                recession,
                property_age,
                property_type_condo,
                insurance,
                property_tax,
                year_sold
            )

            # Ensure input columns match training columns
            if hasattr(model, "feature_names_in_"):
                input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_data)[0]

            st.divider()
            st.subheader("Prediction Result")
            st.success(f"Estimated Property Price: ${prediction:,.2f}")

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.info(f"**Price per sqft (approx):** ${prediction / sqft:,.2f}")
            with result_col2:
                st.info(f"**Property Age:** {property_age} years")

            with st.expander("See input data sent to model"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")