import streamlit as st
import pandas as pd
import joblib
import numpy as np

from src.data_management import load_house_price_data
from src.data_management import load_inherited_house_price_data


@st.cache_resource
def load_pipeline():
    return joblib.load("outputs/models/house_price_pipeline.joblib")


def format_currency(value):
    return f"${value:,.0f}"


def page_predictions_body():
    st.write("### House Sale Price Prediction")
    st.info(
        "This page addresses Business Requirement 2: predicting house sale prices "
        "for the 4 inherited houses and for user-defined house attributes."
    )

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Could not load pipeline: {e}")
        st.stop()

    # --------------------------------------------------
    # Section 1: 4 inherited houses
    # --------------------------------------------------
    st.write("## Predictions for the 4 inherited houses")

    st.write("This section lists the characteristics of the 4 inherited houses and their predicted sale prices.")
    st.write("The summed predicted sale price for all 4 houses is also displayed below.")

    try:
        inherited_df = load_inherited_house_price_data()
        inherited_log_predictions = pipeline.predict(inherited_df)
        inherited_predictions = np.exp(inherited_log_predictions)

        with st.expander("Show characteristics of inherited houses"):
            inherited_results = inherited_df.copy()
            inherited_results["PredictedSalePrice"] = inherited_predictions
            st.dataframe(inherited_results)

        # inherited_results = inherited_df.copy()
        # inherited_results['PredictedSalePrice'] = inherited_predictions
        st.dataframe(pd.DataFrame({
            "PredictedSalePrice": inherited_predictions
        }))

        total_price = inherited_results['PredictedSalePrice'].sum()
        st.success(
            f"The summed predicted sale price for all 4 inherited houses is "
            f"**{format_currency(total_price)}**."
        )

    except FileNotFoundError:
        st.warning(
            "The file for the 4 inherited houses was not found. Please ensure it exists at 'outputs/datasets/collection/inherited_houses.csv'."
        )
    except Exception as e:
        st.error(f"Could not predict prices for the inherited houses: {e}")

    # --------------------------------------------------
    # Section 2: interactive prediction
    # --------------------------------------------------
    st.write("---")
    st.write("## Interactive house price prediction")


    st.info(
        "Enter house attributes below to predict the sale price of a house in Ames, Iowa."
    )

    col1, col2 = st.columns(2)

    with col1:
        first_flr_sf = st.number_input("First Floor Area (1stFlrSF)", min_value=0, value=0)
        second_flr_sf = st.number_input("Second Floor Area (2ndFlrSF)", min_value=0, value=0)
        bedroom_abv_gr = st.number_input("Bedrooms Above Ground (BedroomAbvGr)", min_value=0, value=0)
        bsmt_exposure = st.selectbox(
            "Basement Exposure (BsmtExposure)",
            ["Gd", "Av", "Mn", "No", "No_Basement"]
        )
        bsmt_fin_sf1 = st.number_input("Finished Basement Area (BsmtFinSF1)", min_value=0, value=0)
        bsmt_fin_type1 = st.selectbox(
            "Basement Finish Type 1 (BsmtFinType1)",
            ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "No_Basement"]
        )
        bsmt_unf_sf = st.number_input("Unfinished Basement Area (BsmtUnfSF)", min_value=0, value=0)
        enclosed_porch = st.number_input("Enclosed Porch Area", min_value=0, value=0)
        garage_area = st.number_input("Garage Area", min_value=0, value=0)
        garage_finish = st.selectbox(
            "Garage Finish",
            ["Fin", "RFn", "Unf", "No_Garage"]
        )
        garage_yr_blt = st.number_input("Garage Year Built (GarageYrBlt)", min_value=0, value=0)
        gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea)", min_value=0, value=0)

    with col2:
        kitchen_qual = st.selectbox(
            "Kitchen Quality",
            ["Po", "Fa", "TA", "Gd", "Ex"]
        )
        lot_area = st.number_input("Lot Area", min_value=0, value=0)
        lot_frontage = st.number_input("Lot Frontage", min_value=0.0, value=0.0)
        mas_vnr_area = st.number_input("Masonry Veneer Area (MasVnrArea)", min_value=0.0, value=0.0)
        open_porch_sf = st.number_input("Open Porch Area (OpenPorchSF)", min_value=0, value=0)
        overall_cond = st.slider("Overall Condition", min_value=1, max_value=10, value=5)
        overall_qual = st.slider("Overall Quality", min_value=1, max_value=10, value=5)
        total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF)", min_value=0, value=0)
        wood_deck_sf = st.number_input("Wood Deck Area (WoodDeckSF)", min_value=0, value=0)
        year_built = st.number_input("Year Built", min_value=0, value=0)
        year_remod_add = st.number_input("Year Remodeled (YearRemodAdd)", min_value=0, value=0)

    # Build one-row dataframe from user input
    user_data = pd.DataFrame({
        "1stFlrSF": [first_flr_sf],
        "2ndFlrSF": [second_flr_sf],
        "BedroomAbvGr": [bedroom_abv_gr],
        "BsmtExposure": [bsmt_exposure],
        "BsmtFinSF1": [bsmt_fin_sf1],
        "BsmtFinType1": [bsmt_fin_type1],
        "BsmtUnfSF": [bsmt_unf_sf],
        "EnclosedPorch": [enclosed_porch],
        "GarageArea": [garage_area],
        "GarageFinish": [garage_finish],
        "GarageYrBlt": [garage_yr_blt],
        "GrLivArea": [gr_liv_area],
        "KitchenQual": [kitchen_qual],
        "LotArea": [lot_area],
        "LotFrontage": [lot_frontage],
        "MasVnrArea": [mas_vnr_area],
        "OpenPorchSF": [open_porch_sf],
        "OverallCond": [overall_cond],
        "OverallQual": [overall_qual],        
        "TotalBsmtSF": [total_bsmt_sf],
        "WoodDeckSF": [wood_deck_sf],
        "YearBuilt": [year_built],
        "YearRemodAdd": [year_remod_add],
    })

    st.write("## Data entered")
    st.dataframe(user_data)

    if st.button("Predict Sale Price"):
        try:
            raw_prediction = pipeline.predict(user_data)[0]
            prediction = np.exp(raw_prediction)
            st.success(f"Predicted sale price: **{format_currency(prediction)}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
