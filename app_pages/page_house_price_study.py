import numpy as np
import streamlit as st
from src.data_management import load_house_price_data

def page_house_price_study_body():

    df = load_house_price_data()

    st.write("### House Price Exploratory Analysis")
    st.info(
        f"* The client is interested in discovering how the house attributes correlate with sale price\n"
    )

    if st.checkbox("Inspect house prices and attributes"):
        st.write(
            f"* The dataset has {df.shape[0]} observations and {df.shape[1]} variables, "
            f"find below the first 10 rows."
        )

        st.write(df.head(10))