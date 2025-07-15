import numpy as np
import streamlit as st
#import seaborn as sns
#import matplotlib.pyplot as plt
from src.data_management import load_house_price_data

def page_correlations():

    df = load_house_price_data()

    st.write("## Correlation Analysis of the House Prices with House Features")
    st.info(
        f"* This section provides information on how house attributes " 
        f"measured in numerical scale correlate with house price"
        f"in the Ames, Iowa housing market"
    )

    st.write(
        "### Correlation of House Prices and Attributes"
    )

    # Correlation of indicators with house price 
    if st.checkbox("Inspect correlation of house price with house features"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr = df[numeric_cols].corr()['SalePrice'].sort_values(key=abs, ascending=False)[1:]
        st.write(corr)

    st.write("---")