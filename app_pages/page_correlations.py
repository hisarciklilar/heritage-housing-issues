import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
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
        "### Correlation Coefficients of House Sale Prices with House Features"
    )

    # Correlation of indicators with house price 
    if st.checkbox("Inspect correlation of house price with house features"):
        numeric_cols = df.select_dtypes(include=np.number).columns
        corr = df[numeric_cols].corr()['SalePrice'].sort_values(key=abs, ascending=False)[1:]

        st.write(corr)

    st.write("---")

    st.write(
        "### Correlation Heatmap of House Sale Price with House Features"
    )

    if st.checkbox("Reveal correlation heatmap of house price and features"):

        numeric_cols = df.select_dtypes(include=np.number).columns
        corr = df[numeric_cols].corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt="0.2f", 
            mask=mask, linewidths=1, cbar_kws={"shrink": 0.75}
        )
        plt.title("Correlation Heatmap")
        st.pyplot(fig)