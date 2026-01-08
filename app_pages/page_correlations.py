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

        st.info(
            f"**Background Information:**\n\n"    
            "Correlation coefficients range from -1 to 1. "
            "A correlation coefficient close to 1 implies a strong positive correlation, "
            "while a coefficient close to -1 implies a strong negative correlation. "
            "A coefficient around 0 indicates no correlation."
        )
    
        st.write(corr)

        st.success(
            f"**Key Insights from the correlation table:**\n\n"
            f"The top five house features that have the highest correlation with house sale price are:\n\n"
            f"* Overall material and finish quality of the house (`OverallQual`) with a correlation coefficient of 0.79,\n"
            f"* Above grade (ground) living area in square feet (`GrLivArea`) with a correlation coefficient of 0.71,\n"
            f"* Size of garage in square feet (`GarageArea`) with a correlation coefficient of 0.62,\n"
            f"* Total square feet of basement area (`TotalBsmtSF`) with a correlation coefficient of 0.61, and\n"
            f"* First Floor square feet (`1stFlrSF`) with a correlation coefficient of 0.61.\n\n"
            f"While these features show strong positive correlations with house sale price, " 
            f"unfinished square feet of basement area (`BsmtUnfSF`) shows a weak negative correlation with a correlation coefficient of -0.18."
        )

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
        
        st.info(
            f"**Background Information:**\n\n"
            "A correlation heatmap visually represents the correlation coefficients "
            "between multiple variables in a dataset. "
            "The colors in the heatmap indicate the strength and direction of the correlations. "
            "In the heatmap provided below, red shades represent positive correlations and "
            " blue shades represent negative correlations. "
            "The darker the color, the stronger the correlation."
        )

        plt.title("Correlation Heatmap")
        st.pyplot(fig)

        st.success(
            f"**Key Insights from the correlation heatmap:**\n\n"
            "The last row of the heatmap reveals the correlations of various house features with house sale price. "
            "These are the correlations also reported above in the correlation table.\n\n "
            "The strength of the correlation between house features indicate which features are closely related to eachother. "
            "The features with strong correlations should be examined further for potential multicollinearity issues in predictive modeling.\n\n "
            "For example, total basement area (`TotalBsmtSF`) and first floor square feet (`1stFlrSF`) show a strong positive correlation of 0.82, "
            "indicating that houses with larger basements tend to have larger first floor areas as well. "
            "Similarly, there is a strong correlation (0.83) between year the house was built and the year garage was built. "
            "At the modelling stage, these correlations should be considered to avoid multicollinearity issues."
        )