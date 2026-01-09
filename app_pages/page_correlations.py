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
        f"correlate with house price"
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
            f"* Overall material and finish quality of the house (`OverallQual`) and above ground living area (`GrLivArea`) "
            f"have strong correlations (0.79 and 0.71, respectively) with house price indicating that "
            f"houses with better quality and larger above ground living areas tend to have higher sale prices.\n"
            f"* Size of garage (`GarageArea`), size of basement area (`TotalBsmtSF`) show moderate to strong correlation (0.62 and 0.61 respectively) "
            f"confirming that the larger houses are priced higher. \n"
            f"* Year of built (`YearBuilt`) and remodel date (`YearRemodAdd`) show moderate positive correlation (0.52 and 0.51), indicating "
            f"that newer houses tend to have higher sale prices.\n"
            f"* Open porch area (`OpenPorchSF`), wood deck area (`WoodDeckSF`), and linear feet of street connected to the property (`LotFrontage`) "
            f"show weak positive correlations (0.32, 0.25, 0.35 respectively) indicating that features related to outdoor space have a smaller impact on house sale price.\n"
            f"* Unfinished square feet of basement area (`BsmtUnfSF`) shows a weak negative correlation with a correlation coefficient of -0.18."
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
    
    st.write("---")

    st.write("### Sale Price Relationships with Key House Features")

    if st.checkbox("Visualize relationship between house price and overall quality rating of the house"):
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="OverallQual", y="SalePrice", ax=ax)
        ax.set_title("Sale Price by Overall Quality Rating")
        ax.set_xlabel("Overall Quality")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)
    
        st.success(
        f"Sale prices increase markedly with increasing levels of overall quality rating. "
        f"The increase appears to be exponential. "
        f"As the overall quality increases, the median sale price changes at an increasing rate. "
        f"We also observe higher variation in house prices at increasing levels of overall house quality. "
        f"This may be an indication of heteroscedasticity in the data (something to be considered during modelling)." 
        )

    if st.checkbox("Visualize relationship between house price and size of above-ground living area"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", ax=ax)
        ax.set_title("Sale Price vs Above-Ground Living Area")
        ax.set_xlabel("Above-Ground Living Area (sq ft)")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)

        st.success(
            "Larger above-ground living area is associated with higher sale prices. "
            "This indicates that house size is a key determinant of value. Larger above-ground living areas show "
            "greater variability in price, an indicator of heteroscedasticity that should be considered during modeling."
        )

    if st.checkbox("Visualize relationship between house price and total basement size"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="TotalBsmtSF", y="SalePrice", ax=ax)
        ax.set_title("Sale Price vs Total Basement Area")
        ax.set_xlabel("Total Basement Area (sq ft)")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)

        st.success(
            "Houses with larger basements tend to sell for more, indicating that "
            "basement space contributes positively to perceived value. The spread across sale prices "
            "suggests that in addition to size, basement quality and finish may also matter."
        )

    if st.checkbox("Visualize relationship between house price and garage size"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="GarageArea", y="SalePrice", ax=ax)
        ax.set_title("Sale Price vs Garage Area")
        ax.set_xlabel("Garage Area (sq ft)")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)

        st.success(
            "Larger garage areas are broadly associated with higher sale prices. "
            "This reflects the value buyers place on garage capacity in Ames."
        )

    if st.checkbox("Visualize relationship between house price and year of built"):
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="YearBuilt", y="SalePrice", ax=ax)
        ax.set_title("Sale Price vs Year Built")
        ax.set_xlabel("Year Built")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)

        st.success(
            "Most recently built houses tend to have higher prices, suggesting a premium for "
            "modern construction. However, the spread of points indicates that other factors may also play "
            "important roles. The relationship is not as strong as with other features such as overall quality or size."
        )

    if st.checkbox("Visualize relationship between house price and kitchen quality"):
        kitchen_order = ["Po", "Fa", "TA", "Gd", "Ex"]
        kitchen_labels = {
            "Po": "Poor",
            "Fa": "Fair",
            "TA": "Typical / Average",
            "Gd": "Good",
            "Ex": "Excellent"
        }
                     
        fig, ax = plt.subplots()
        sns.boxplot(
            data=df,
            x="KitchenQual",
            y="SalePrice",
            order=kitchen_order,
            ax=ax
        )
        ax.set_xticklabels([kitchen_labels[k] for k in kitchen_order])
        ax.set_title("Sale Price by Kitchen Quality")
        ax.set_xlabel("Kitchen Quality")
        ax.set_ylabel("Sale Price")
        st.pyplot(fig)

        st.success(
            "Higher kitchen quality is associated with higher sale prices, "
            "highlighting the importance of kitchens as a visible quality signal to buyers."
            "We observe higher price variation at mid to high kitchen quality levels."
        )

    st.write("---")

    st.markdown("### Conclusions")
    st.markdown(
        """
    - Sale price in Ames is strongly influenced by **house quality**, **living space**, and
    **key amenities** such as basements and garages.
    - The correlation coefficients and the visual insights help identify which attributes are most relevant when pricing
    the client's inherited properties.
    - The same features will be prioritised as inputs to the machine-learning model used
    for sale price prediction.
    """
    )