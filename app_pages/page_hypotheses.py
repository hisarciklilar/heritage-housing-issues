import streamlit as st


def page_hypotheses_body():

    st.write("### Hypotheses")

    st.write('''
    Based on the exploratory data analysis,
    the following five hypotheses are formulated about the relationships
    between house attributes and sale price. This mainly addresses Business
    Requirement 1, which is about discovering how house attributes correlate
    with sale price.
    ''')

    st.success(
        f"H1. There is a strong positive relationship between overall house quality (**OverallQual**) and sale price"
    )

    st.success(
        f"H2. There is a positive relationship between kitchen quality (**KitchenQual**) and sale price."
    )

    st.success(
        f"H3. There is a positive relationship between above-ground living area (**GrLivArea**) and sale price."
    )

    st.success(
        f"H4. There is a positive relationship between garage area (**GarageArea**) and sale price"
    )

    st.success(
        f"H5. There is a positive relationship between basement area (**TotalBsmtSF**) and sale price"
    )

    st.markdown("""
    The hypotheses above are about the direction and strength of the
                relationship between each indicator and house sale price.
                These are confirmed by a combination of the following tools:

    1. Graphical representation of the relationship between each indicator
                and sales price (provided on the
                `Correlate House Price with Features` page):
    - Scatterplots of overall house quality, ground living area, garage area,
                and basement area with house sale price indicate a strong
                positive relationship between these indicators and house prices
    - Box plot of sales price for ordered outcomes of the kitchen quality
                variable reveals an increasing (non-linear) pattern, with
                faster increases at higher quality levels

    2. Calculation of correlation coefficients between each numerical
                indicator and sale price (provided on the
                `Correlate House Price with Features` page):
    - The four numerical indicators `OverallQual`, `GrLivArea`, `GarageArea`,
                and `TotalBsmtSF` have the highest correlations with sales 
                price

    3. Coefficient estimates and statistical significance tests based on
                Linear Regression (not reported here, as Random Forest
                performed better in terms of R², RMSE, and MAE)

    4. Random Forest-specific checks, such as assessing the predictive
                importance of each indicator
    """)