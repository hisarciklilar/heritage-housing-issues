import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_house_price_data

def page_house_price_study_body():

    df = load_house_price_data()

    st.write("## Exploratory Analysis of the Housing Market in Ames, Iowa")
    st.info(
        f"* This section provides an understanding of house attributes and the sale price\n"
        f"in the Ames, Iowa housing market"
    )

    st.write(
        "### Overview of House Prices and Attributes"
    )

    # Brief information about data 
    if st.checkbox("Inspect house prices and attributes"):
        st.write(
            f"* The dataset has {df.shape[0]} observations and {df.shape[1]} variables, "
            f"find below the first 10 rows."
        )

        st.write(df.head(10))

        # Info on missingness
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        st.write("Number of Missing Observations for Indicators:\n")
        st.write(missing)

    st.write("---")

    # Distribution of numeric variables
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    st.write("### Distribution of Variables in Data")
    selected_col = st.selectbox("Select a column", df.columns)

    if selected_col in numeric_cols:
        st.write(f"{selected_col} is a numeric variable")

        fig, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [1, 3]})

        # Boxplot (top)
        sns.boxplot(x=df[selected_col].dropna(), ax=axs[0])
        axs[0].set(title=f"Boxplot: {selected_col}")
        axs[0].set(xlabel='')
        axs[0].tick_params(axis='x', labelbottom=False)

        # Histogram with KDE (bottom)
        sns.histplot(df[selected_col].dropna(), kde=True, ax=axs[1])
        axs[1].set(title=f"Histogram: {selected_col}", xlabel=selected_col, ylabel="Count")

        plt.tight_layout()
        st.pyplot(fig)

    elif selected_col in cat_cols:

        st.write(f"{selected_col} is a categorical variable")

        fig, ax = plt.subplots(figsize=(8, 4))

        # Bar plot        
        df[selected_col].value_counts().plot(kind='bar')
        plt.title(f"Category Counts: {selected_col}")
        plt.xlabel(selected_col)
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        st.pyplot(fig)