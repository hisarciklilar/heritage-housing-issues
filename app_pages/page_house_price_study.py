import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import load_house_price_data

def page_house_price_study_body():

    df = load_house_price_data()

    st.write("### House Price Exploratory Analysis")
    st.info(
        f"* The client is interested in discovering how the house attributes correlate with sale price\n"
    )

    # Brief information about data 
    if st.checkbox("Inspect house prices and attributes"):
        st.write(
            f"* The dataset has {df.shape[0]} observations and {df.shape[1]} variables, "
            f"find below the first 10 rows."
        )

        st.write(df.head(10))
    
    st.write("---")

    # Distribution of numeric variables
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    st.write("### Distribution of Variables in Data")
    selected_col = st.selectbox("Select a numeric column", df.columns)

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