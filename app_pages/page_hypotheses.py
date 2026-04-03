import streamlit as st


def page_hypotheses_body():

    st.write("### Hypotheses")

    st.write('''
    Based on the exploratory data analysis and correlation analysis,
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
