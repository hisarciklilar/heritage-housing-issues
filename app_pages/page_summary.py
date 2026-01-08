import streamlit as st

def page_summary_body():

    st.write("### Project Summary")

    st.info(
        f"**Project Dataset**\n"
        f"The dataset has information on 1,460 house records from Ames, Iowa"
        f"It has sale price information for houses built between 1872 and 2010"
        f"The data is sourced from Kaggle"
    )

    st.write(
        f"For additional information, please visit and read the "
        f"Project readme file."
    )

    st.success(
        f"The project has 2 business requirements:\n"
        f"1. The client is interested in discovering how the house attributes correlate with sale price\n\n"
        f"2. The client is interested in predicting house sale price from"
        f"her four inherited houses and any other house in Ames, Iowa"
    )