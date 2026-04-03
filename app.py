import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_house_price_study import page_house_price_study_body
from app_pages.page_correlations import page_correlations
from app_pages.page_predictions import page_predictions_body
from app_pages.page_technical_details import page_technical_details_body
from app_pages.page_hypotheses import page_hypotheses_body

app = MultiPage(app_name="House Prices")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Explore House Prices", page_house_price_study_body)
app.add_page("Correlate House Price with Features", page_correlations)
app.add_page("Project Hypotheses", page_hypotheses_body)
app.add_page("Predict House Sale Prices", page_predictions_body)
app.add_page("Technical Details", page_technical_details_body)

app.run()
