import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_summary import page_summary_body
from app_pages.page_house_price_study import page_house_price_study_body

app = MultiPage(app_name="House Prices")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Explore House Prices", page_house_price_study_body)
app.run()