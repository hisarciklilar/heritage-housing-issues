import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_house_price_data():
    df=pd.read_csv('outputs/datasets/collection/house_prices_records.csv')
    return df