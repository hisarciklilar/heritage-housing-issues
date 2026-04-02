import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_house_price_data():
    df=pd.read_csv('outputs/datasets/collection/house_prices_records.csv')
    return df

@st.cache_data
def load_inherited_house_price_data():
    df = pd.read_csv('outputs/datasets/collection/inherited_houses.csv')
    return df

@st.cache_data
def load_train_test_data():
    X_train = pd.read_csv("outputs/datasets/splits/X_train.csv")
    X_test = pd.read_csv("outputs/datasets/splits/X_test.csv")
    y_train = pd.read_csv("outputs/datasets/splits/y_train.csv")
    y_test = pd.read_csv("outputs/datasets/splits/y_test.csv")

    # Convert y to Series
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    return X_train, X_test, y_train, y_test
