import streamlit as st
import json
from pathlib import Path
import joblib


def load_model_performance_metrics():
    metrics_path = Path("outputs/metrics/model_performance_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        return metrics
   
def page_technical_details_body():
    st.title("Technical Details")

    st.markdown("""
    This page explains the technical details behind the house price prediction business case.
    The modelling is based on the Ames House Prices dataset and uses a machine learning pipeline
    to estimate house sale prices from given house characteristics.
    """)

    st.header("1. Data Source")
    st.write("""
    The project uses housing data based on the Ames House Prices dataset downloaded from Code Institute Kaggle page.
    The dataset contains information on various house characteristics such as floor area,
    basement size, garage features, construction year, quality ratings, etc.
    """)

    st.header("2. Data Preparation and Feature Engineering")
    st.write("""
    Before modelling, the data were cleaned and prepared.
    This included:
    - checking variable types
    - identifying missing values
    - distinguishing true missing values from structural absence
      (for example, no garage or no basement) and deciding on an appropriate coding strategy
    - preparing numerical and categorical variables for modelling
    - creating additional features to better capture housing characteristics 
            and relationships with sale price
             
    The transformations and feature engineering steps for each indicator are summarised in the table below.
    """)

    st.markdown("""
            | Variable        | Action                                                            |
            |-----------------|-------------------------------------------------------------------|
            | `SalePrice`     | Logarithmic transformation                                        |
            | `1stFlrSF`      | Logarithmic transformation                                        |
            | `2ndFlrSF`      | (1) Replace missing with zero                                     |
            |                 | (2) Create `Has2ndFlr`                                            |
            |                 | (3) Create `HasExtraLivArea`                                      |
            | `GrLivArea`     | Logarithmic transformation                                        |
            | `BsmtFinSF1`    | (1) Log1p transformation                                          |
            |                 | (2) Create `HasBsmtFin`                                       |  
            | `BsmtUnfSF`     | (1) Log1p transformation                                          |
            |                 | (2) Create `HasBsmtUnf`                                           | 
            | `BsmtFinType1`  | (1) Replace missing with "No_basement" if TotalBsmtSF==0          |
            |                 | (2) Replace missing with "Unf" if `BsmtUnfSF>0` & `BsmtFinSF1==0` |
            |                 | (3) Create `MissingBsmtFinType1` variable                         |
            |                 | (4) Replace remaining missing with mode                           |
            |                 | (5) Create set of dummies based on categories                     |
            | `BsmtExposure`  | (1) Replace missing with "No_basement" if TotalBsmtSF==0          |
            |                 | (2) Replace remaining missing with "No" if TotalBsmtSF>0          | 
            |                 | (3) Create set of dummies based on categories                     |
            | `TotalBsmtSF`   | (1) Log1p transformation                                          |
            |                 | (2) Create `HasBasement`                                          |
            | `LotFrontage`   | (1) Create `MissingLotFrontage` variable                          |
            |                 | (2) Replace missing with zero                                     |
            |                 | (3) Log1p transformation                                          |
            | `LotArea`       | (1) Logarithmic transformation                                    |
            |                 | (2) Create `HasLargeLotArea`                                      |
            |                 | (3) Create `HasSmallLotArea`                                      |
            | `BedroomAbvGr`  | (1) Create `MissingBedroomAbvGr` variable                         |
            |                 | (2a) Replace missing with mean; Substitute: Impute with mode      |
            |                 | (2b) Replace missing with mode; Substitute: Impute with mean      |
            | `GarageArea`    | (1) Create `HasGarage`                                            |
            |                 | (2) Log1p transformation                                          |
            | `GarageFinish`  | (1) Replace missing with "No_garage" if `GarageArea`==0           |
            |                 | (2) Replace remaining missing with "Missing"                      |
            |                 | (3) Create set of dummies based on categories                     |
            | `MasVnrArea`    | (1) Create `MissingMasVnrArea` variable                           |
            |                 | (2) Replace missing with zero                                     |
            |                 | (3) Log1p transformation                                          |
            |                 | (4) Create `HasMasVnr`                                            |
            | `GarageYrBlt`   | (1) Create `MissingGarageYrBlt`                                   |
            |                 | (2) Replace missing with zero                                     |
            | `EnclosedPorch` | (1) Replace missing with zero                                     |
            |                 | (2) Create `TotalPorch` = `EnclosedPorch` + `OpenPorchSF`         |
            |                 | (3) Create `HasEnclosedPorch`                                     |
            | `OpenPorchSF`   | Create `HasOpenPorch`                                             |
            | `KitchenQual`   | Create set of dummies based on categories                         |
            | `WoodDeckSF`    | (1) Replace missing with zero                                     |
            |                 | (2) Create `HasWoodDeck`                                          |
            | `OverallCond`   | No change; include as it is provided in data                      |
            | `OverallQual`   | No change; include as it is provided in data                      |
            | `YearBuilt`     | No change; include as it is provided in data                      |
            |                 | Create `BuiltPre1950` for truncation of `YearRemodAdd` at 1950    |
            | `YearRemodAdd`  | Include together with `BuiltPre1950`                              |
    """)

    st.markdown("""
     These transformations and feature engineering steps were designed to:
     - address skewness in numerical variables through logarithmic transformations
     - handle missing values in a way that preserves information and observations
     - create new features to better capture housing characteristics and relationships with sale price
     """)


    st.header("3. Model Pipeline")
    st.write("""
    After splitting data into train and test samples, the model to be used for
             predictions is chosen based on model fit and predictive
             performance comparisons from estimation of Linear Regression,
             Ridge Regression, and Random Forest. Hyperparameter tuning is
             performed for the latter two approaches to optimise performance.
             Random Forest is selected as the final model based on its superior performance.             
             The prediction workflow is automated through a machine learning pipeline.
    """)

    pipeline = joblib.load("outputs/models/house_price_pipeline.joblib")

    st.subheader("Pipeline Steps")

    for step_name, step_obj in pipeline.steps:
        st.write(f"**{step_name}**: {step_obj.__class__.__name__}")

    with st.expander("Show full pipeline object"):
        st.write(pipeline)

    st.header("4. Model Performance")

    st.write("""Target variable: logarithmic transformation of house sale price""")

    metrics = load_model_performance_metrics()

    st.subheader("Train Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{metrics['train']['R2']:.3f}")
    col2.metric("RMSE", f"{metrics['train']['RMSE']:,.4f}")
    col3.metric("MAE", f"{metrics['train']['MAE']:,.4f}")

    st.subheader("Test Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{metrics['test']['R2']:.3f}")
    col2.metric("RMSE", f"{metrics['test']['RMSE']:,.4f}")
    col3.metric("MAE", f"{metrics['test']['MAE']:,.4f}")

    st.header("5. Limitations")
    st.write("""
    The predictions are based on patterns in the available dataset.
    They should not be interpreted as guaranteed market values.
    Only a limited number of models were compared for this project.
    Model performance may improve after a more careful examination of outliers,
    and integrating some non-linearities or interactions between house features. 
    Alternative modelling strategies may also be adopted.
    """)