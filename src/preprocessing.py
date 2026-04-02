import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def create_preprocessor(
    log_vars,
    log1p_vars,
    numeric_and_binary_vars,
    categorical_vars,
):
    """
    Create the preprocessing ColumnTransformer for the housing pipeline.

    Parameters
    ----------
    log_vars : list
        Variables to be median-imputed and log-transformed with np.log.

    log1p_vars : list
        Variables to be median-imputed and transformed with np.log1p.

    numeric_and_binary_vars : list
        Numeric and binary variables to be median-imputed only.

    categorical_vars : list
        Categorical variables to be imputed with most frequent category
        and one-hot encoded.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer ready to be used inside a pipeline.
    """

    log_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
    ])

    log1p_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("log", log_transformer, log_vars),
            ("log1p", log1p_transformer, log1p_vars),
            ("num", numeric_transformer, numeric_and_binary_vars),
            ("cat", categorical_transformer, categorical_vars),
        ],
        remainder="drop",
    )

    return preprocessor
