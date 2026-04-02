from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineerHPData(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for the housing price pipeline.

    Parameters
    ----------
    lotarea_large_q : float, default=0.99
        Quantile used to define the threshold for large lot areas.

    lotarea_small_q : float, default=0.01
        Quantile used to define the threshold for small lot areas.

    bedroom_impute_strategy : {"mean", "mode"}, default="mean"
        Strategy used to impute missing BedroomAbvGr values.
    """

    def __init__(
        self,
        lotarea_large_q=0.99,
        lotarea_small_q=0.01,
        bedroom_impute_strategy="mean",
    ):
        self.lotarea_large_q = lotarea_large_q
        self.lotarea_small_q = lotarea_small_q
        self.bedroom_impute_strategy = bedroom_impute_strategy

    def fit(self, X, y=None):
        X = X.copy()

        # LotArea thresholds learned from training data
        self.large_lot_threshold_ = X["LotArea"].quantile(self.lotarea_large_q)
        self.small_lot_threshold_ = X["LotArea"].quantile(self.lotarea_small_q)

        # Modes learned from training data
        self.bsmtfintype1_mode_ = X["BsmtFinType1"].mode(dropna=True)[0]
        self.bedroom_mode_ = X["BedroomAbvGr"].mode(dropna=True)[0]

        # Mean learned from training data
        self.bedroom_mean_ = X["BedroomAbvGr"].mean()

        if self.bedroom_impute_strategy == "mean":
            self.bedroom_fill_value_ = self.bedroom_mean_
        elif self.bedroom_impute_strategy == "mode":
            self.bedroom_fill_value_ = self.bedroom_mode_
        else:
            raise ValueError(
                "bedroom_impute_strategy must be 'mean' or 'mode'"
            )

        return self

    def transform(self, X):
        X = X.copy()

        # =============================
        # Create missingness indicators
        # =============================
        # Create missingness indicators before imputations
        X["MissingLotFrontage"] = X["LotFrontage"].isna().astype(int)
        X["MissingMasVnrArea"] = X["MasVnrArea"].isna().astype(int)
        X["MissingGarageYrBlt"] = X["GarageYrBlt"].isna().astype(int)
        X["MissingBedroomAbvGr"] = X["BedroomAbvGr"].isna().astype(int)

        # This is updated later if BsmtFinType1 remains missing after rule-based fills
        X["MissingBsmtFinType1"] = 0

        # ===============================
        # Replace missing cells with zero
        # ===============================
        X["2ndFlrSF"] = X["2ndFlrSF"].fillna(0)
        X["LotFrontage"] = X["LotFrontage"].fillna(0)
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)
        X["EnclosedPorch"] = X["EnclosedPorch"].fillna(0)
        X["WoodDeckSF"] = X["WoodDeckSF"].fillna(0)

        # ============================
        # Impute missing BedroomAbvGr
        # ============================
        X["BedroomAbvGr"] = X["BedroomAbvGr"].fillna(self.bedroom_fill_value_)

        # ======================================
        # Create binary house feature indicators
        # ======================================
        X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasExtraLivArea"] = (
            X["GrLivArea"] > (X["1stFlrSF"] + X["2ndFlrSF"])
        ).astype(int)
        X["HasBasement"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["HasBsmtFin"] = (X["BsmtFinSF1"] > 0).astype(int)
        X["HasBsmtUnf"] = (X["BsmtUnfSF"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasMasVnr"] = (X["MasVnrArea"] > 0).astype(int)
        X["HasEnclosedPorch"] = (X["EnclosedPorch"] > 0).astype(int)
        X["HasOpenPorch"] = (X["OpenPorchSF"] > 0).astype(int)
        X["HasWoodDeck"] = (X["WoodDeckSF"] > 0).astype(int)
        X["BuiltPre1950"] = (X["YearBuilt"] < 1950).astype(int)

        # ===========================================
        # Create large/small lot size dummy variables
        # ===========================================
        X["HasLargeLotArea"] = (X["LotArea"] > self.large_lot_threshold_).astype(int)
        X["HasSmallLotArea"] = (X["LotArea"] < self.small_lot_threshold_).astype(int)

        # =================================
        # Impute missing basement variables
        # =================================
        X.loc[
            (X["BsmtExposure"].isna()) & (X["TotalBsmtSF"] == 0),
            "BsmtExposure"
        ] = "No_basement"

        X.loc[
            (X["BsmtExposure"].isna()) & (X["TotalBsmtSF"] > 0),
            "BsmtExposure"
        ] = "No"

        X.loc[
            (X["BsmtFinType1"].isna()) & (X["TotalBsmtSF"] == 0),
            "BsmtFinType1"
        ] = "No_basement"

        X.loc[
            (X["BsmtFinType1"].isna()) &
            (X["BsmtUnfSF"] > 0) &
            (X["BsmtFinSF1"] == 0),
            "BsmtFinType1"
        ] = "Unf"

        X.loc[X["BsmtFinType1"].isna(), "MissingBsmtFinType1"] = 1
        X.loc[X["BsmtFinType1"].isna(), "BsmtFinType1"] = self.bsmtfintype1_mode_

        # ============================
        # Impute missing garage finish
        # ============================
        X.loc[
            (X["GarageFinish"].isna()) & (X["GarageArea"] == 0),
            "GarageFinish"
        ] = "No_garage"

        X.loc[
            (X["GarageFinish"].isna()) & (X["GarageArea"] > 0),
            "GarageFinish"
        ] = "Missing"

        # ==============================
        # Create new feature (numerical)
        # ==============================
        X["TotalPorchSF"] = X["EnclosedPorch"] + X["OpenPorchSF"]

        return X