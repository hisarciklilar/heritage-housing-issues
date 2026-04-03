# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Analysis of the Housing Market in Ames, Iowa

## Dataset Content

* The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above ground (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above ground living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate

#### Overall Quality

H1: There is a strong positive relationship between overall house quality (`OverallQual`) and sale price.

#### Kitchen Quality

H2: There is a positive relationship between kitchen quality (`KitchenQual`) and sale price.

#### Living Area

H3: There is a positive relationship between above-ground living area (`GrLivArea`) and sale price.

#### Garage Area
H4: There is a positive relationship between garage area (`GarageArea`) and sale price.

#### Basement Area

H5: There is a positive relationship between basement area (`TotalBsmtSF`) and sale price.

The hypotheses listed above include a selected group of house characteristics that are expected to be highly correlated with sale price. The hypotheses are about the direction and strength of the relationship between each indicator and house sale price. These can be confirmed or rejected by a combination of the following tools:

- Graphical representation of the relationship between each indicator and sales price. A scatterplot provides a very informative illustration for the numerical variables while box-plot of sales price for ordered possible outcomes of the ordinal categorical variable `KitchenQual` shows how price increases with quality rating.
- Calculation of correlation coefficient between each numerical indicator and sale price. The four numerical indicators of `OverallQual`, `GrLivArea`, `GarageArea`, and `TotalBsmtSF` are ranked as indicators with the highest correlation coefficient.
- Coefficient estimates and statistical significance tests based on Linear Regression (not reported for this project because Random Forest was selected and reported as the best performing method during comparisons of R2 score, RMSE and MAE)
- Specific checks designed for Random Forest estimation, such as assessment of the predictive performance of each indicator.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

The project follows a structured analytical approach to explore the relationships between house characteristics and sale prices and develop a Machine Learning pipeline to predict house prices with given features.

### Data Collection

Data collection is provided under the `01_Data_Collection` Jupyter notebook.

This stage includes the following:

* Download the data using the Kaggle API
* Develop an understanding of the features provided in the data, how they are measured, and the possible values they can get by checking the information provided by the data source, and looking at the data cells at the few top and bottom rows of the data.

### Exploratory Data Analysis

Exploratory data analysis is provided under the `02_Exploratory_Data_Analysis` Jupyter notebook. This notebook mainly addresses the first Business Requirement of discovering how house attributes correlate with house price and fulfills the client's expectation of seeing data visualisations of the correlated variables against the sale price.

Exploratory analysis of the data includes strategies to develop an understanding of the data and the information it provides. The main steps at this stage are:

* Checking the degree of missingness in the data, consideration of the possible reasons behind the missing observations (for example, the second floor area in square feet is likely to be missing for houses that do not have a second floor) and whether these can be imputed using the existing information.
* For features measured in numerical scale, examining the descriptive statistics to have a preliminary understanding of the range of values that each feature takes, their central location and the variation they depict.
* For features measured in categorical scale, examining the frequency distribution for each of the possible outcomes.
* Visualisation of features using a plot style appropriate to the level of measurement. Box-Whisker plots and histograms are used for the features measured in numerical scale and bar charts are used for features measured by categorical scale.

### Correlation Analysis

Correlation analysis is provided under the `03_Correlation_Analysis` Jupyter notebook.

The correlation analysis looks at the strength of the relationship between house prices and the house features and sets the expectations about which features has greater or weaker effect on the house sale price. The analysis conducted here follows the steps below:

* Calculation and ranking of correlation coefficient between the house price and each of the numerical features. In absoluter terms, the closer the value of the correlation coefficient to 1, the stronger the linear relationship between house price and the respective feature while the closer the correlation coefficient to zero, the weaker the linear relationship between them.
* Visual demonstration of correlation coefficients between each of the numerical variables in the data through a heatmap. The heatmap colors the positive (same direction) correlations in red shades and the negative (opposite direction) correlations in blue shades. The stronger the degree of correlation, the darker the color is.
* Visual representation of the degree of correlation between house price and a selected set of features that appeared to have strong correlation. Scatter plots are used for the numerical features and box-plots of house price for each potential outcome are used for the categorical features.

### EDA and Feature Engineering Design

A detailed exploratory data analysis and feature engineering design is provided under `04_EDA_and_Feature_Engineering_Design` Jupyter notebook. 

Exploratory Data Analysis in this notebook is taken to:

* Understand the missingness patterns in the data, while differentiating structural missing observations from random ones.
* Decide on appropriate imputation approach based on the nature of missingness. For example, assign 'No_basement' to basement exposure (`BsmtExposure`) when total basement area is zero (`TotalBsmtSF`==0) or assign mean / median / mode value for indicators when there is random missingness.
* Create binary dummies for cases where imputations were done for random missingness.
* Where necessary, apply appropriate transformations to the features. For example, using logarithmic transformation for numerical features with a skewed distribution while doing this operation by adding 1 to the values of the indicators when there are zero values (log1p transformation).
* Convert multinomial categorical variables into sets of dummies.
* Create dummies based on outliers.

Decisions on transformations and feature engineering steps are summarised in the table below:

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

Table below also provide supporting information about the feature engineering decisions for the relevant sets of indicators:

#### Logarithmic Transformations

| Variable    | Notes                   |
|-------------|-------------------------|
| `SalePrice` | No missing observations |
| `1stFlrSF`  | No missing observations |
| `GrLivArea` | No missing observations |
| `LotArea`   | No missing observations |

### Logarithmic Transformations After adding One

| Variable      | Notes                                                                          |
|---------------|--------------------------------------------------------------------------------|
| `BsmtFinSF1`  | No missing observations; Complement: `BsmtUnfSF`; Substitute: `TotalBsmtSF`    |
| `BsmtUnfSF`   | No missing observations; Complement: `BsmtFinSF`; Substitute: `TotalBsmtSF`    |
| `TotalBsmtSF` | No missing observations; Substitute: `BsmtFinSF1` and `BsmtUnfSF`              |
| `GarageArea`  | No missing observations                                                        |
| `LotFrontage` | Pre-action: Create `MissingLotFrontage`; Pre-action: replace missing with zero |
| `MasVnrArea`  | Pre-action: Create `MissingMasVnrArea`; Pre-action: Replace missing with zero  |

#### Replace missing with zero

| Variable        | Notes                                                                                                 |
|-----------------|-------------------------------------------------------------------------------------------------------|
| `2ndFlrSF`      | Post-action: Create `Has2ndFlr` dummy                                                                 |
| `GarageYrBlt`   | Replace missing with zero; Pre-action: Create `MissingGarageYrBlt`                                    |
| `LotFrontage`   | Replace missing with zero; Pre-action: Create `MissingLotFrontage`; Post-action: Take log1p           |
| `MasVnrArea`    | Replace missing with zero; Pre-action: Create `MissingMasVnrArea`; Post-action: Take log1p            |
| `EnclosedPorch` | Replace missing with zero; Post-action: Create `TotalPorchSF`; Post-action: Create `HasEnclosedPorch` |
| `WoodDeckSF`    | Replace missing with zero; Post-action: Create `HasWoodDeck`                                          |

#### Imputation for missing cells (structural missing)

| Variable       | Notes                                                                                                    |
|----------------|----------------------------------------------------------------------------------------------------------|
| `BedroomAbvGr` | Impute with mean; Substutute: Impute with Mode; Pre-action: Create `MissingBedroomAbvGr`                 |
|                | Impute with mode; Substutute: Impute with mean; Pre-action: Create `MissingBedroomAbvGr`                 |
| `BsmtExposure` | Replace missing with "No_basement" if TotalBsmtSf==0                                                     |
|                | Replace remaining missing with "No" if TotalBsmtSF>0; Post-action: Dummyfy                               | 
| `BsmtFinType1` | Replace missing with "No_basement" if TotalBsmtSf==0                                                     |
|                | Replace missing with "Unf" if `BsmtUnfSF>0` & `BsmtFinSF1`==0; Post-action: Create `MissingBsmtFinType1` | 
|                | Replace remaining missing with Mode; Pre-action: Create `MissingBsmtFinType1`                            |
| `GarageFinish` | Replace missing with "No_garage"  if `GarageArea`==0                                                     |
|                | Replace remaining missing with "Missing"; Post-action: Dummyfy                                           | 

#### Dummies for Missingness at Random

| Variable       | Variable Created      | Notes                                                                                                               |
|----------------|-----------------------|---------------------------------------------------------------------------------------------------------------------|
| `BedroomAbvGr` | `MissingBedroomAbvGr` | Post-action: Impute with mean; impute with mode                                                                     |
| `BsmtFinType1` | `MissingBsmtFinType1` | Pre-action: Replace structural missings; Post-action: Replace random missings with mode; Post-action: Dummyfy       |
| `LotFrontage`  | `MissingLotFrontage`  | Post-action: Replace missing with zero; Post-action: Take log1p                                                     |
| `MasVnrArea`   | `MissingMasVnrArea`   | Post-action: Replace missing with zero                                                                              |
| `GarageYrBlt`  | `MissingGarageYrBlt`  | Post-action: Replace missing with zero                                                                              |

#### Create dummy groups for multinomial variables

| Variable       | Variable Created                  | Notes                        |
|----------------|-----------------------------------|------------------------------|
| `BsmtExposure` | Set of dummies for `BsmtExposure` | Pre-action: Replace missings |
| `BsmtFinType1` | Set of dummies for `BsmtFinType1` | Pre-action: Replace structural missings; Pre-action: Create `MissingBsmtFinType1`; Pre-action: Replace random missings with mode |
| `GarageFinish` | Set of dummies for `GarageFinish` | Pre-action: Replace missings |
| `KitchenQual`  | Set of dummies for `KitchenQual`  |                              |

#### Create new feature variable (numerical)

| New Variable   | Description                   | Notes                                                 |
|----------------|-------------------------------|-------------------------------------------------------|
| `TotalPorchSF` | `EnclosedPorch` + `OpenPorch` | Pre-action: Replace missing `EnclosedPorch` with zero |

#### Create new feature variable (categorical)

| New Variable       | Description                                             | Notes                                 |
|--------------------|---------------------------------------------------------|---------------------------------------|
| `Has2ndFlr`        | =1 if 2ndFlrSF>0; =0 otherwise                          | Pre-action: Replace missing `2ndFlrSF` with zero|
| `HasExtraLivArea`  | =1 if `GrLivArea` > (`1stFlrSF` + `2ndFlrSF`); =0 otherwise | Pre-action: Replace missing `2ndFlrSF` with zero |  
| `HasBasement`      | =1 if `TotalBsmtSF`>0; =0 otherwise                     |                                       |
| `HasBsmtFin`       | =1 if `BsmtFinSF1`>0; =0 otherwise                      |                                       |
| `HasBsmtUnf`       | =1 if `BsmtUnSF`>0; =0 otherwise                        |                                       |
| `HasGarage`        | =1 if `GarageArea`>0; =0 otherwise                      |                                       |
| `HasLargeLotArea`  | =1 if `LotArea`>`LotArea`.quantile(0.99); =0 otherwise  |                                       |
| `HasSmallLotArea`  | = 1 if `LotArea`<`LotArea`.quantile(0.01); =0 otherwise |                                       |
| `HasMasVnr`        | =1 if MasVnrArea>0; =0 otherwise                        |                                       |
| `HasEnclosedPorch` | =1 if `EnclosedPorch`>0; =0 otherwise                   | Pre-action: Replace missing with zero |
| `HasOpenPorch`     | =1 if `OpenPorchSF`>0; =0 otherwise                     |                                       | 
| `HasWoodDeck`      | =1 if `WoodDeckSF`>0; =0 otherwise                      | Pre-action: Replace missing with zero |
| `BuiltPre1950`     | =1 if `YearBuilt`<1950; =0 otherwise                    |                                       |

#### No change variables

| Variable       |
| `OverallCond`  |
| `OverallQual`  |
| `YearBuilt`    |
| `YearRemodAdd` |

### Modelling and ML Pipeline

Application of Feature Engineering and ML Pipeline development are provided under `05_Modelling_and_ML_Pipeline` Jupyter Notebook.

* The feature engineering design decisions are applied to the data
* Data is split into train and test samples
* Three alternative predictive modelling approaches are applied to the test sample:
  * Linear Regression
  * Ridge Regression
  * Random Forest
* Hyperparameter tuning is applied on Ridge Regression and Random Forest
* Fit and predictive performance of models are compared and evaluated in terms of the following indicators for both the train and test samples:
  * R2 score
  * RMSE
  * MAE
* Scatterplots showing the actual and predicted logatihmic sale prices are provided for both the train and test samples. A $45^o$ line is added to the plots to help with visual assessment of the fit.
* **All models** estimated yielded an R2 above 0.80 (rounded to two decimal points), which is comfortably **above the 0.75 threshold** agreed with the customer.
* **Random Forest** after hyperparameter tuning is demonstrated to provide the best fit with the highest predictive power with an **R2 score value of 0.91 for the train and 0.80 for the test samples**.

### Inherited House Price Predictions

Inherited house price predictions are provided under `06_Inherited_House_Price_Predictions` Jupyter notebook.

* Sale price predictions for each of the four inherited houses are provided. The models are run for logarithmic sale prices, so the model predictions are converted back into levels using an anti-logarithmic transformation.
* The total value of the four inherited houses are calculated by summing the predicted value (i.e. sale price) of each property.

## ML Business Case

The customer wishes to (i) maximise the sale price of their four inherited properties, (ii) be able to predict the market value of a property with given characteristics.

## Dashboard Design

### Page 1: Quick Project Summary

* Quick project summary
  * Describe project dataset
  * State business requirements

### Page 2: Explore House Prices

* Exploratory analysis of the housing market in Ames, Iowa
  * Overview of house prices, attributes, and missingness
  * Distribution of variables in data. Box plot and histogram are provided for numerical indicators and a bar plot for categorical ones.
  * Summary statistics for variables in data

### Page 3: Correlate House Price with Features

* Correlation analysis of the house prices with house features
  * Correlation coefficients of house sale price with house features
  * Correlation heatmap of house sale price with features
  * Visualisation of the relationship between house prices and a selected group of indicators. Scatterplots are provided for continuous indicators and box plots of house price for each potential outcome for categorical ones.
  * Conclusions from the correlation analysis. 

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://house-prices-3badc716b127.herokuapp.com>
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. You may also demove the redundant files from the repository. 

## Main Data Analysis and Machine Learning Libraries

This project uses the following Python libraries for data handling, visualisation, machine-learning pipeline creation, and interactive web application development.

### **Streamlit**

**Purpose:** Building the interactive web application and user interface.

Streamlit is used for:

* Creating the app layout and pages
* Displaying interactive plots and tables
* Collecting user inputs (e.g. feature selection, filters)

**Example:**

```python
import streamlit as st

st.write("## Exploratory Analysis of the Housing Market in Ames, Iowa")

st.info(
        f"* This section provides an understanding of house attributes and the sale price\n"
        f"in the Ames, Iowa housing market"
    )

if st.checkbox("Inspect house prices and attributes"):
        st.write(
            f"* The dataset has {df.shape[0]} observations and {df.shape[1]} variables, "
            f"find below the first 10 rows."
        )
```

### **NumPy**

**Purpose:** Numerical computations and array-based operations.

NumPy is used for:

* Efficient numerical transformations
* Creating derived variables
* Handling missing values and conditional logic

**Example:**

```python
import numpy as np

df["LogSalePrice"] = np.log(df["SalePrice"])
```

### **Matplotlib**

**Purpose:** Plotting library used for custom visualisations.

Matplotlib is used for:

* Creating scatter plots, bar charts, and box plots
* Fine-grained control over axes, labels, and layout
* Serving as the base for Seaborn plots

**Example:**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.histplot(df['SalePrice'], kde=True)
plt.title(f"Distribution of Sale Price")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

### **Seaborn**

**Purpose:** Statistical data visualisation built on top of Matplotlib.

Seaborn is used for:

* Creating aesthetically pleasing statistical plots
* Visualising distributions and relationships between variables
* Generating boxplots and categorical comparisons

**Example:**

```python
import seaborn as sns

fig, ax = plt.subplots()
sns.boxplot(
    data=df,
    x="KitchenQual",
    y="SalePrice",
    order=["Po", "Fa", "TA", "Gd", "Ex"],
    ax=ax
)

ax.set_xlabel("Kitchen Quality")
ax.set_ylabel("Sale Price")

st.pyplot(fig)
```

## Credits

* Contents of `multipage.py` are copied from Code Instiude Walkthrough Project 2: Churnometer files
* `page_summary.py` is also a modified version of the example provided by Code Institute in the Walkthrough Project 2: Churnometer files
