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

## Hypothesis and how to validate?

* Project hypothesis(es) and how they will ve validated will be added in the future version of the project.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

The project follows a structured analytical approach to explore the relationships between house characteristics and sale prices and develop a Machine Learning pipeline to predict house prices with given features.

### Data Collection

This stage includes the following:

* Download the data using the Kaggle API
* Develop an understanding of the features provided in the data, how they are measured, and the possible values they can get by checking the information provided by the data source, and looking at the data cells at the few top and bottom rows of the data.

### Exploratory Data Analysis

Exploratory data analysis is provided under the "Explore HousePrices" page. 

Exploratory analysis of the data includes strategies to develop an understanding of the data and the information it provides. The main steps at this stage are:

* Checking the degree of missingness in the data, consideration of the possible reasons behind the missing observations (for example, the second floor area in square feet is likely to be missing for houses that do not have a second floor) and whether these can be imputed using the existing information.
* For features measured in numerical scale, examining the descriptive statistics to have a preliminary understanding of the range of values that each feature takes, their central location and the variation they depict.
* For features measured in categorical scale, examining the frequency distribution for each of the possible outcomes.
* Visualisation of features using a plot style appropriate to the level of measurement. Box-Whisker plots and histograms are used for the features measured in numerical scale and bar charts are used for features measured by categorical scale.

### Correlation Analysis

Correlation analysis is provided under the "Correlate House Prices with Features" page.

The correlation analysis looks at the strength of the relationship between house prices and the house features and sets the expectations about which features has greater or weaker effect on the house sale price. The analysis conducted here follows the steps below:

* Calculation and ranking of correlation coefficient between the house price and each of the numerical features. In absoluter terms, the closer the value of the correlation coefficient to 1, the stronger the linear relationship between house price and the respective feature while the closer the correlation coefficient to zero, the weaker the linear relationship between them.
* Visual demonstration of correlation coefficients between each of the numerical variables in the data through a heatmap. The heatmap colors the positive (same direction) correlations in red shades and the negative (opposite direction) correlations in blue shades. The stronger the degree of correlation, the darker the color is.
* Visual representation of the degree of correlation between house price and a selected set of features that appeared to have strong correlation. Scatter plots are used for the numerical features and box-plots of house price for each potential outcome are used for the categorical features.

### Data Transformations

There is information about 23 features in the dataset.

* In addition to `1stFlrSF` and `2ndFlrSF`, there is information on `GrLiveArea`. Except 64 observations, 

### Modelling

The features 

## ML Business Case

* To be added in the future version of the project.

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
