# House Price Prediction using Machine Learning

## Project Overview
This project is an end-to-end Machine Learning pipeline that predicts house prices based on various housing features (such as area, number of bedrooms, bathrooms, and additional amenities). The goal is to identify the most critical factors influencing property value using regression techniques.

## Project Structure
- `analysis.ipynb`: The main Jupyter Notebook containing the data exploration, preprocessing, model training (Linear Regression and Random Forest), evaluation, and visualizations.
- `summary.docx`: A business-ready summary report detailing the findings, methodology, and recommendations for stakeholders.
- `Housing.csv`: The primary dataset containing 545 samples and 13 property features.
- `charts/`: A directory containing all the generated high-quality PNG visualizations from the analysis:
  - `actual_vs_predicted.png`
  - `correlation_heatmap.png`
  - `feature_importance.png`
  - `price_distribution.png`
  - `price_vs_area.png`

## Dataset Description
The dataset used is the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) by Yasser H.
- **Target Variable**: `price` (The sale price of the house)
- **Features**: `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`.

## Key Insights
1. **Area** is the single most dominant factor influencing the price of a house.
2. Amenities like **bathrooms** and **air conditioning** add a significant premium to the property value.
3. The **Random Forest Regressor** outperformed the baseline Linear Regression model, demonstrating that housing market features often share non-linear relationships.

## How to Use
1. Install the required Python dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Launch Jupyter Notebook and open `analysis.ipynb`:
   ```bash
   jupyter notebook analysis.ipynb
   ```
3. Run the cells sequentially to replicate the data cleaning, model training, and chart generation.

## Code Quality
The codebase follows standard PEP-8 conventions, is well-commented, and ensures reproducible results by utilizing static random seeds (e.g., `random_state=42`).
