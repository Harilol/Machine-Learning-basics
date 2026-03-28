#  Linear vs Polynomial Regression

# Objective
Compare Linear Regression and Polynomial Regression on a real dataset to understand how model complexity affects performance.

# Dataset
- California Housing dataset
- Features include income, house age, rooms, population, etc.
- Target: House Price

# Steps Performed

# 1. Data Loading
- Loaded dataset using sklearn
- Converted into pandas DataFrame

# 2. Exploratory Data Analysis (EDA)
- Checked shape, null values, statistics
- Visualized correlations using heatmap

# 3. Feature Engineering
- Created new feature:
  - `Rooms_per_House = AveRooms / HouseAge`

# Models Used

# Linear Regression (Baseline)
- Simple model assuming linear relationship

# Polynomial Regression
- Added polynomial features (degree = 2)
- Captures non-linear relationships

# Results

| Model                  | R² Score |
|------------------------|--------|
| Linear Regression      | 0.57   |
| Polynomial Regression  | 0.66   |

# Insights

- Polynomial Regression improved performance
- Indicates presence of non-linear relationships
- Higher complexity can improve accuracy but may risk overfitting

# Conclusion

- Linear Regression is simple and fast
- Polynomial Regression provides better flexibility
- Model selection depends on data pattern 
