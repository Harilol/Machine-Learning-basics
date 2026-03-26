# Seaborn Basics

# 1. Import Libraries
# seaborn for visualization
```import seaborn as sns```

# matplotlib to display plots
```import matplotlib.pyplot as plt```

# loads built-in dataset called "tips"
```df = sns.load_dataset("tips")```

# view first 5 rows
```print(df.head())```

# 3. Scatter Plot
# shows relationship between total bill and tip
```sns.scatterplot(x="total_bill", y="tip", data=df)```
```plt.show()```

# 4. Line Plot
# shows trend between variables
```sns.lineplot(x="total_bill", y="tip", data=df)```
```plt.show()```

# 5. Bar Plot
# compares average total bill across days
```sns.barplot(x="day", y="total_bill", data=df)```
```plt.show()```

# 6. Count Plot
# counts number of entries for each day
```sns.countplot(x="day", data=df)```
```plt.show()```

# 7. Box Plot
# shows distribution and outliers
```sns.boxplot(x="day", y="total_bill", data=df)```
```plt.show()```

# 8. Violin Plot
# shows distribution + density
```sns.violinplot(x="day", y="total_bill", data=df)```
```plt.show()```

# 9. Histogram
# shows frequency distribution
```sns.histplot(df["total_bill"], bins=10)```
```plt.show()```

# 10. KDE Plot
# smooth density curve
```sns.kdeplot(df["total_bill"])```
```plt.show()```

# 11. Pair Plot
# relationships between all numeric columns
```sns.pairplot(df)```
```plt.show()```

# 12. Heatmap (Correlation)
# compute correlation matrix
```corr = df.corr(numeric_only=True)```

# visualize correlation
```sns.heatmap(corr, annot=True, cmap="coolwarm")```
```plt.show()```

# 13. Styling
# set background style
```sns.set_style("darkgrid")```

# 14. Hue Example
# adds extra category (sex) with colors
```sns.scatterplot(x="total_bill", y="tip", hue="sex", data=df)```
```plt.show()```
