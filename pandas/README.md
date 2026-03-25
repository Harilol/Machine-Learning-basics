# Pandas Basics 

This file contains basic Pandas operations with simple examples for beginners.



# Importing Pandas

```import pandas as pd```



# Creating DataFrame

```data = {```
   ``` "Name": ["Hari", "John", "Alice"],```
   ``` "Age": [21, 22, 23],```
 ```   "Salary": [50000, 60000, 70000]```
```}```

```df = pd.DataFrame(data)```
```print(df)```



# Viewing Data

```print(df.head())```     # first 5 rows
```print(df.tail())```     # last 5 rows
```print(df.shape)```      # (rows, columns)
```print(df.columns) ```   # column names
```print(df.info())```     # summary



# Selecting Data

```print(df["Name"])  ```               # single column
```print(df[["Name", "Age"]])```        # multiple columns
```print(df.iloc[0])```                 # first row (index based)
```print(df.loc[0])  ```                # first row (label based)



# Filtering Data

```print(df[df["Age"] > 21])|```



# Adding New Column

```df["Bonus"] = df["Salary"] * 0.1```
```print(df)```



# Updating Data

```df.loc[0, "Age"] = 25```
```print(df)```


# Deleting Column

```df = df.drop("Bonus", axis=1)```
```print(df)```


# Basic Operations

```print(df["Salary"].mean())```
```print(df["Salary"].max())```
```print(df["Salary"].min())```


# GroupBy

```print(df.groupby("Age")["Salary"].mean())```


# Sorting

```print(df.sort_values(by="Salary", ascending=False))```


# Handling Missing Values

```df.isnull() ```         # check nulls
```df.dropna()  ```        # remove nulls
```df.fillna(0)  ```       # replace nulls



# Reading CSV File

```df = pd.read_csv("data.csv")```


# Saving CSV File

```df.to_csv("output.csv", index=False)```


# Iterating Rows

```for index, row in df.iterrows():```
   ``` print(index, row["Name"])```
