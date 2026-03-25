# Matplotlib Basics 

This file contains basic Matplotlib operations with simple examples for beginners.


# Importing Library

```import matplotlib.pyplot as plt```  # matplotlib plotting library
```import numpy as np ```              # used for generating sample data


# Simple Line Plot

```x = [1, 2, 3, 4, 5]```              # x-axis values
```y = [10, 20, 15, 25, 30]```         # y-axis values

```plt.plot(x, y) ```                  # create line plot
```plt.show()```                       # display graph


# Adding Title and Labels

```x = [1, 2, 3, 4, 5]```
```y = [10, 20, 15, 25, 30]```

```plt.plot(x, y)```

```plt.title("Simple Line Graph")```   # graph title
```plt.xlabel("X Axis") ```            # x-axis label
```plt.ylabel("Y Axis")  ```           # y-axis label

```plt.show()```


# Styling the Line

```x = [1, 2, 3, 4, 5]```
```y = [10, 20, 15, 25, 30]```

```plt.plot(x, y, color='red', linestyle='--', marker='o')``` 
```color = line color```
```linestyle = dashed line```
```marker = circle points```

```plt.show()```



# Bar Chart

```categories = ["A", "B", "C"] ```    # categories
```values = [10, 20, 15] ```           # values

```plt.bar(categories, values) ```     # create bar chart
```plt.show()```


# Scatter Plot

```x = [1, 2, 3, 4, 5]```
```y = [10, 20, 15, 25, 30]```

```plt.scatter(x, y)  ```              # scatter plot (points only)
```plt.show()```


# Multiple Lines in One Graph

```x = [1, 2, 3, 4, 5]```
```y1 = [10, 20, 15, 25, 30]```
```y2 = [5, 15, 10, 20, 25]```

```plt.plot(x, y1, label="Line 1")```  # first line
```plt.plot(x, y2, label="Line 2")``` # second line

```plt.legend() ```                    # show legend
```plt.show()```


# Histogram

```data = np.random.randn(1000) ```    # generate random data

```plt.hist(data, bins=30)  ```        # histogram with 30 bins
```plt.show()```


# Subplots (Multiple graphs in one figure)

```x = [1, 2, 3, 4, 5]```
```y = [10, 20, 15, 25, 30]```

```plt.subplot(1, 2, 1)  ```           # 1 row, 2 columns, position 1
```plt.plot(x, y)```

```plt.subplot(1, 2, 2) ```            # position 2
```plt.bar(x, y)```

```plt.show()```


# Saving Figure

```x = [1, 2, 3]```
```y = [10, 20, 15]```

```plt.plot(x, y)```

```plt.savefig("plot.png") ```         # saves image as file
```plt.show()```


# Important Tips

# Always call plt.show() to display graph

# Use labels and title to make graphs understandable

# Use legend() when plotting multiple lines

# Matplotlib is mainly used for data visualization
