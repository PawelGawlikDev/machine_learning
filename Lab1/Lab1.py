# %%
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# %%
# Fetch dataset
raisin = fetch_ucirepo(id=850)
# data (as pandas dataframes)
X = raisin.data.features
y = raisin.data.targets

# Class description
print(y['Class'].value_counts())

# Description of features
X.describe()

# %%
# Minimum values
min_values = X.min()

# Maximum values
max_values = X.max()

# Mean
mean_values = X.mean()

# Median
median_values = X.median()

# Second (lower) quartile
q2 = X.quantile(0.25)

# Third (upper) quartile
q3 = X.quantile(0.75)

# Standard deviation
std_deviation = X.std()

# The number of samples in the set
sample_count = len(X)

# Occurrence of incomplete data
missing_data_count = X.isnull().sum()

# %%
# Sample counts for each class (if data is classified)
for column in X.columns:
    class_counts = X[column].value_counts()
    print("\nLiczność próbek dla poszczególnych klas:")
    print(class_counts)

# %%
# Display the results
print("Minimum values:")
print(min_values)
print("\nMaximum values:")
print(max_values)
print("\nMean:")
print(mean_values)
print("\nMedian:")
print(median_values)
print("\nSecond (lower) quartile:")
print(q2)
print("\nThird (upper) quartile:")
print(q3)
print("\nStandard deviation:")
print(std_deviation)
print("\nThe number of samples in the set:", sample_count)
print("\nOccurrence of incomplete data:")
print(missing_data_count)

# %%
# Graphs for three selected variables
selected_columns = X.columns

for column in selected_columns:
    # Boxplot
    plt.figure(figsize=(8, 6))
    X.boxplot(column=column)
    plt.title(f'Boxplot for {column}')
    plt.show()

    # Line graph
    plt.figure(figsize=(8, 6))
    plt.plot(X.index, X[column])
    plt.title(f'Line graph for {column}')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 6))
    X[column].plot(kind='hist', bins=20)
    plt.title(f'Histogram for {column}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
