

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1: Load and Explore the Dataset

# Step 1: Load the dataset
try:
    # Load the Iris dataset from a URL
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                       header=None, 
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
except FileNotFoundError:
    print("The dataset file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Step 2: Display the first few rows of the dataset
print(iris.head())

# Step 3: Explore the structure of the dataset
print(iris.info())
print(iris.isnull().sum())  # Check for missing values

# Step 4: Clean the dataset (if necessary)
# In this case, there are no missing values in the Iris dataset

# Question 2: Basic Data Analysis

# Step 5: Compute basic statistics
print(iris.describe())

# Step 6: Group by species and compute the mean of numerical columns
grouped_means = iris.groupby('species').mean()
print(grouped_means)

# Step 7: Identify interesting findings
largest_petal_length = grouped_means['petal_length'].idxmax()
print(f"The species with the largest average petal length is: {largest_petal_length}")

# Question 3: Data Visualization

# Step 8: Create visualizations
# Set the style of seaborn
sns.set(style="whitegrid")

# 1. Bar chart showing average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal_length'], palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 2. Histogram of petal length
plt.figure(figsize=(10, 6))
sns.histplot(iris['petal_length'], bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter plot of sepal length vs. petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species', style='species', s=100)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# 4. Line chart showing average petal length per species
plt.figure(figsize=(10, 6))
sns.lineplot(x=grouped_means.index, y=grouped_means['petal_length'], marker='o')
plt.title('Average Petal Length per Species (Line Chart)')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)
plt.show()

