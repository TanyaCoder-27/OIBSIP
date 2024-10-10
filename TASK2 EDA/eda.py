import pandas as pd

# Load the dataset
retail_sales_data = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes EDA/retail_sales_dataset.csv')

# Convert 'Date' column to datetime format
retail_sales_data['Date'] = pd.to_datetime(retail_sales_data['Date'], format='%d-%m-%Y')

# Check for missing or duplicate values
missing_values = retail_sales_data.isnull().sum()
duplicates = retail_sales_data.duplicated().sum()

# Display updated information after conversion and check for issues
retail_sales_data_info = retail_sales_data.info()

print(missing_values, duplicates, retail_sales_data_info)

# Descriptive statistics for numeric columns
descriptive_stats = retail_sales_data[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].describe()

print(descriptive_stats)

# Group sales by date and calculate the total amount
daily_sales = retail_sales_data.groupby('Date')['Total Amount'].sum()

# Visualize the sales trend over time
import matplotlib.pyplot as plt

daily_sales.plot(kind='line', title='Daily Sales Trend', ylabel='Total Sales Amount', xlabel='Date')
plt.show()

# Analyze customer demographics
customer_age_gender = retail_sales_data.groupby(['Age', 'Gender']).size().unstack(fill_value=0)

# Analyze product performance
product_performance = retail_sales_data.groupby('Product Category')['Total Amount'].sum()

print(customer_age_gender,'\n', product_performance)

#Visualizations
import seaborn as sns

# Bar chart for product performance
plt.figure(figsize=(10, 6))
sns.barplot(x=product_performance.index, y=product_performance.values, palette="viridis")
plt.title('Product Performance')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.show()

# Heatmap for customer demographics
plt.figure(figsize=(10, 6))
sns.heatmap(customer_age_gender, annot=True, cmap="YlGnBu", cbar=False)
plt.title('Customer Demographics (Age and Gender)')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()
