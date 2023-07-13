##############################################################
# Chipotle Restaurant Chain: Analysis of Sold Products
##############################################################

##############################################################
## 1. Business Problem
##############################################################
## The objective is to analyze the data pertaining to Chipotle in order to derive valuable insights and facilitate
## the development of their business strategy. Through these analyses, Chipotle aims to understand customer preferences,
## identify popular products, optimize pricing strategies, and enhance customer satisfaction.

## Variables

## order_id -- Unique value associated with each order
## quantity -- Number of orders received for a product
## item_name -- Name of sold product
## choice_description -- Customer preference in a sold product
## item_price -- Total payment amount

###############################################################
## 2. Data Preparation
###############################################################

### Importing libraries
##############################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width' , 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('/Users/handeatasagun/Documents/GitHub/Chipotle_Data_Analysis/chipotle.csv')

### Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Shape ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)

### Removing the "$" sign from the 'item price' variable and changing the data type
df['item_price'] = df['item_price'].str.strip('$')
df['item_price'] = df['item_price'].astype(float)

### Replace non-alphanumeric characters in the 'item_name' column
df['item_name'].unique()
df['item_name'] = df['item_name'].str.replace('\W+', '_')
df.head()

###############################################################
## 3. Data Analysis
###############################################################

### Products with a unit price higher than $8
##################################################

### Calculating 'price per item'
df['price_per_item'] = df['item_price'] / df['quantity']

### Number of unique for 'item_name'
df['item_name'].nunique()

# The removal of duplicate item_names from the dataset
unique_items = df.drop_duplicates('item_name')
unique_items.shape

### Number of products with a unit price higher than $8
(unique_items['price_per_item'] > 8).sum()

# Product ranking by unit price
sorted_df = unique_items.sort_values(by='price_per_item', ascending=False)
sorted_df[['item_name', 'price_per_item']]


### Number of sold products
##################################################

# How many times has a particular product been ordered?
df['item_name'].value_counts()

# How many times has the item 'Chicken Bowl' been ordered?"
df_cb = df[df['item_name'] == 'Chicken Bowl']
len(df_cb)

# How many orders have been placed for the item 'Chicken Bowl'?
df_cb['quantity'].sum()

# How many times have more than 1 canned beverage been ordered?
can_bev = df[(df['item_name'].str.contains('Canned')) & (df['quantity'] > 1)]
len(can_bev)

# How many products contain chicken?
chicken = df[df['item_name'].str.contains('Chicken')]
len(chicken)

# Which are the top 5 products sold by quantity?
df.pivot_table('quantity', 'item_name', aggfunc= 'sum').sort_values(by= 'quantity', ascending=False).head()
# df.pivot_table('quantity', 'item_name', aggfunc= 'sum').nlargest(5, 'quantity')

##############################################################
## 3. Market Basket Analysis
##############################################################
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

### Group the data by orders and concatenate the items
grouped_df = df.groupby('order_id')['item_name'].apply(list).reset_index()

### Convert the dataset using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit_transform(grouped_df['item_name'])
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

### Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

### Extract association rules
if frequent_itemsets.empty:
    print("No frequent itemsets found.")
else:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    if rules.empty:
        print("No association rules found.")
    else:
        print(rules)

# NO ASSOCIATION RULES FOUND.