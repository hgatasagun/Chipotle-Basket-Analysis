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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv('/Users/handeatasagun/Documents/GitHub/Chipotle_Data_Analysis/chipotle.csv')
df.head(20)

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
df['item_name'] = df['item_name'].str.replace(r'\W+', '_', regex=True)
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
df_cb = df[df['item_name'] == 'Chicken_Bowl']
len(df_cb)

# How many orders have been placed for the item 'Chicken Bowl'?
df_cb['quantity'].sum()

# How many times have more than 1 canned beverage been ordered?
can_bev = df[(df['item_name'].str.contains('Canned')) & (df['quantity'] > 1)]
len(can_bev)

# How many products contain chicken?
chicken = sorted_df[df['item_name'].str.contains('Chicken')]
len(chicken)

# Which are the top 5 products sold by quantity?
df.pivot_table('quantity', 'item_name', aggfunc= 'sum').\
    sort_values(by= 'quantity', ascending=False).head()
# df.pivot_table('quantity', 'item_name', aggfunc= 'sum').nlargest(5, 'quantity')


##############################################################
## 3. Market Basket Analysis
##############################################################
def create_invoice_product_df(dataframe):
    """
    Creates an invoice-product matrix from the given dataframe.

    Parameters:
        dataframe (DataFrame): The original dataframe containing order and item information.

    Returns:
        DataFrame: An invoice-product matrix where each row represents an order and each column represents a product,
        with 1 indicating the product is in the order and 0 otherwise.
    """
    return dataframe.groupby(['order_id', 'item_name'])['quantity'].sum().\
        unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

create_invoice_product_df(df)

def create_rules(dataframe):
    """
    Generates association rules using the Apriori algorithm.

    Parameters:
        dataframe (DataFrame): The dataframe containing the invoice-product matrix.

    Returns:
        DataFrame: Association rules containing antecedents, consequents, support, confidence, lift, etc.
    """
    dataframe = create_invoice_product_df(dataframe)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

create_rules(df)

def arl_recommender(rules_df, product_name, rec_count=1):
    """
    Recommends products based on association rules.

    Parameters:
        rules_df (DataFrame): The association rules dataframe containing antecedents, consequents, lift, etc.
        product_name (str): The name of the product for which recommendations are needed.
        rec_count (int, optional): The number of recommendations to return. Defaults to 1.

    Returns:
        list: A list of recommended product names.
    """
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    seen_products = set()

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_name:
                consequents = list(sorted_rules.iloc[i]["consequents"])
                for consequent in consequents:
                    if consequent not in seen_products and consequent != product_name:
                        recommendation_list.append(consequent)
                        seen_products.add(consequent)

    return recommendation_list[:rec_count]


## ARL Recommender - 3 recommendations for 'Chicken Bowl' product
recommended_products = arl_recommender(rules, 'Chicken Bowl', 3)
print(recommended_products)

