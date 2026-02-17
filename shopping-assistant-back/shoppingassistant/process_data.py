import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
from pathlib import Path

from shoppingassistant.params import BASE_DIR
from shoppingassistant.clustering import get_similar_items
from shoppingassistant.helper_functions import get_image_path, get_article_id_from_path

def filter_data(product_group_names=['Shoes']):
    '''
    Filters the dataset to only keep the rows corresponding to the specified
    product group names. Filters over the column 'product_group_name'
    for the articles.csv, transactions.csv and images.
    Removes rows that do not have corresponding images.
    Saves the filtered data to articles_filtered.csv and transactions_filtered.csv.
    '''
    path_to_data = Path(BASE_DIR + "/raw_data/")
    if not (path_to_data / "articles_filtered.csv").exists() or not (path_to_data / "transactions_filtered.csv").exists():
        # Loading Data
        articles_df = pd.read_csv(path_to_data / "articles.csv")
        transactions_df = pd.read_csv(path_to_data / "transactions.csv")
        images_path = path_to_data / "images_256_256"

        # Filtering Data using input product group names
        articles_df_filtered = articles_df[articles_df['product_group_name']==product_group_names]
        transactions_df['t_dat']= pd.to_datetime(transactions_df['t_dat'])
        df_trans_filtered = transactions_df[transactions_df['article_id'].isin(articles_df_filtered['article_id'])]

        # Creating images_filtered directory that will contain only the images of the filtered articles
        needed_article_ids = articles_df_filtered['article_id'].unique()
        source_images_path = Path(BASE_DIR + '/raw_data/images_256_256')
        dest_images_path = Path(BASE_DIR + '/raw_data/images_filtered')
        copied_count = 0
        missing_count = 0

        for article_id in needed_article_ids:
            article_str = str(article_id).zfill(10)
            subfolder = article_str[:3]

            source_subfolder = source_images_path / subfolder
            dest_subfolder = dest_images_path / subfolder
            source_file = source_subfolder / f"{article_str}.jpg"
            dest_file = dest_subfolder / f"{article_str}.jpg"

            if source_file.exists():
                dest_subfolder.mkdir(exist_ok=True)

                shutil.copy2(source_file, dest_file)
                copied_count += 1
            else:
                missing_count += 1
                articles_df_filtered.drop(articles_df_filtered[
                    articles_df_filtered['article_id'] == article_id].index, inplace=True)
                df_trans_filtered.drop(df_trans_filtered[
                    df_trans_filtered['article_id'] == article_id].index, inplace=True)

        print(f"Copied {copied_count} images")
        print(f"Missing images: {missing_count}")
        print(f"Total article IDs: {needed_article_ids.shape}")
        articles_df_filtered.to_csv(BASE_DIR + "/raw_data/articles_filtered.csv")
        df_trans_filtered.to_csv(BASE_DIR + "/raw_data/transactions_filtered.csv")
        print(f"✅ Filtered data saved to articles_filtered.csv and transactions_filtered.csv")
    else:
        print("✅ Filtered data already exists.")
    return None



def load_dataframes(keep_colors=False):
    '''
    Docstring for preprocess
    Preprocess the data
    '''
    path_to_data = Path(BASE_DIR + "/raw_data/")
    filter_data()

    transactions_df = pd.read_csv(path_to_data / "transactions_filtered.csv")
    articles_df = pd.read_csv(path_to_data / "articles_filtered.csv")

    # Additional preprocessing steps can be added here
    transactions_df = transactions_df[['t_dat', 'article_id', 'price', 'customer_id', 'sales_channel_id']]
    transactions_df['t_dat']= pd.to_datetime(transactions_df['t_dat'])

    # Drop color-related columns if keep_colors is False
    if not keep_colors:
        articles_df = articles_df.drop(columns=['colour_group_code',
                                                'colour_group_name',
                                                'perceived_colour_value_id',
                                                'perceived_colour_value_name',
                                                'perceived_colour_master_id',
                                                'perceived_colour_master_name'])

    # Only select relevant columns
    relevant_columns = ['article_id','product_code', 'product_type_name', 'product_group_name',
                        'index_group_name', 'prod_name']
    if keep_colors:
        relevant_columns += ['perceived_colour_master_name']

    articles_df = articles_df[relevant_columns]

    return articles_df, transactions_df




def sort_by_revenue(article_ids):
    """
    Returns a list of article_ids sorted by highest total revenue (descending).

    Revenue is computed as the sum of 'price' per article_id
    from transactions_filtered.csv.
    """
    # Load filtered transactions
    df_trans = pd.read_csv(BASE_DIR + "/raw_data/transactions_filtered.csv")

    # Sum income per article
    revenue_per_article = df_trans.groupby("article_id")["price"].sum()

    # Sort article_ids by income (highest first)
    sorted_ids = revenue_per_article.reindex(article_ids).fillna(0).sort_values(ascending=False).index.tolist()

    return sorted_ids

def get_most_popular_articles(article_id, n=10, transactions_df=None):
    """
    Returns a list of the n most popular article_ids based on total revenue
    and the given article_id, i.e. which are the most revenue-generating articles
    that were bought together with the given article_id at the same day.
    """
    # Load filtered transactions
    if transactions_df is None:
        _, transactions_df = load_dataframes()

    # Get customer_ids who bought the given article_id
    customers = transactions_df[transactions_df["article_id"] == article_id]["customer_id"].unique()

    def fallback_case(article_id, n):
        # Function for fallback case in case article doesnt have items sold with it
        items = get_similar_items(get_image_path(article_id))
        ids= [p['article_id'] for p in items]
        sorted_by_revenue = sort_by_revenue(ids)
        return sorted_by_revenue[:n]

    if customers.size == 0:
        return fallback_case(article_id, n)

    most_sold_with_article = pd.DataFrame(columns=['article_id', 'price'])

    for cust in customers:
        df_cust = transactions_df[transactions_df['customer_id']==cust] # Transactions of that customer
        sell_date = df_cust[df_cust['article_id']==article_id]['t_dat'].values[0] # Date when the article was bought
        articles_bought_same_day = df_cust[df_cust['t_dat']==sell_date]
        articles_bought_same_day = articles_bought_same_day[articles_bought_same_day!=article_id]
        articles_bought_same_day = articles_bought_same_day.dropna()
        articles_bought_same_day = articles_bought_same_day.sort_values('price')
        most_sold_with_article = pd.concat([most_sold_with_article, articles_bought_same_day[['article_id', 'price']].head(1)], ignore_index=True)
    revenue_with_article = most_sold_with_article.groupby('article_id').sum().reset_index()
    revenue_with_article = revenue_with_article.sort_values('price', ascending=False)
    if most_sold_with_article.size == 0:
        return fallback_case(article_id, n)
    return revenue_with_article['article_id'].head(n).map(int).tolist()

def get_price(article_path, transactions_df=None):
    article_id = get_article_id_from_path(article_path)
    if transactions_df is None:
        _, transactions_df = load_dataframes()
    prices = transactions_df[transactions_df['article_id']==article_id]['price']
    if prices.size > 0:
        price = float(prices.iloc[-1])
    else:
        price = 0.03725424307201469
    return round(price*2684.258, 2)


if __name__ == "__main__":
    # Example usage

    _, df_trans = load_dataframes()
    article_id = df_trans['article_id'].iloc[0]  # Get an example article_id from the transactions
    most_popular_articles = get_most_popular_articles(article_id, n=10)
    print(f"The most popular articles bought with article {article_id} are: {most_popular_articles}")
