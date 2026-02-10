from shoppingassistant.params import *
from shoppingassistant.helper_functions import get_article_id_from_path, get_image_paths
from shoppingassistant.process_data import get_most_popular_articles
import os

def suggest_from_sales(closest_matchs, transacions_df=None):
    """
    Suggest items based on recent sales data. Would give one suggestion per closest match.
    Args:
        closest_match (str): Paths to the closest matching items.
        model: Pre-trained model for feature extraction and similarity computation.
                If no model is provided, models will be loaded using load_model function.

    Returns:
        list: A list of image paths for suggested items.
    """
    suggestions = []
    if type(closest_matchs) is not list:
        closest_matchs = [closest_matchs]

    for item in closest_matchs:
        article_id = get_article_id_from_path(item)
        best_sales = get_most_popular_articles(article_id, transactions_df=transacions_df)
        if best_sales[0] not in suggestions:
            suggestions.append(best_sales[0])
        else:
            suggestions.append(best_sales[1])
    return get_image_paths(suggestions)




if __name__ == "__main__":
    # Example usage
    closest_matchs = [
        "path/to/closest_match_1.jpg",
        "path/to/closest_match_2.jpg",
        "path/to/closest_match_3.jpg",
    ]
    suggestions = suggest_from_sales(closest_matchs, top_k=5)
    print("Suggested items:", suggestions)
