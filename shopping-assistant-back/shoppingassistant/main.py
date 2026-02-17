
from PIL import Image
import requests
from pathlib import Path
import os
from shoppingassistant.params import *
from shoppingassistant.classification import classify_subcategory, classify_gender
from shoppingassistant.clustering import get_similar_items
from shoppingassistant.suggestions import suggest_from_sales
from shoppingassistant.process_data import load_dataframes, get_price
from tensorflow.keras.models import load_model
from shoppingassistant.helper_functions import get_image, display_suggestions, get_prod_name
import base64
import timeit

def load_model_class():
    """
    Loads and returns the pre-trained model for classification.
    """
    pass

def save_model(model):
    """
    Saves the given model either to mlflow if MODEL_TARGET is 'mlflow'
    or to a local path if MODEL_TARGET is 'local'.
    """

    pass


def suggest_articles(image_path, top_k=5, subcategory=None, gender=None, model_sub=None, model_gen=None):
    """
    Suggest similar items based on the input image using the provided model.

    Args:
        image_path (str): Path to the input image, can additionally provide url.
        model: Pre-trained model for feature extraction and similarity computation.
               If no model is provided, models will be loaded using load_model function.
        top_k (int): Number of similar items to suggest.

    Returns:
        list: A list of image paths for similar images.
    """
    assert top_k > 0, "top_k must be a positive integer"
    assert top_k <= 10, "top_k must be less than or equal to 10 to avoid long processing times"
    image_path = get_image(image_path)
    articles_df, transactions_df = load_dataframes()

    start = timeit.default_timer()
    if subcategory is None:
        if model_sub is None:
            model_sub = load_model(os.path.join(BASE_DIR, 'models', 'subcategory_classifier_best.keras'))
        subcategory = classify_subcategory(image_path, model=model_sub, articles_df=articles_df)
    stop = timeit.default_timer()
    sub_classifier_run_time = stop - start

    start = timeit.default_timer()
    if gender is None:
        if model_gen is None:
            model_gen = load_model(os.path.join(BASE_DIR, 'models', 'gender_classifier.keras'))
        gender = classify_gender(image_path, model_gen)
    stop = timeit.default_timer()
    gen_classifier_run_time = stop - start
    print(f"Predicted category: {subcategory}")
    print(f"Predicted gender: {gender}")


    # Further processing to suggest similar items based on category_pred
    # Get n+1//2 similar items and the rest from sales data
    start = timeit.default_timer()
    similar_articles = get_similar_items(image_path, subcategory=subcategory, gender=gender, n=(top_k+1)//2)
    stop = timeit.default_timer()
    clustering_run_time = stop - start

    similar_images = []
    for item in similar_articles:
        similar_images.append(item['image_path']) # or article_id

    sales_suggestions = suggest_from_sales(similar_images, transacions_df=transactions_df)
    # if len(sales_suggestions) < (top_k - len(similar_images)):  # in case not enough sales suggestions fill out with similar articles
    suggestions = similar_images + sales_suggestions
    suggestions = suggestions[:top_k]
    # Return image object instead of path using Image
    # put images in docker image or Drive if needed
    # return suggestions
    print(f" -------------- ⏰⏰⏰⏰⏰⏰⏰⏰⏰ -------------- ")
    print(f"⌛ The runtime of the Subcategory classification model is {sub_classifier_run_time} ⌛")
    print(f"⌛ The runtime of the Gender classification model is {gen_classifier_run_time} ⌛")
    print(f"⌛ The runtime of the similarity search model is {clustering_run_time} ⌛")
    print(f" -------------- ⏰⏰⏰⏰⏰⏰⏰⏰⏰ -------------- ")
    return [
        {'name': get_prod_name(p, articles_df=articles_df),
         'data': base64.b64encode(Path(p).read_bytes()).decode(),
         'description': f"Similar item #{i+1}" if i<len(similar_images) else f"Sales based suggestion #{i+1 - len(similar_images)}",
         'subcategory': f"Subcategory: {subcategory}",
         'gender': f"Gender: {gender}",
         'price': get_price(p, transactions_df=transactions_df),
         'path': p
         }
        for i, p in enumerate(suggestions)
    ]


if __name__ == "__main__":
    # Example usage
    from shoppingassistant.helper_functions import get_image_path, display_results
    from shoppingassistant.process_data import load_dataframes

    image = BASE_DIR + "/raw_data/test_images/temp2.jpg"


    # Some urls for testing the function
    # url = 'https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcS9e8LgfdduH__eVABV2kFceX-nAnhQXsFuwKMClMNwS11B6aKB6QSsKWEQfl3Q3HjH3z3YGVbRN7j9ellAW4qiokIcxLwdtRhoKO0YFffMbop9GmxTRFMEbAHa3ghUoMU79CsfavVjLA&usqp=CAc'
    url = 'https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcRbVSH29symEB856MgsktvF2Awj8tKjB6JU2rqdIki9cM2H8dt_5ygqhRa_p2YDf9Awv9K6vWs69iSkY72z0kjKvY8HoKKekfzgtAk_B0-h_8QW_u3Q6vfKzxZ9ju01hFHwxzvzw7HsVg&usqp=CAc'
    # url = 'https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcR9gf6qxciZPFtmy7il5ZnaMenaohfEmfK2FQ_ieT-QbzJrw0X3AA1GqoSHU914_Rt_CU7xASXA-Iohe3U_tr4NfC8N-UliXiupYrukZHiQF-Vbwu8njW9Q&usqp=CAc'
    suggestions = suggest_articles(url, top_k=4, gender="Menswear")
    display_suggestions(suggestions)
