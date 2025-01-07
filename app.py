from flask import Flask, render_template, request
# import numpy as np  
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

app = Flask(__name__)

with open('recipe_knn_model.sav', 'rb') as f:
    knn = pickle.load(f)

with open('knn_vectorizer.sav', 'rb') as f:
    vectorizer = pickle.load(f)

with open('parsed_recipes.sav', 'rb') as f:
    recipe_data = pickle.load(f)


def find_closest_recipe(ingredients_list):
    ingredients_text = ' '.join(ingredients_list)
    input_vector = vectorizer.transform([ingredients_text])

    # use the pre-fitted KNN model to find the closest recipe index
    closest_index = knn.kneighbors(input_vector, return_distance=False)[0][0]

    # get closest recipe details from the original dataset
    closest_recipe = recipe_data.iloc[closest_index]

    return {
        'title': closest_recipe['titles'],
        'ingredients': closest_recipe['ingredients'],
        'directions': closest_recipe['directions']
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        input_ingredients = ingredients.split(',')

        closest_recipe = find_closest_recipe(input_ingredients)

        return render_template('index.html', closest_recipe=closest_recipe)
    return render_template('index.html', closest_recipe=None)

if __name__ == '__main__':
    app.run(debug=True)