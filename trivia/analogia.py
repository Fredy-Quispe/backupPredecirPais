import pandas as pd
import numpy as np
import pickle
import re

# Cargar los embeddings de palabras
word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))

# Cargar los datos de las capitales
capitals_df = pd.read_csv("capitals.txt", sep=" ", header=None, names=["Capital", "Country"])

# Función para encontrar el país más cercano a un vector
def find_closest_country(v):
    country_embedding_matrix = np.array([word_embeddings[capitals_df.iloc[i]["Country"]] for i in range(len(capitals_df))])
    diff = country_embedding_matrix - v
    delta = np.sum(diff * diff, axis=1)
    index = np.argmin(delta)
    return capitals_df.iloc[index]["Country"]

# Función para predecir la analogía completa
def predict_complete_analogy(city1, country1, city2):
    country1_actual = capitals_df[capitals_df["Capital"] == city1].iloc[0]["Country"]
    country2_actual = capitals_df[capitals_df["Capital"] == city2].iloc[0]["Country"]
    vector_country1_actual = word_embeddings[country1_actual]
    vector_city1 = word_embeddings[city1]
    vector_city2 = word_embeddings[city2]
    analogy_vector = vector_country1_actual + vector_city2 - vector_city1
    predicted_country = find_closest_country(analogy_vector)

    # Crear la oración completa
    analogy_sentence = f"{city1} es a {country1} como {city2} es a {predicted_country}"

    return analogy_sentence

# Función para predecir la analogía completa a partir de una oración
def predict_analogy_from_sentence(sentence):
    city_pattern = re.compile(r'\b(?:' + '|'.join(capitals_df["Capital"].values) + r')\b', flags=re.IGNORECASE)
    country_pattern = re.compile(r'\b(?:' + '|'.join(capitals_df["Country"].values) + r')\b', flags=re.IGNORECASE)

    cities_in_sentence = city_pattern.findall(sentence)
    countries_in_sentence = country_pattern.findall(sentence)

    if len(cities_in_sentence) >= 2 and len(countries_in_sentence) >= 1:
        city1 = cities_in_sentence[0]
        country1 = countries_in_sentence[0]
        city2 = cities_in_sentence[1]

        analogy_sentence = predict_complete_analogy(city1, country1, city2)

        return analogy_sentence
    else:
        return "No se pueden detectar suficientes ciudades y países en la oración."

# Ingresar la oración completa
input_sentence = input("Ingrese una oración que contenga dos ciudades y un país: ")

# Obtener y mostrar la analogía completa
analogy_result = predict_analogy_from_sentence(input_sentence)
print(analogy_result)