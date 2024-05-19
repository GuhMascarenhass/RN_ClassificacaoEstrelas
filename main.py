import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
import pandas as pd
import numpy as np


# Função para salvar o modelo treinado
def salvar_modelo(modelo, caminho_arquivo):
    modelo.save(caminho_arquivo)
    print(f"Modelo salvo em: {caminho_arquivo}")


# Função para carregar o modelo treinado
def carregar_modelo(caminho_arquivo):
    modelo = keras.models.load_model(caminho_arquivo)
    print(f"Modelo carregado de: {caminho_arquivo}")
    return modelo


# Carregar o dataset
data_set = pd.read_csv(r"6 class csv.csv")

# Extração das variáveis
temperatura = data_set["Temperature (K)"].values
lumens = data_set["Luminosity(L/Lo)"].values
radius = data_set["Radius(R/Ro)"].values
magnitude = data_set["Absolute magnitude(Mv)"].values
color = data_set["Star color"].values
spectro = data_set["Spectral Class"].values
star_typer = data_set["Star type"].values

# Transformar variáveis categóricas (color e spectro) em números
color_dict = {color: idx for idx, color in enumerate(np.unique(color))}
spectro_dict = {spectro: idx for idx, spectro in enumerate(np.unique(spectro))}

color_numeric = np.array([color_dict[c] for c in color])
spectro_numeric = np.array([spectro_dict[s] for s in spectro])

# Conversão para tf.data.Dataset
features = np.stack([temperatura, lumens, radius, magnitude, color_numeric, spectro_numeric], axis=1)
features_ds = tf.data.Dataset.from_tensor_slices(features.astype(float))
labels_ds = tf.data.Dataset.from_tensor_slices(star_typer.astype(int))

# Combinação dos datasets
data_set_compact = tf.data.Dataset.zip((features_ds, labels_ds))
data_set_compact = data_set_compact.batch(32)

# Definição do modelo
model = keras.Sequential([
    keras.layers.Input(shape=(6,), name="FirstInput"),  # Ajuste para 6 características de entrada
    keras.layers.Dense(10, activation="relu", name="layer1"),
    keras.layers.Dense(6, activation="softmax", name="output")  # 6 classes de saída para Star type
])

# Compilação do modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Treinamento do modelo
#model.fit(data_set_compact, epochs=3000)

# Salvar o modelo treinado
# caminho_modelo = r"C:\Users\gustavo\Desktop\modelo_rede_neural.keras"
# salvar_modelo(model, caminho_modelo)

# Previsões

x = np.array([
    [3042, 0.0005, 0.1542, 16.6, color_dict['Red'], spectro_dict['M']],
    [3834, 272000.0, 1183.0, -9.2, color_dict['Red'], spectro_dict['M']],
    [6757, 1.43, 1.12, 2.41, color_dict['yellow-white'], spectro_dict['F']],
    [21020, 0.0015, 0.0112, 11.52, color_dict['Blue'], spectro_dict['B']],
    [3600, 240000.0, 1190.0, -7.89, color_dict['Red'], spectro_dict['M']]
], dtype=float)

# Carregar o modelo salvo
model_carregado = carregar_modelo("modelo_rede_neural.keras")

# Previsões com o modelo carregado
resultado_RN = np.argmax(model_carregado.predict(x), axis=1)

# Impressão dos resultados
print("Categorias previstas para as estrelas:")
for i, categoria in enumerate(resultado_RN):
    print(f"Estrela {i + 1}: Categoria {categoria}")

print(model_carregado.summary())
