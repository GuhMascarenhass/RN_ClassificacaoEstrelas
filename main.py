import os
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


# Função para gerar novos dados de estrelas
# Função para gerar novos dados de estrelas
def gerar_dados_estrelas(n):
    novos_dados = []
    novos_dados_semCE = []

    for _ in range(n):
        temperatura = round(random.uniform(2500, 40000), 1)  # Temperatura em Kelvin
        luminosidade = round(random.uniform(0.0001, 1000000), 5)  # Luminosidade relativa ao Sol
        raio = round(random.uniform(0.1, 2000), 3)  # Raio relativo ao Sol
        magnitude = round(random.uniform(-10, 20), 2)  # Magnitude absoluta

        if temperatura > 25000:
            classe_espectral = 'O'
        elif temperatura >= 11000:
            classe_espectral = 'B'
        elif temperatura >= 7500:
            classe_espectral = 'A'
        elif temperatura >= 5000:
            classe_espectral = 'G'
        elif temperatura >= 3500:
            classe_espectral = 'K'
        else:
            classe_espectral = 'M'

        novos_dados.append([temperatura, luminosidade, raio, magnitude, classe_espectral])
        novos_dados_semCE.append([temperatura, luminosidade, raio, magnitude])

    return novos_dados, novos_dados_semCE


# Carregar o dataset
data_set = pd.read_csv("6 class csv2.csv")

# Extração das variáveis
temperatura = data_set["Temperature (K)"].values
lumens = data_set["Luminosity(L/Lo)"].values
radius = data_set["Radius(R/Ro)"].values
magnitude = data_set["Absolute magnitude(Mv)"].values
spectro = data_set["Spectral Class"].values

# Transformar a classe espectral em números
spectro_dict = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}
# Criar um array de mesma forma que star_typer, preenchido com os valores numéricos das classes espectrais
spectro_numeric = np.array([spectro_dict[s] for s in spectro])
# Combinação das características em um único array
features = np.stack([temperatura, lumens, radius, magnitude, spectro_numeric], axis=1)

# Normalização dos dados de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_scaled, spectro_numeric, test_size=0.3, random_state=42)

# Definição do modelo
model = keras.Sequential([
    keras.layers.Input(shape=(5,), name="FirstInput"),
    keras.layers.Dense(32, activation="relu", name="layer1"),  # Ajuste o número de unidades para 32
    keras.layers.Dense(7, activation="softmax", name="output")  # 7 unidades para as 7 classes espectrais
])

# Compilação do modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Treinamento do modelo
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[callback_early_stopping])

# Geração de novos dados de estrelas
novos_dados_estrelas = gerar_dados_estrelas(5)
novos_dados_estrelas = np.array(novos_dados_estrelas)

# Previsões com o modelo treinado nos novos dados de estrelas
resultado_RN = np.argmax(model.predict(novos_dados_estrelas), axis=1)

# Impressão dos resultados
np.set_printoptions(suppress=True)
print("Categorias previstas para as estrelas:")
for i, categoria in enumerate(resultado_RN):
    print(f"Estrela {i + 1}: Classe espectral {categoria}")

print(novos_dados_estrelas)
