import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


def gerar_dados_estrelas(n):
    novos_dados = []

    for _ in range(n):
        temp = round(random.uniform(2500, 40000), 1)  # Temperatura em Kelvin
        raio = round(random.uniform(0.1, 2000), 3)  # Raio relativo ao Sol
        magn = round(random.uniform(-10, 20), 2)  # Magnitude absoluta

        if temp>= 30000.0:
            spc_class = 0  # classe O
        elif temp >= 10000.0:
            spc_class = 1  # classe B
        elif temp >= 7500.0:
            spc_class = 2  # classe A
        elif temp >= 6000.0:
            spc_class = 3  # classe F
        elif temp >= 5200.0:
            spc_class = 4  # classe G
        elif temp >= 3700.0:
            spc_class = 5  # classe K
        else:
            spc_class = 6  # classe M

        novos_dados.append([temp, raio, magn, spc_class])

        with open("dados.txt", mode="a") as file:
            # Formatar os dados em uma única string
            linha = f"{temp}, {raio}, {magn}, {spc_class}\n"
            file.write(linha)

    return novos_dados


# Carregar o dataset
data_set = pd.read_csv("stars.csv")

# Extração das variáveis
temperatura = data_set["Temperature (K)"].values
radius = data_set["Radius(R/Ro)"].values
magnitude = data_set["Absolute magnitude(Mv)"].values
spectro = data_set["Spectral Class"].values

# Transformar a classe espectral em números
spectro_dict = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}

# Criar um array de mesma forma que star_typer, preenchido com os valores numéricos das classes espectrais
spectro_numeric = np.array([spectro_dict[s] for s in spectro])
# Combinação das características em um único array
features = np.stack([temperatura, radius, magnitude, spectro_numeric], axis=1)

# Normalização dos dados de entrada
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Dividir os dados em conjuntos de treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X_scaled, spectro_numeric, test_size=0.1, random_state=42)

# Definição do modelo
model = keras.Sequential([
    keras.layers.Input(shape=(4,), name="FirstInput"),
    keras.layers.Dense(8, activation="relu", kernel_initializer='glorot_uniform', name="layer1"),
    keras.layers.Dense(8, activation="relu", kernel_initializer='glorot_uniform', name="layer2"),
    keras.layers.Dense(7, activation="softmax", kernel_initializer='glorot_uniform', name="output")
])

# Compilação do modelo com Adam optimizer e sparse_categorical_crossentropy loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Treinamento do modelo com 40 épocas e parada antecipada
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=80, validation_data=(X_val, y_val), callbacks=[callback_early_stopping])

# Salvar o modelo treinado
model.save("modelo_treinado.keras")

# Geração de novos dados de estrelas
novos_dados_estrelas = gerar_dados_estrelas(30)
novos_dados_estrelas = np.array(novos_dados_estrelas)

# Previsões com o modelo treinado nos novos dados de estrelas
resultado_RN = model.predict(novos_dados_estrelas)

# Impressão dos resultados
np.set_printoptions(suppress=True)
# Dicionário reverso para mapear índices de classe para letras espectrais
spectro_dict_revers = {0: 'O', 1: 'B', 2: 'A', 3: 'F', 4: 'G', 5: 'K', 6: 'M'}

print("Categorias previstas para as estrelas:")
for i, categoria in enumerate(resultado_RN):
    # Encontrar o índice do maior valor (ou seja, a classe com a maior probabilidade)
    indice_max_probabilidade = np.argmax(categoria)

    # Mapear o índice para a classe espectral correspondente
    classe_espectral = spectro_dict_revers[indice_max_probabilidade]

    print(f"Estrela {i + 1}: Classe espectral {classe_espectral}")

print(resultado_RN)

print(novos_dados_estrelas)
