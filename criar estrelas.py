import random


def generate_star(n):
    # Define os parâmetros das classes espectrais
    spectral_classes = [
        {"class": "O", "temp_range": (25000, 50000)},
        {"class": "B", "temp_range": (11000, 25000)},
        {"class": "A", "temp_range": (7500, 11000)},
        {"class": "G", "temp_range": (5000, 6000)},
        {"class": "K", "temp_range": (3500, 5000)},
        {"class": "M", "temp_range": (2000, 3500)}
    ]

    # Seleciona aleatoriamente uma classe espectral
    spectral_class = random.choice(spectral_classes)

    # Define a temperatura dentro do intervalo da classe espectral
    temperature = random.uniform(*spectral_class["temp_range"])

    # Define outros parâmetros aleatoriamente (assumindo algumas faixas razoáveis para luminosidade, raio e magnitude absoluta)
    luminosity = random.uniform(0.1, 100)  # L/Lo
    radius = random.uniform(0.1, 10)  # R/Ro
    absolute_magnitude = random.uniform(-10, 15)  # Mv
    result = {
        "Temperature (K)": round(temperature, 2),
        "Luminosity (L/Lo)": round(luminosity, 2),
        "Radius (R/Ro)": round(radius, 2),
        "Absolute Magnitude (Mv)": round(absolute_magnitude, 2),
        "Spectral Class": spectral_class["class"]
    }

    # Converte dicionários para listas de números
    stars_with_class_numeric = [
        ["Temperature (K)"], ["Luminosity (L/Lo)"], ["Radius (R/Ro)"], ["Absolute Magnitude (Mv)"],["Spectral Class"]]

    stars_without_class_numeric = [
        ["Temperature (K)"],["Luminosity (L/Lo)"], ["Radius (R/Ro)"], ["Absolute Magnitude (Mv)"]]

    return stars_with_class_numeric, stars_without_class_numeric


# Gerar 10 estrelas
stars_with_class_numeric, stars_without_class_numeric = generate_star(10)

# Exibir resultados
print("Estrelas com Classe Espectral (Apenas Números):")
for star in stars_with_class_numeric:
    print(star)

print("\nEstrelas sem Classe Espectral (Apenas Números):")
for star in stars_without_class_numeric:
    print(star)
