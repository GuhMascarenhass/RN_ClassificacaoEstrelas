import pandas as pd

# Função para capitalizar a primeira letra de cada palavra

df = pd.read_csv('6 class csv.csv')

# Especificar a coluna que deseja analisar


# Capitalizar a primeira letra de cada palavra na coluna especificada
df.pop('Star type')
df.pop('Star color')

# Remover espaços em branco no início e no final das palavras na coluna especificada
# Salvar o DataFrame modificado em um novo arquivo CSV
df.to_csv('6 class csv2.csv', index=False)

# Imprimir a lista de palavras únicas (opcional)
