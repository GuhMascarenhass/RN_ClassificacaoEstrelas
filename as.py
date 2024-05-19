import pandas as pd

data_set = pd.read_csv(r"C:\Users\gustavo\Desktop\input dados.csv")
#data_set.pop('Star type')
data_set.to_csv(r"C:\Users\gustavo\Desktop\input2.csv")

Temperatura = data_set.keys()

print(Temperatura)