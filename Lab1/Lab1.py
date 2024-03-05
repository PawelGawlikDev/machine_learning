# %%
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# %%
# fetch dataset
raisin = fetch_ucirepo(id=850)
# data (as pandas dataframes)
X = raisin.data.features
y = raisin.data.targets

# Opis klas
print(y['Class'].value_counts())

# Opis cech
X.describe()

# %%
# Wartości minimalne
min_values = X.min()

# Wartości maksymalne
max_values = X.max()

# Średnia
mean_values = X.mean()

# Mediana
median_values = X.median()

# Drugi (dolny) kwartyl
q2 = X.quantile(0.25)

# Trzeci (górny) kwartyl
q3 = X.quantile(0.75)

# Odchylenie standardowe
std_deviation = X.std()

# Liczność próbek w zbiorze
sample_count = len(X)

# Występowanie danych niepełnych
missing_data_count = X.isnull().sum()

# %%
# Liczność próbek dla poszczególnych klas (jeśli dane są sklasyfikowane)
for column in X.columns:
    class_counts = X[column].value_counts()
    print("\nLiczność próbek dla poszczególnych klas:")
    print(class_counts)

# %%
# Wyświetlenie wyników
print("Wartości minimalne:")
print(min_values)
print("\nWartości maksymalne:")
print(max_values)
print("\nŚrednia:")
print(mean_values)
print("\nMediana:")
print(median_values)
print("\nDrugi (dolny) kwartyl:")
print(q2)
print("\nTrzeci (górny) kwartyl:")
print(q3)
print("\nOdchylenie standardowe:")
print(std_deviation)
print("\nLiczność próbek w zbiorze:", sample_count)
print("\nWystępowanie danych niepełnych:")
print(missing_data_count)

# %%
# Wykresy dla trzech wybranych zmiennych
selected_columns = X.columns

for column in selected_columns:
    # Wykres pudełkowy
    plt.figure(figsize=(8, 6))
    X.boxplot(column=column)
    plt.title(f'Wykres pudełkowy dla {column}')
    plt.show()

    # Wykres liniowy
    plt.figure(figsize=(8, 6))
    plt.plot(X.index, X[column])
    plt.title(f'Wykres liniowy dla {column}')
    plt.xlabel('Indeks próbki')
    plt.ylabel('Wartość')
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 6))
    X[column].plot(kind='hist', bins=20)
    plt.title(f'Histogram dla {column}')
    plt.xlabel('Wartość')
    plt.ylabel('Częstotliwość')
    plt.show()
