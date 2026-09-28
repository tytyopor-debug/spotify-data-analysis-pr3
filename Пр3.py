import pandas as pd
import matplotlib.pyplot as plt

# Задание 1
df = pd.read_csv("vgsales.csv")
print("--- Задание 1 ---")
print(df.head(10))

# Задание 2
print("\n--- Задание 2 ---")
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("Названия столбцов:", list(df.columns))
print("\nТипы данных:")
print(df.dtypes)
print("\nСтатистические характеристики:")
print(df.describe())

# Задание 3
print("\n--- Задание 3 ---")
nulls = df.isnull().sum()
print("Пропущенные значения:")
print(nulls)
print(f"Больше всего пропусков в столбце: {nulls.idxmax()} ({nulls.max()})")
print("\nСтроки без значения Year:")
print(df[df["Year"].isnull()].head())

# Задание 4
print("\n--- Задание 4 ---")
df_clean = df.dropna(subset=["Year"]).copy()
print(f"Размер исходного датасета: {df.shape}")
print(f"Размер очищенного датасета: {df_clean.shape}")
print("Пропуски в df_clean:")
print(df_clean.isnull().sum())

# Задание 5
print("\n--- Задание 5 ---")
print("Количество жанров:", df_clean["Genre"].nunique())
print("Количество платформ:", df_clean["Platform"].nunique())
print("Количество издателей:", df_clean["Publisher"].nunique())

# Задание 6
print("\n--- Задание 6 ---")
genre_count = df_clean.groupby("Genre")["Name"].count().sort_values(ascending=False)
print("Количество игр по жанрам:")
print(genre_count)

# Задание 7
print("\n--- Задание 7 ---")
genre_sales = df_clean.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
print("Мировые продажи по жанрам (млн):")
print(genre_sales)

# Задание 8
print("\n--- Задание 8 ---")
cols = ["Rank", "Name", "Platform", "Genre", "Global_Sales"]
print("Топ-10 самых продаваемых игр:")
print(df_clean.sort_values(by="Global_Sales", ascending=False)[cols].head(10))

# Задание 9
print("\n--- Задание 9 ---")
print("Топ-10 платформ по количеству игр:")
print(df_clean.groupby("Platform")["Name"].count().sort_values(ascending=False).head(10))
print("\nТоп-10 платформ по продажам:")
print(df_clean.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(10))

# Задание 10
print("\n--- Задание 10 ---")
plt.figure(figsize=(9, 4.5))
genre_sales.sort_values(ascending=True).plot(kind="barh", color="#2b5c8f")
plt.title("Продажи по жанрам (млн)")
plt.xlabel("Продажи, млн")
plt.ylabel("Жанр")
plt.tight_layout()
plt.show()

# Задание 11
print("\n--- Задание 11 ---")
corr_na_global = df_clean[["NA_Sales", "Global_Sales"]].corr().iloc[0, 1]
print(f"Корреляция NA_Sales и Global_Sales: {corr_na_global:.4f}")

# Задание 12
print("\n--- Задание 12 ---")
plt.figure(figsize=(8, 5))
plt.scatter(df_clean["NA_Sales"], df_clean["Global_Sales"], alpha=0.5, color="#2b5c8f")
plt.title("Связь продаж в Северной Америке и мировых продаж")
plt.xlabel("NA_Sales, млн")
plt.ylabel("Global_Sales, млн")
plt.tight_layout()
plt.show()

# Задание 13
print("\n--- Задание 13 ---")
sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
print("Матрица корреляции регионов:")
print(df_clean[sales_cols].corr().round(3))