import pandas as pd

# 📥 CSV laden
csv_path = "C:/Users/Sefer Adiyaman/PycharmProjects/Market_Flash/marketflash_marketing_data_2023_cleaned.csv"
df = pd.read_csv(csv_path)

# 📊 KPIs berechnen
df["ER"] = (df["Likes"] + df["Conversions"]) / df["Views"]
df["CTR"] = df["Clicks"] / df["Views"]
df["CPC"] = df["Expense"] / df["Clicks"]
df["CPA"] = df["Expense"] / df["Conversions"]


# 📤 Export als Excel-Datei
output_path = "C:/Users/Sefer Adiyaman/PycharmProjects/Market_Flash/Metrics.xlsx"
df.to_excel(output_path, index=False, engine='openpyxl')

# ✅ Ausgabe
print("✅ Datei mit Zielwerten exportiert!")

