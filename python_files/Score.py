import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📥 Excel-Datei laden
df = pd.read_excel("C:/Users/Sefer Adiyaman/PycharmProjects/Market_Flash/Metrics.xlsx")

# 🧪 Min-Max-Normalisierung
df["CTR_Norm"] = (df["CTR"] - df["CTR"].min()) / (df["CTR"].max() - df["CTR"].min())
df["CR_Norm"] = (df["CR"] - df["CR"].min()) / (df["CR"].max() - df["CR"].min())
df["ER_Norm"] = (df["ER"] - df["ER"].min()) / (df["ER"].max() - df["ER"].min())
df["CPC_Norm"] = (df["CPC"] - df["CPC"].min()) / (df["CPC"].max() - df["CPC"].min())
df["CPA_Norm"] = (df["CPA"] - df["CPA"].min()) / (df["CPA"].max() - df["CPA"].min())

# 🧠 Gewichteter Performance-Index (basierend auf Random-Forest-Importances)
df["Performance_Index"] = (
    0.0811 * df["CTR_Norm"] +
    0.4306 * df["CR_Norm"] +
    0.1170 * df["ER_Norm"] -
    0.0507 * df["CPC_Norm"] -
    0.3206 * df["CPA_Norm"]
) * 100

# 🎯 Skalierung: Median auf 50 verschieben, Score clippen auf [0, 100]
campaign_median = df["Performance_Index"].median()
df["Performance_Score_0_100"] = (df["Performance_Index"] - campaign_median + 50).clip(0, 100)

# 🔄 Runden
df = df.round(5)

#📤 Export (optional)
output_path = "C:/Users/Sefer Adiyaman/PycharmProjects/Market_Flash/metrics_Score.xlsx"
df.to_excel(output_path, sheet_name="Campaign_Scores", index=False)

# 📈 Visualisierungsfunktion
def plot_distribution(data, column, title_prefix, xlabel, color):
    plt.figure(figsize=(10, 5))
    plt.hist(data[column], bins=20, color=color, edgecolor='black')
    plt.title(f"Histogram of Adjusted Performance Score – {title_prefix}")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.boxplot(data[column], vert=False)
    plt.title(f"Boxplot of Adjusted Performance Score – {title_prefix}")
    plt.xlabel(xlabel)
    plt.show()

# 📊 Plot für Campaign Level
plot_distribution(df, "Performance_Score_0_100", "Campaign Level", "Performance Score", "skyblue")

# 🧾 Final-Check: Verteilung
print("\n🎯 Campaign Level:")
print("Median =", round(df["Performance_Score_0_100"].median(), 2))
print("Mean   =", round(df["Performance_Score_0_100"].mean(), 2))
