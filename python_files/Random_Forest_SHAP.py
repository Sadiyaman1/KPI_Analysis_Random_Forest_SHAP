import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import shap
import numpy as np

# === 1. Daten laden ===
df = pd.read_excel("C:/Users/Sefer Adiyaman/PycharmProjects/Market_Flash/Metrics.xlsx")

# === 2. Features & Zielvariable ===
X = df[['CPA','CPC','CTR', 'ER', 'CR']]
y = df['Conversions']

# === 3. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Modell trainieren ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 5. Vorhersagen & Bewertung ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"🎯 R² Score: {r2:.2f}")

# === 6. Klassische Feature Importances ===
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances,
    'Weight (normalized)': importances / importances.sum()
}).sort_values(by='Importance', ascending=False)

print("\n📊 Klassische Feature Importances:")
print(feature_importance_df)

# === 7. Plot der klassischen Feature Importance ===
plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (klassisch)')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === 8. SHAP auf Random Forest anwenden ===
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test, check_additivity=False)


# === 9. SHAP Summary Plot ===
shap.summary_plot(shap_values.values, X_test, plot_type='dot')


# === 10. SHAP Bar Plot (durchschnittliche Wichtigkeit) ===
shap.plots.bar(shap_values, max_display=10)

# === 11. SHAP Waterfall Chart für Top-Kampagne (ohne Überlappung) ===

# Kampagne mit höchstem vorhergesagtem Conversion-Wert
top_index = np.argmax(y_pred)

print(f"\n📌 SHAP-Waterfall für Kampagne #{top_index} mit {y_pred[top_index]:.2f} vorhergesagten Conversions")

# Neue Figure erzeugen (größer für bessere Lesbarkeit)
fig = plt.figure(figsize=(10, 6))

# SHAP-Waterfall-Plot anzeigen (auf Top-5 Features beschränkt)
shap.plots.waterfall(shap_values[top_index], max_display=5, show=False)

# Schriftgröße verkleinern, um Überlappung zu vermeiden
for text in plt.gca().texts:
    text.set_fontsize(9)

# Layout optimieren und als Bild speichern
plt.tight_layout()
plt.savefig("shap_waterfall_topkampagne.png", dpi=300, bbox_inches='tight')
plt.show()



# === 12. SHAP-Geichtung berechnen ===
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Absolute_SHAP': mean_abs_shap,
    'SHAP_%': 100 * mean_abs_shap / mean_abs_shap.sum()
}).sort_values('SHAP_%', ascending=False).round(2)

print("\n📈 SHAP-basierte Feature-Wichtigkeiten (%):")
print(shap_df)

# === 13. Modellbewertung (zusätzlich zu R²) ===
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Modellbewertung (Random Forest):")
print(f" - R² (Erklärte Varianz):     {r2:.4f}")
print(f" - MAE (durchschnittl. Fehler): {mae:.2f}")
print(f" - MSE (quadratischer Fehler): {mse:.2f}")
print(f" - RMSE (Wurzel aus MSE):      {rmse:.2f}")
