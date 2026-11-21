"""Generate the SYNTHETIC teaching dataset 16_crop_yield.csv.

Run:  python generate_dataset.py   (creates 16_crop_yield.csv in this folder)
Fixed random seed, so the same file is produced every time.
The data deliberately contains missing values, duplicate IDs, zero-area rows
and outliers so every cleaning step of the analysis is exercised.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 600

crops = {  # crop: (base yield t/ha, usual season)
    "Rice": (4.2, "Kharif"), "Wheat": (3.6, "Rabi"), "Maize": (5.0, "Kharif"),
    "Cotton": (2.0, "Kharif"), "Soybean": (2.6, "Kharif"), "Mustard": (1.8, "Rabi"),
}
regions = ["North", "South", "East", "West", "Central"]
soils = {"Loamy": 1.10, "Clay": 1.00, "Sandy": 0.88, "Silty": 1.05}

crop = rng.choice(list(crops), N)
season = [crops[c][1] if rng.random() < 0.85 else rng.choice(["Kharif", "Rabi", "Zaid"]) for c in crop]
soil = rng.choice(list(soils), N)
region = rng.choice(regions, N)
irrig = rng.choice(["Irrigated", "Rain-fed"], N, p=[0.55, 0.45])
rain = np.clip(rng.normal(780, 220, N), 200, 1500).round(1)
fert = np.clip(rng.normal(110, 40, N), 10, 250).round(1)
area = np.round(rng.uniform(0.5, 25, N), 2)
ph = np.clip(rng.normal(6.8, 0.7, N), 4.5, 9.0).round(2)

base = np.array([crops[c][0] for c in crop])
yld = (base * np.array([soils[s] for s in soil])
       * (1 + 0.00025 * (rain - 780)) * (1 + 0.0012 * (fert - 110))
       * np.where(irrig == "Irrigated", 1.12, 1.0) * rng.normal(1, 0.07, N))
prod = (yld * area).round(2)

df = pd.DataFrame({
    "Farm_Record_ID": [f"FR{i:04d}" for i in range(1, N + 1)],
    "Region": region, "Crop_Type": crop, "Season": season, "Soil_Type": soil,
    "Irrigation_Type": irrig, "Rainfall_mm": rain, "Soil_pH": ph,
    "Fertilizer_kg_per_ha": fert, "Area_Hectares": area, "Production_Tons": prod,
})

# --- inject data-quality problems ---
df.loc[rng.choice(N, 25, replace=False), "Rainfall_mm"] = np.nan
df.loc[rng.choice(N, 20, replace=False), "Soil_pH"] = np.nan
df.loc[rng.choice(N, 4, replace=False), "Area_Hectares"] = 0
out = rng.choice(N, 8, replace=False)
df.loc[out[:4], "Production_Tons"] *= 3.2      # unusually high
df.loc[out[4:], "Production_Tons"] *= 0.2      # unusually low
df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)  # duplicate IDs
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("16_crop_yield.csv", index=False)
print("Saved 16_crop_yield.csv", df.shape)
