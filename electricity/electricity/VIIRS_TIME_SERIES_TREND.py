import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 📁 BASE PATH
# -----------------------------
base_path = r"C:\Users\hp\Desktop\Electricity_journal"

# -----------------------------
# 🔥 FILES
# -----------------------------
viirs_files = {
    year: os.path.join(base_path, f"zimbabwe_electricity_{year}.tif")
    for year in range(2015, 2025)
}

# -----------------------------
# 🔥 PROCESS
# -----------------------------
years = []
mean_radiance = []

for year, path in viirs_files.items():
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    with rasterio.open(path) as src:
        data = src.read(1)
        data = np.where(data < 0, np.nan, data)

    mean_val = np.nanmean(data)

    years.append(year)
    mean_radiance.append(mean_val)

# -----------------------------
# 📊 PLOT
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(years, mean_radiance, marker='o', linewidth=2)

plt.title('VIIRS Mean Radiance Trend (2015–2024)')
plt.xlabel('Year')
plt.ylabel('Mean Radiance')
plt.grid(True)
plt.xticks(years)

plt.show()