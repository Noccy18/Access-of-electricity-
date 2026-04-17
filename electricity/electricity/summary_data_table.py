import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("final_dataset.csv")

# ===============================
# DROP LEAKAGE SOURCE
# ===============================
# 🚨 DO NOT USE radiance to predict electricity
X = df[["population", "built_up", "urban"]]
y = df["electricity"]

# ===============================
# SCALE FEATURES (CRITICAL)
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ===============================
# ADD CONSTANT
# ===============================
X_scaled = sm.add_constant(X_scaled)

# ===============================
# FIT MODEL (REGULARIZED = STABLE)
# ===============================
model = sm.Logit(y, X_scaled).fit_regularized()

# ===============================
# OUTPUT
# ===============================
print(model.summary())