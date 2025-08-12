import joblib, os
d = joblib.load("product_model_paths.pkl")
new_d = {}
for name, p in d.items():
    base = os.path.basename(p)
    new_d[name] = os.path.join("product_models", base).replace("\\", "/")
joblib.dump(new_d, "product_model_paths.pkl")
print("rewritten:", len(new_d), "entries")
