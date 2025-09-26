
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 読み込むCSVファイル名（同じディレクトリに置くこと）
DATA_PATH = "商品別売上_統合_統合済v1.13.csv"

# 保存先ディレクトリ
MODEL_DIR = "product_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 特徴量とターゲット
FEATURE_COLS = [
    "曜日", "祝日", "最高気温", "最低気温", "天気", "休日フラグ", "特異日フラグ",
    "月", "季節", "イベント有無", "長期休みの種類", "長期休みフラグ",
    "前週同曜日_売上", "売上_移動平均7日", "売上"
]
TARGET_COL = "商品数"

def main():
    df = pd.read_csv(DATA_PATH)
    product_model_paths = {}

    for product_name, group in df.groupby("商品名"):
        if len(group) < 10:
            print(f"[スキップ] {product_name}（データ不足）")
            continue

        X = group[FEATURE_COLS].copy()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = group[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = os.path.join(MODEL_DIR, f"{product_name}.pkl")
        joblib.dump(model, model_path)
        product_model_paths[product_name] = model_path

        print(f"[保存完了] {product_name} → {model_path}")

    # モデルパス辞書を保存
    joblib.dump(product_model_paths, "product_model_paths.pkl")
    print("✅ product_model_paths.pkl を保存しました")

if __name__ == "__main__":
    main()
