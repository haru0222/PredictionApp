# train_sales.py
import sys
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

DATA_PATH = "商品別売上_統合_統合済v1.13.csv"

SALES_FEATURES = [
    "曜日","祝日","最高気温","最低気温","天気",
    "休日フラグ","特異日フラグ","月","季節",
    "イベント有無","長期休みの種類","長期休みフラグ",
    "前週同曜日_売上","売上_移動平均7日"
]
TARGET = "売上"

def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] CSVの読み込みに失敗しました: {e}")
        sys.exit(1)

    missing = [c for c in SALES_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print("[ERROR] 学習に必要な列が見つかりません:", ", ".join(missing))
        sys.exit(1)

    X = df[SALES_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(X_train, y_train)

    # --- 精度確認 ---
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.1f}")
    print(f"R^2 : {r2:.3f}")

    # --- 保存 ---
    joblib.dump(model, "sales_model.pkl")
    print("✅ sales_model.pkl を出力しました")
    print(f"  学習データ: {DATA_PATH}")
    print(f"  特徴量    : {len(SALES_FEATURES)}列 / サンプル数: {len(X)}")

if __name__ == "__main__":
    main()
