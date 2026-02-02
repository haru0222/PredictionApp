# train_models_from_v113.py
# v1.13 CSVから
# 1) 日別総売上モデル sales_model.pkl
# 2) 商品別数量モデル product_models/*.pkl と product_model_paths.pkl
# を作る

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


CSV_PATH = "商品別売上_統合_統合済v1.13.csv"

OUT_SALES_MODEL = "sales_model.pkl"
OUT_PRODUCT_DIR = "product_models"
OUT_PRODUCT_PATHS = "product_model_paths.pkl"

RANDOM_STATE = 42


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 日付
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df = df.dropna(subset=["日付"]).sort_values("日付").reset_index(drop=True)

    # 天気を数値へ（既に数値ならそのまま）
    # v1.13は「晴れ/曇り/雨」想定。もし英語が混ざっても対応。
    weather_map = {"晴れ": 0, "曇り": 1, "雨": 2, "Clear": 0, "Clouds": 1, "Rain": 2}
    if df["天気"].dtype == "object":
        df["天気"] = df["天気"].map(weather_map).fillna(0)

    # 数値化（壊れた値はNaN→0）
    num_cols = [
        "曜日", "祝日", "最高気温", "最低気温", "天気",
        "休日フラグ", "特異日フラグ", "月", "季節", "イベント有無",
        "長期休みの種類", "長期休みフラグ",
        "前週同曜日_売上", "売上_移動平均7日",
        "売上", "商品数", "販売商品数",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 欠損埋め（気温など欠損は0ではなく中央値が良いが、ここは安全側で中央値にする）
    for c in ["最高気温", "最低気温", "前週同曜日_売上", "売上_移動平均7日"]:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # その他は0埋め
    fill_zero = [
        "曜日", "祝日", "天気", "休日フラグ", "特異日フラグ", "月", "季節", "イベント有無",
        "長期休みの種類", "長期休みフラグ",
    ]
    for c in fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def make_daily_sales_table(df: pd.DataFrame) -> pd.DataFrame:
    # 日別総売上（全商品売上の合計）
    daily_sales = df.groupby("日付", as_index=False)["売上"].sum().rename(columns={"売上": "日別総売上"})

    # 日付ごとの特徴量は「同じ日の行なら同値」のはずなので先頭を採用
    feat_cols = [
        "曜日", "祝日", "最高気温", "最低気温", "天気",
        "休日フラグ", "特異日フラグ", "月", "季節", "イベント有無",
        "長期休みの種類", "長期休みフラグ",
        "前週同曜日_売上", "売上_移動平均7日",
    ]
    daily_feat = df.groupby("日付", as_index=False)[feat_cols].first()

    daily = pd.merge(daily_feat, daily_sales, on="日付", how="inner").sort_values("日付").reset_index(drop=True)
    return daily


def train_xgb_regressor_time_series(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> XGBRegressor:
    # ざっくり強めの汎用設定（過学習しにくい寄り）
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        tree_method="hist",
    )

    # 時系列CVで早期打ち切りしながら学習
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model = None
    best_mae = float("inf")

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    for fold, (tr, va) in enumerate(tscv.split(X_np), start=1):
        X_tr, y_tr = X_np[tr], y_np[tr]
        X_va, y_va = X_np[va], y_np[va]

        m = XGBRegressor(**model.get_params())
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )

        pred = m.predict(X_va)
        mae = mean_absolute_error(y_va, pred)
        # print(f"[fold {fold}] MAE={mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_model = m

    # 最良モデルを返す
    return best_model


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSVが見つかりません: {CSV_PATH}")

    df = load_data(CSV_PATH)

    # ---------- 1) 売上モデル ----------
    daily = make_daily_sales_table(df)

    sales_feature_cols = [
        "曜日", "祝日", "最高気温", "最低気温", "天気",
        "休日フラグ", "特異日フラグ", "月", "季節", "イベント有無",
        "長期休みの種類", "長期休みフラグ",
        "前週同曜日_売上", "売上_移動平均7日",
    ]
    X_sales = daily[sales_feature_cols]
    y_sales = daily["日別総売上"]

    sales_model = train_xgb_regressor_time_series(X_sales, y_sales, n_splits=5)
    joblib.dump({"model": sales_model, "feature_cols": sales_feature_cols}, OUT_SALES_MODEL)

    # ---------- 2) 商品別数量モデル ----------
    os.makedirs(OUT_PRODUCT_DIR, exist_ok=True)

    # 商品数列の決定（優先：商品数 → なければ販売商品数）
    if "商品数" in df.columns and df["商品数"].notna().any():
        qty_col = "商品数"
    else:
        qty_col = "販売商品数"

    # 日別総売上を各商品行へ付与して、商品モデルの説明変数に「売上」を入れる
    daily_sales_map = daily[["日付", "日別総売上"]].rename(columns={"日別総売上": "売上"})
    df2 = pd.merge(df, daily_sales_map, on="日付", how="left", suffixes=("", "_daily"))

    product_feature_cols = sales_feature_cols + ["売上"]

    product_paths = {}
    summary = []

    for product_name, g in df2.groupby("商品名"):
        g = g.sort_values("日付").dropna(subset=[qty_col, "売上"])

        # データが少なすぎる商品はスキップ（学習が不安定）
        if len(g) < 15:
            continue

        X_p = g[product_feature_cols]
        y_p = g[qty_col]

        m = train_xgb_regressor_time_series(X_p, y_p, n_splits=5)

        safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(product_name))[:120]
        out_path = os.path.join(OUT_PRODUCT_DIR, f"{safe_name}.pkl").replace("\\", "/")

        joblib.dump({"model": m, "feature_cols": product_feature_cols, "product_name": product_name}, out_path)
        product_paths[product_name] = out_path

        summary.append({"商品名": product_name, "rows": len(g)})

    joblib.dump(product_paths, OUT_PRODUCT_PATHS)

    # 学習サマリを出力（確認用）
    info = {
        "sales_model": OUT_SALES_MODEL,
        "product_models": len(product_paths),
        "product_dir": OUT_PRODUCT_DIR,
        "qty_target_col": qty_col,
        "daily_rows": len(daily),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))

    # 代表で数件だけ表示
    summary = sorted(summary, key=lambda x: -x["rows"])[:10]
    print("top products by rows:")
    for s in summary:
        print(f"- {s['商品名']} ({s['rows']})")


if __name__ == "__main__":
    main()
