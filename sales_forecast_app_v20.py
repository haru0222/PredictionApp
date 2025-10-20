# sales_forecast_app_v20.py
# 予測結果を画面下に「エクセルにコピペできる表」で出力（Excel書き込みはしない）

import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
import os
import numpy as np
import jpholiday

# 追加インポート（世界の祝日）
try:
    import holidays as pyholidays
except Exception:
    pyholidays = None

from functools import lru_cache
import datetime as dt

# 国コード→表示名（必要な国はここに足せます）
COUNTRIES = {
    "JP": "日本", "CN": "中国", "US": "アメリカ", "KR": "韓国",
    "TW": "台湾", "HK": "香港", "SG": "シンガポール", "TH": "タイ", "VN": "ベトナム",
    "MY": "マレーシア", "ID": "インドネシア", "PH": "フィリピン",
    "GB": "イギリス", "FR": "フランス", "DE": "ドイツ", "IT": "イタリア", "ES": "スペイン",
    "CA": "カナダ", "AU": "オーストラリア", "IN": "インド", "BR": "ブラジル"
}

# 地域差が大きい国の州・省コードを必要なら指定（例：US:CA など）
SUBDIV = {
    # "US": "CA",
    # "CN": None,
}

# ========= 日本の長期休み・繁忙期ユーティリティ =========
def get_all_long_holidays(year):
    import datetime as dt
    obon_days = [dt.date(year, 8, d) for d in range(13, 17)]
    all_days = [dt.date(year, 1, 1) + dt.timedelta(days=i) for i in range(370)]

    def is_holiday_like(d):
        return jpholiday.is_holiday(d) or d.weekday() >= 5 or d in obon_days

    long_holidays = set()
    current_block = []
    for d in all_days:
        if is_holiday_like(d):
            current_block.append(d)
        else:
            if len(current_block) >= 3:
                long_holidays.update(current_block)
            current_block = []
    if len(current_block) >= 3:
        long_holidays.update(current_block)

    # 学校の長期休み
    summer = [dt.date(year, 7, d) for d in range(20, 32)] + [dt.date(year, 8, d) for d in range(1, 32)]
    winter = [dt.date(year, 12, d) for d in range(25, 32)] + [dt.date(year + 1, 1, d) for d in range(1, 8)]
    spring = [dt.date(year, 3, d) for d in range(20, 32)] + [dt.date(year, 4, d) for d in range(1, 6)]
    long_holidays.update(summer + winter + spring)
    return long_holidays

@lru_cache(maxsize=8)
def _long_holiday_days_for_year(year: int):
    return get_all_long_holidays(year)

def is_long_holiday(date):
    # 年越し対応（選択日付の年で判定）
    return date in _long_holiday_days_for_year(date.year)

def is_crowded_day(date):
    y = date.year
    return (
        datetime.date(y, 7, 20) <= date <= datetime.date(y, 8, 31) or
        datetime.date(y, 12, 25) <= date <= datetime.date(y + 1, 1, 7) or
        datetime.date(y, 3, 20) <= date <= datetime.date(y, 4, 5) or
        datetime.date(y, 8, 13) <= date <= datetime.date(y, 8, 16) or
        datetime.date(y, 4, 29) <= date <= datetime.date(y, 5, 6)
    )

# ========= 世界の祝日・長期連休ユーティリティ =========
@lru_cache(maxsize=256)
def _country_holidays_cached(code: str, year: int, subdiv):
    """holidays.CountryHoliday の構築を年×国×subdivでキャッシュ"""
    if pyholidays is None:
        return None
    try:
        h = pyholidays.CountryHoliday(code, years=[year, year + 1], subdiv=subdiv)
        return h
    except Exception:
        return None

def get_international_holidays(date):
    """date に該当する各国の祝日名を ['中国：春节', 'アメリカ：Independence Day', ...] 形式で返す"""
    results = []
    if pyholidays is None:
        return results
    for code, label in COUNTRIES.items():
        h = _country_holidays_cached(code, date.year, SUBDIV.get(code))
        if not h:
            continue
        if date in h:
            names = h.get(date)
            if isinstance(names, (list, tuple, set)):
                name_str = "・".join(map(str, names))
            else:
                name_str = str(names)
            results.append(f"{label}：{name_str}")
    return results

def is_long_holiday_in_country(date, country_code):
    """週末 or その国の公休日 を『休日らしい日』とみなし、3日以上連続に date が含まれるなら True"""
    if country_code == "JP":
        # 日本は既存ロジック（お盆・学校休み含む）を優先
        return is_long_holiday(date)
    if pyholidays is None:
        return False

    y = date.year
    hol = _country_holidays_cached(country_code, y, SUBDIV.get(country_code))
    holiday_set = set(hol.keys()) if hol else set()

    # 年をまたぐ可能性あり：当年+前後を含めて走査
    start = dt.date(y, 1, 1) - dt.timedelta(days=7)
    days = [start + dt.timedelta(days=i) for i in range(370 + 14)]

    def is_holiday_like(d):
        return (d in holiday_set) or (d.weekday() >= 5)

    block = []
    for d in days:
        if is_holiday_like(d):
            block.append(d)
        else:
            if len(block) >= 3 and date in block:
                return True
            block = []
    # 末尾ブロック
    if len(block) >= 3 and date in block:
        return True
    return False

# ========= モデル・各種データ読み込み =========
sales_model = joblib.load("sales_model.pkl")
product_model_paths = joblib.load("product_model_paths.pkl")
product_models = {name: joblib.load(path) for name, path in product_model_paths.items() if os.path.exists(path)}

df_menu = pd.read_csv("商品別売上_統合_統合済v1.13.csv")
constant_items = df_menu[df_menu["恒常メニュー"] == 1]["商品名"].unique().tolist()
seasonal_items_all = df_menu[df_menu["シーズンメニュー"] == 1]["商品名"].unique().tolist()

API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
CITY_NAME = st.secrets.get("CITY_NAME", "Odaiba,JP")
df_prev_weather = pd.read_csv("前年_東京羽田_天気気温.csv")
df_prev_weather["date"] = pd.to_datetime(df_prev_weather["date"])

def fetch_weather_forecast(date):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY_NAME}&appid={API_KEY}&units=metric&lang=ja"
    try:
        data = requests.get(url, timeout=10).json()
        for item in data.get("list", []):
            dt_ = datetime.datetime.fromtimestamp(item["dt"])
            if dt_.date() == date:
                return item["weather"][0]["main"], item["main"]["temp_max"], item["main"]["temp_min"]
    except Exception:
        pass
    return None, None, None

def make_features(entry):
    date = entry["date"]
    weekday = date.weekday()
    holiday = int(jpholiday.is_holiday(date))
    month = date.month
    season = 0 if month in [3, 4, 5] else 1 if month in [6, 7, 8] else 2 if month in [9, 10, 11] else 3
    tokui_flag = int((date.month == 2 and date.day == 14) or (date.month == 12 and date.day == 25))
    long_holiday_flag = int(is_long_holiday(date))
    busy_flag = int(is_crowded_day(date))
    weather_map = {"Clear": 0, "Clouds": 1, "Rain": 2, "晴れ": 0, "曇り": 1, "雨": 2}

    df_feat = pd.DataFrame([{
        "曜日": weekday,
        "祝日": holiday,
        "最高気温": entry["temp_max"],
        "最低気温": entry["temp_min"],
        "天気": weather_map.get(entry["weather"], 0),
        "休日フラグ": int(weekday >= 5 or holiday == 1),
        "特異日フラグ": tokui_flag,
        "月": month,
        "季節": season,
        "イベント有無": entry["event"],
        "長期休みの種類": 0,
        "長期休みフラグ": long_holiday_flag,
        "繁忙期フラグ": busy_flag,
        "前週同曜日_売上": 200000,
        "売上_移動平均7日": 200000,
        "売上": 200000
    }]).apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_feat

# ========= 出力列（ご指定の順） =========
BASE_COLUMNS = ["日付", "曜日", "天気", "最高気温", "最低気温", "予測売上"]
FIXED_PRODUCT_COLUMNS = [
    "01 PBA金の房プレミアムバナナミルク",
    "02 BA完熟バナナミルク",
    "03 STBAつぶつぶいちごバナナミルク",
    "04 MIXトロピカルマンゴーミックス",
    "05 KBケールバナナ",
    "06 ACAIアサイースムージー",
    "07 OP果実たっぷりオレンジパイン",
    "08 KOPつぶつぶキウイオレンジパイン",
    "09 BOCベリーベリーオレンジココナッツ",
    "10 MGつぶつぶマンゴーミルク",
    "11 KIWIごろごろキウイ",
    "12 LMNゴクゴクレモネードソーダ",
    "13 LLS搾りたてレモンライムソーダ",
    "14 PGS搾りたてピンクグレープフルーツソーダ",
    "BLS ブルーレモンソーダ",
    "SS 東京サンセットソーダ",
    "GY グリークヨーグルト",
    "SU100 君島農園すいか100%生絞りジュース",
    "MS つぶつぶメロンシェイク",
    "PS桃スムージー",
    "SU100 あべ農園すいか100%生絞りジュース",
    "NS100 切りたて梨100%生搾りジュース",
    "KPS まるごと巨峰とパインスムージー",
    "MK100 極早生みかん果汁100%ジュース",
    "IMO 蜜いもミルクシェイク",
]

# ========= UI =========
st.set_page_config(page_title="売上・商品数予測アプリ", layout="wide")
st.title("売上・商品数予測アプリ")

selected_dates = st.date_input("予測したい日付を選択（複数可）", [], format="YYYY-MM-DD")

if isinstance(selected_dates, tuple):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
        selected_dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    else:
        selected_dates = list(selected_dates)
elif isinstance(selected_dates, datetime.date):
    selected_dates = [selected_dates]

# ---- 世界の祝日プレビュー（天気入力の前）----
if selected_dates:
    st.write("### 🌍 国横断：その日がどこの国の祝日・長期連休に当たるか（プレビュー）")

    if pyholidays is None:
        st.info("世界の祝日判定には 'holidays' パッケージが必要です。requirements.txt に 'holidays>=0.57' を追加後、再実行してください。")

    rows = []
    for d in selected_dates:
        hits = get_international_holidays(d)  # 祝日名ヒット（国名：祝日名）
        long_hits = []
        # 全対象国で「長期連休」ヒットを拾う（祝日名が無くても週末合体で3連休+ならヒットさせる）
        for code, label in COUNTRIES.items():
            try:
                if is_long_holiday_in_country(d, code):
                    long_hits.append(f"{label}：長期連休")
            except Exception:
                continue

        if not hits and not long_hits:
            status = "該当なし"
        else:
            status = " / ".join(hits + long_hits)

        rows.append({
            "日付": d.strftime("%Y-%m-%d"),
            "該当国の祝日・長期連休": status
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 詳細（国別）を折り畳みで
    with st.expander("国別の詳細（祝日名／長期連休ヒット）"):
        detail_rows = []
        for d in selected_dates:
            # 1回だけ取得してから国別に整形（無駄な再計算を減らす）
            names_all = get_international_holidays(d)  # ['中国：春节', 'アメリカ：Independence Day', ...]
            by_country = {}
            for s in names_all:
                if "：" in s:
                    label, name = s.split("：", 1)
                    by_country[label] = (by_country.get(label, []) + [name])
            for code, label in COUNTRIES.items():
                long_f = is_long_holiday_in_country(d, code)
                if (label in by_country) or long_f:
                    detail_rows.append({
                        "日付": d.strftime("%Y-%m-%d"),
                        "国": label,
                        "祝日名": " / ".join(by_country.get(label, [])),
                        "長期連休": "◯" if long_f else ""
                    })
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
        else:
            st.info("該当なし")

# ---- 以降は既存どおり（天気プレビュー→入力→予測）----
selected_season = []
if selected_dates:
    st.write("### 🌤️ 選択日付の天気と気温（前年データも含む）")
    weather_rows = []
    today = datetime.date.today()

    for date in selected_dates:
        delta = (date - today).days
        if delta <= 5:
            weather, temp_max, temp_min = fetch_weather_forecast(date)
        else:
            weather, temp_max, temp_min = "", "", ""

        prev_date = date.replace(year=date.year - 1)
        prev = df_prev_weather[df_prev_weather["date"] == pd.Timestamp(prev_date)]
        if not prev.empty:
            prev_weather = prev.iloc[0]["weather"]
            prev_max = prev.iloc[0]["temp_max"]
            prev_min = prev.iloc[0]["temp_min"]
        else:
            prev_weather, prev_max, prev_min = "", "", ""

        weather_rows.append({
            "日付": date.strftime("%Y-%m-%d"),
            "天気": weather,
            "最高気温": temp_max,
            "最低気温": temp_min,
            "昨年の天気": prev_weather,
            "昨年の最高気温": prev_max,
            "昨年の最低気温": prev_min,
        })

    st.dataframe(pd.DataFrame(weather_rows), use_container_width=True)

    st.write("### 各日付の情報入力")
    date_inputs = []
    for date in selected_dates:
        with st.expander(f"{date.strftime('%Y-%m-%d')} の設定"):
            delta = (date - today).days
            if delta <= 7:
                weather, temp_max, temp_min = fetch_weather_forecast(date)
                if weather is None:
                    st.warning("天気取得失敗。手動で入力してください。")
                    temp_max = st.number_input("最高気温", key=f"max_manual_{date}", step=1, value=20)
                    temp_min = st.number_input("最低気温", key=f"min_manual_{date}", step=1, value=20)
                    weather = st.selectbox("天気", ["晴れ", "曇り", "雨"], key=f"weather_manual_{date}")
                else:
                    # 英語天気→和名候補
                    candidates = ["晴れ", "曇り", "雨"]
                    default = 0
                    if weather in ["Clear", "Clouds", "Rain"]:
                        default = ["Clear", "Clouds", "Rain"].index(weather)
                    weather = st.selectbox("天気", candidates, index=default, key=f"weather_{date}")
                    temp_max = st.number_input("最高気温", key=f"max_{date}", value=int(temp_max) if temp_max else 20)
                    temp_min = st.number_input("最低気温", key=f"min_{date}", value=int(temp_min) if temp_min else 20)
            else:
                weather = st.selectbox("天気", ["晴れ", "曇り", "雨"], key=f"weather_{date}")
                temp_max = st.number_input("最高気温", key=f"max_{date}", value=20)
                temp_min = st.number_input("最低気温", key=f"min_{date}", value=20)
            event = st.checkbox("イベント有無", key=f"event_{date}")
            date_inputs.append({"date": date, "weather": weather, "temp_max": temp_max, "temp_min": temp_min, "event": int(event)})

    st.write("### シーズンメニュー選択")
    with st.expander("シーズンメニューを選ぶ"):
        selected_season = st.multiselect("出力するシーズンメニューを選んでください", seasonal_items_all)

# ========= 実行ボタン =========
if st.button("予測を実行"):
    if not selected_dates:
        st.warning("日付を選択してください。")
        st.stop()

    rows_for_table = []
    all_products_used = set()  # 予測で触れた全商品（列追加のため）

    for entry in date_inputs:
        date_str = entry['date'].strftime('%Y-%m-%d')
        feat = make_features(entry)

        # 売上予測（※ multiplier はご提示どおり 1.0 のまま）
        raw_sales = sales_model.predict(feat.drop(columns=["売上", "繁忙期フラグ"]))[0]
        multiplier = 1.0 if feat.at[0, "繁忙期フラグ"] == 1 else 1.0
        pred_sales = int(raw_sales * multiplier)

        # 商品数量予測
        feat["売上"] = pred_sales
        qty_dict = {}
        for item in (constant_items + selected_season):
            if item in product_models:
                qty = int(product_models[item].predict(feat.drop(columns=["繁忙期フラグ"]))[0])
                qty_dict[item] = qty
            # モデルなしは未出力にしてOK（列は後で作るが値は空）
            all_products_used.add(item)

        # 行データ（基本項目）
        weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][entry["date"].weekday()]
        row = {
            "日付": date_str,
            "曜日": weekday_jp,
            "天気": entry["weather"],
            "最高気温": int(entry["temp_max"]) if entry["temp_max"] not in ["", None] else "",
            "最低気温": int(entry["temp_min"]) if entry["temp_min"] not in ["", None] else "",
            "予測売上": pred_sales,
        }

        # 先に固定列（指定順）を埋める
        for col in FIXED_PRODUCT_COLUMNS:
            row[col] = qty_dict.get(col, "")

        # 後ろに“その他商品”を並べる（PS桃スムージーの後）
        others = sorted(list(all_products_used - set(FIXED_PRODUCT_COLUMNS)))
        for prod in others:
            row[prod] = qty_dict.get(prod, "")

        rows_for_table.append(row)

    # 列構成：基本 → 固定商品 → その他商品（PS桃スムージーの後）
    all_other_products = sorted(list((all_products_used - set(FIXED_PRODUCT_COLUMNS))))
    final_columns = BASE_COLUMNS + FIXED_PRODUCT_COLUMNS + all_other_products

    df_out = pd.DataFrame(rows_for_table)
    # 足りない列は作って順序を揃える
    for c in final_columns:
        if c not in df_out.columns:
            df_out[c] = ""
    df_out = df_out[final_columns]

    st.write("## 📋 コピペ用の結果表（このままExcelへ貼り付け可）")
    st.dataframe(df_out, use_container_width=True)

    st.write("#### タブ区切りテキスト（Ctrl/Cmd + A → コピー → Excelに貼り付け）")
    tsv = df_out.to_csv(sep="\t", index=False)
    st.text_area("TSV", tsv, height=200)

    st.download_button(
        "CSVをダウンロード",
        data=df_out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"pred_{datetime.date.today().isoformat()}.csv",
        mime="text/csv"
    )
