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

# --- 日本語サイトのイベント プレビュー（ビッグサイト/ダイバーシティ/お台場） ---
import re
from functools import lru_cache
from bs4 import BeautifulSoup

# 収集対象URL（日本語のみ）
EVENT_SOURCES_JP = [
    # 東京ビッグサイト（イベント情報）
    ("https://www.bigsight.jp/visitor/event/", "東京ビッグサイト"),
    # ダイバーシティ東京プラザ（イベント・キャンペーン）
    ("https://mitsui-shopping-park.com/divercity-tokyo/event/", "ダイバーシティ東京プラザ"),
    # お台場 公式ポータル（イベント一覧・カレンダー）
    ("https://www.tokyo-odaiba.net/event_index/", "お台場（公式一覧）"),
    ("https://www.tokyo-odaiba.net/event_calender/", "お台場（公式カレンダー）"),
    # Zepp DiverCity（ライブスケジュール）
    ("https://www.zepp.co.jp/hall/divercity/schedule/", "Zepp DiverCity"),
]

# --------------------------------------------
# 大規模イベントだけに絞るフィルタ関数
# --------------------------------------------
BIG_EVENT_KEYWORDS = [
    "フェス", "フェスタ", "祭",
    "博", "博覧会", "展示会", "見本市",
    "エキスポ", "EXPO",
    "コミックマーケット", "コミケ",
    "フェア", "ショー",
    "花火","花火大会","HANABI",
]

def _filter_big_events(events):
    """
    本当に人が多く来る大規模イベントだけに絞るフィルタ
    ・UIテキスト/商談系を除外
    ・お台場たこ焼きミュージアム系など常設寄り企画を除外
    ・デザインフェスタ／アミューズメントエキスポ等はタイトルを正規化して重複削除
    ・同じイベント（会場×タイトル）は「期間が一番短いもの」だけ残す
      → デザフェスみたいに 10/04〜11/15 とか 11/14〜11/30 が混在しても、
         本番の 11/15〜11/16 だけを採用する狙い
    """
    # サイトのUIテキスト・検索説明・商談系など、明らかにイベント名じゃないもの
    noise_keywords = [
        "アクセス", "フロアマップ", "イベント情報", "ショップ＆レストラン",
        "イベント検索", "日付検索", "検索結果",
        "カレンダー", "カレンダーから探す",
        "ジャンル", "条件選択",
        "カテゴリーから探す", "キーワードから探す",
        "年間の主要イベント",
        "イベント・キャンペーン",
        "入場区分",     # 商談系のヘッダごと全部カット
        "開催期間",     # 「開催期間 2025年…」だけの行をカット
        "開催時間",     # 時間だけの行
        "商談日時",     # 商談日時だけの行
        # ※「エ リ ア マ ッ プ」はタイトルにも含まれるのでここには入れない
    ]

    # 日常寄りのロングラン企画（売上にあまり効かなそうなもの）
    # ただしタイトルに「東京ビッグサイト」が含まれる場合は残したいので条件で絞る
    small_odaiba_keywords = [
        "お台場たこ焼きミュージアム",
        "台場一丁目商店街",
        "デックス東京ビーチ",
    ]

    # key = (会場, 正規化タイトル) → (期間日数, イベントdict)
    # で管理して、「同じイベント」は一番短い期間だけ残す
    best_events = {}

    for ev in events:
        title = ev.get("イベント（抜粋）", "")
        venue = ev.get("会場", "")
        start_s = ev.get("開始日") or ""
        end_s = ev.get("終了日") or ""

        # 0) タイトルがスカスカなら捨てる
        if not title or len(title) < 6:
            continue

        # 1) UIテキスト・説明文っぽいものは即除外
        if any(k in title for k in noise_keywords):
            continue

        # 1.5) 「2025-11-14 ～ 2025-11-15 東京ビッグサイト」みたいな
        #      日付＋会場だけで、イベント名が無さそうな行を除外
        if re.match(r"\d{4}-\d{2}-\d{2}\s*～\s*\d{4}-\d{2}-\d{2}", title):
            continue

        # 2) 期間（日数）を計算（失敗したら 9999 日扱い）
        try:
            start = datetime.date.fromisoformat(start_s)
            end = datetime.date.fromisoformat(end_s)
            days = (end - start).days + 1
        except Exception:
            start = end = None
            days = 9999

        # 3) お台場の常設寄り企画をカット
        #    ただしタイトルに「東京ビッグサイト」がある場合は残す
        if any(k in title for k in small_odaiba_keywords) and "東京ビッグサイト" not in title:
            continue

        # 4) どの程度「大きいイベント」とみなすか
        keep = False

        # 4-1) 東京ビッグサイト本体 or タイトルにビッグサイトと書いてある → 大規模扱いで残す
        if venue == "東京ビッグサイト" or "東京ビッグサイト" in title:
            keep = True
        else:
            # 4-2) それ以外の会場（ダイバーシティ・防災公園など）
            #      10日以上続く長期企画は、常設寄りとして除外
            if days >= 10:
                keep = False
            else:
                # 「フェス」「エキスポ」など “イベントっぽい” キーワードがあるものだけ残す
                if any(k in title for k in BIG_EVENT_KEYWORDS):
                    keep = True

        if not keep:
            continue

        # 5) タイトル正規化（重複削除用＋表示用の整理）
        norm_title = title

        # デザフェス
        if "デザインフェスタ" in norm_title:
            norm_title = "デザインフェスタ vol.62 ＜東京ビッグサイト＞"
        # アミューズメントエキスポ
        elif "アミューズメント エキスポ" in norm_title:
            norm_title = "アミューズメント エキスポ 2025 ＜東京ビッグサイト＞"
        # プロジェクションマッピング
        elif "プロジェクションマッピングアワード" in norm_title:
            norm_title = "東京国際プロジェクションマッピングアワード Vol.10"
        # 防災フェスタ
        elif "防災フェスタ" in norm_title:
            norm_title = "防災フェスタ2025「備蓄を考える」"

        key = (venue, norm_title)

        # すでに同じイベントが登録されている場合は、
        # 「期間が短い方」を優先して残す（本番期間を採用したい）
        prev = best_events.get(key)
        if prev is None or days < prev[0]:
            ev_new = dict(ev)  # 元のdictを壊さないようにコピー
            ev_new["イベント（抜粋）"] = norm_title
            best_events[key] = (days, ev_new)

    # dict からイベントだけ取り出して返す
    return [v[1] for v in best_events.values()]

# 日付表記のゆれに対応した正規表現（日本語寄り）
# 例：2025/10/1～2025/10/3, 2025年10月1日〜3日, 10/01(水)〜10/03(金) など
RANGE_PATTERNS = [
    r"(?P<y1>\d{4})[./年\-](?P<m1>\d{1,2})[./月\-](?P<d1>\d{1,2})[日]?\s*[～\-–~〜至からto～～─―]+\s*(?P<y2>\d{4})[./年\-](?P<m2>\d{1,2})[./月\-](?P<d2>\d{1,2})[日]?",
    r"(?P<m1>\d{1,2})[./月\-](?P<d1>\d{1,2})[日]?\s*[～\-–~〜]+\s*(?P<m2>\d{1,2})[./月\-](?P<d2>\d{1,2})[日]?(\s*\((?P<w2>.)\))?",
]
SINGLE_PATTERNS = [
    r"(?P<y>\d{4})[./年\-](?P<m>\d{1,2})[./月\-](?P<d>\d{1,2})[日]?",
    r"(?P<m>\d{1,2})[./月\-](?P<d>\d{1,2})[日]?(?:\((?P<w>.)\))?",
]

def _to_date(y, m, d):
    return datetime.date(int(y), int(m), int(d))

def _normalize_date_str(s: str) -> str:
    # 全角や和文区切りをざっくりASCII寄せ
    return (
        s.replace("年", "/").replace("月", "/").replace("日", "")
         .replace("．", ".").replace("ー", "-").replace("―", "-")
         .replace("～", "~").replace("〜", "~").replace("：", ":")
    )

def _extract_date_ranges_jp(text: str, base_year: int):
    """テキストから (start, end) の日付レンジ配列を抽出（単日は start=end）"""
    t = _normalize_date_str(text)

    ranges = []

    # 範囲表記
    for pat in RANGE_PATTERNS:
        for m in re.finditer(pat, t):
            gd = m.groupdict()
            try:
                if "y1" in gd and gd.get("y1") and gd.get("y2"):
                    y1, m1, d1 = gd["y1"], gd["m1"], gd["d1"]
                    y2, m2, d2 = gd["y2"], gd["m2"], gd["d2"]
                else:
                    # 年省略 → 同一年として扱う（年跨ぎは詳細ページで拾うのが確実）
                    y1 = y2 = str(base_year)
                    m1, d1 = gd["m1"], gd["d1"]
                    m2, d2 = gd["m2"], gd["d2"]

                a = _to_date(y1, m1, d1)
                b = _to_date(y2, m2, d2)
                if a <= b:
                    ranges.append((a, b))
            except Exception:
                pass

    # 単日表記
    singles = []
    for pat in SINGLE_PATTERNS:
        for m in re.finditer(pat, t):
            gd = m.groupdict()
            try:
                if gd.get("y"):
                    d = _to_date(gd["y"], gd["m"], gd["d"])
                else:
                    d = _to_date(base_year, gd["m"], gd["d"])
                singles.append(d)
            except Exception:
                pass

    # 既存レンジに含まれていなければ単日→レンジ化
    for d in singles:
        if not any(a <= d <= b for a, b in ranges):
            ranges.append((d, d))

    return ranges

@lru_cache(maxsize=64)
def _fetch_html(url: str) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.ok:
            return r.text
    except Exception:
        pass
    return ""

def _scan_event_pages_jp(target_date: datetime.date):
    """上記日本語ページを走査し、target_date を含むイベント候補を返す"""
    hits = []
    for url, site in EVENT_SOURCES_JP:
        html = _fetch_html(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")

        # 大まかに見出し＋本文の塊を走査（サイト構造に依らず拾える汎用パターン）
        for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div", "a", "span"]):
            text = " ".join(node.get_text(" ", strip=True).split())
            if not text or len(text) < 6:
                continue
            ranges = _extract_date_ranges_jp(text, base_year=target_date.year)
            for a, b in ranges:
                if a <= target_date <= b:
                    title = text
                    # タイトル（抜粋）が長すぎる場合は適度に丸める
                    if len(title) > 120:
                        title = title[:117] + "..."
                    hits.append({
                        "会場": site,
                        "開始日": a.isoformat(),
                        "終了日": b.isoformat(),
                        "イベント（抜粋）": title,
                        "リンク": url,
                    })
                    break  # そのノードからは1件だけ拾う

    # 重複除去（会場×期間×抜粋でユニーク化）
    uniq, seen = [], set()
    for h in hits:
        key = (h["会場"], h["開始日"], h["終了日"], h["イベント（抜粋）"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq

# 追加インポート（世界の祝日）
try:
    import holidays as pyholidays
except Exception:
    pyholidays = None

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
    obon_days = [datetime.date(year, 8, d) for d in range(13, 17)]
    all_days = [datetime.date(year, 1, 1) + datetime.timedelta(days=i) for i in range(370)]

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
    summer = [datetime.date(year, 7, d) for d in range(20, 32)] + [datetime.date(year, 8, d) for d in range(1, 32)]
    winter = [datetime.date(year, 12, d) for d in range(25, 32)] + [datetime.date(year + 1, 1, d) for d in range(1, 8)]
    spring = [datetime.date(year, 3, d) for d in range(20, 32)] + [datetime.date(year, 4, d) for d in range(1, 6)]
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
    start = datetime.date(y, 1, 1) - datetime.timedelta(days=7)
    days = [start + datetime.timedelta(days=i) for i in range(370 + 14)]

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
    "ACAIB アサイーボウル", # APK置きたい暫定でおいてる
    "【BF限定】GKB 黒ゴマきなこのバナナミルク", 
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

        status = " / ".join(hits + long_hits) if (hits or long_hits) else "該当なし"
        rows.append({"日付": d.strftime("%Y-%m-%d"), "該当国の祝日・長期連休": status})

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

# ---- ビッグサイト/ダイバーシティ/お台場 イベント候補（日本語のみ） ----
if selected_dates:
    st.write("### 🎪 ビッグサイト／ダイバーシティ東京プラザ／お台場：イベント開催プレビュー（日本語）")

    event_rows = []
    for d in selected_dates:
        found = _scan_event_pages_jp(d)
        found = _filter_big_events(found)
        if found:
            for ev in found:
                event_rows.append({
                    "日付": d.strftime("%Y-%m-%d"),
                    "会場": ev["会場"],
                    "開始日": ev["開始日"],
                    "終了日": ev["終了日"],
                    "イベント（抜粋）": ev["イベント（抜粋）"],
                    "リンク": ev["リンク"],
                })
        else:
            event_rows.append({
                "日付": d.strftime("%Y-%m-%d"),
                "会場": "-",
                "開始日": "",
                "終了日": "",
                "イベント（抜粋）": "該当なし",
                "リンク": "",
            })

    df_events = pd.DataFrame(event_rows)
    try:
        st.dataframe(
            df_events,
            use_container_width=True,
            column_config={"リンク": st.column_config.LinkColumn("リンク")}
        )
    except Exception:
        # 古いStreamlitなどで LinkColumn が無い場合のフォールバック
        st.dataframe(df_events, use_container_width=True)

    st.caption("※ 公式サイトの一覧/カレンダーから日付表記を抽出しています。表記ゆれにより取りこぼす場合があります。")

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
