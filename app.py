# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

# --- Sentiment (VADER) ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# =============================
# Helper: baca CSV fleksibel
# =============================
def read_strava_csv(source):
    """
    Baca CSV dengan fallback:
    - Coba separator default (,)
    - Kalau gagal (ParserError), coba pakai ';' (umum dari Excel di Windows)
    """
    try:
        return pd.read_csv(source)
    except pd.errors.ParserError:
        # Kalau source adalah file-like object (upload), perlu reset pointer
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, sep=";", engine="python", on_bad_lines="skip")


# Pastikan lexicon terunduh
@st.cache_resource
def load_vader():
    nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


sia = load_vader()


# =============================
# Helper: Deteksi device dari teks
# =============================
def detect_device_from_text(text: str) -> str:
    """
    Deteksi brand / device dari teks review (rule-based sederhana).
    """
    if not isinstance(text, str):
        return "Unknown"

    t = text.lower()

    device_map = {
        "garmin": "Garmin",
        "huawei": "Huawei Band",
        "band 7": "Huawei Band",
        "fitbit": "Fitbit",
        "suunto": "Suunto",
        "polar": "Polar",
        "apple watch": "Apple Watch",
        "iwatch": "Apple Watch",
        "coros": "Coros",
        "xiaomi": "Xiaomi",
        "mi band": "Mi Band",
        "samsung watch": "Samsung Watch",
        "galaxy watch": "Samsung Watch",
    }

    for kw, label in device_map.items():
        if kw in t:
            return label

    return "Unknown"


# =============================
# Ambil data dari Play Store
# =============================
@st.cache_data(show_spinner=True)
def fetch_reviews_from_playstore(n_reviews: int = 1000):
    """
    Ambil review Strava dari Google Play Store.
    Butuh library: google-play-scraper

    pip install google-play-scraper
    """
    from google_play_scraper import reviews, Sort

    all_reviews = []
    token = None

    while len(all_reviews) < n_reviews:
        batch, token = reviews(
            "com.strava",
            lang="en",  # bisa diganti "id" kalau ingin Indonesia
            country="us",  # atau "id"
            sort=Sort.NEWEST,
            count=min(2000, n_reviews - len(all_reviews)),
            continuation_token=token,
        )
        all_reviews.extend(batch)
        if token is None:
            break

    df = pd.DataFrame(all_reviews)

    base_cols = ["reviewId", "content", "score", "at"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan di hasil scrap Play Store: {missing}")
        return pd.DataFrame(
            columns=["review_id", "review_text", "rating", "review_date", "device_mentioned"]
        )

    df = df[base_cols].copy()

    df = df.rename(
        columns={
            "reviewId": "review_id",
            "content": "review_text",
            "score": "rating",
            "at": "review_date",
        }
    )

    df["review_date"] = pd.to_datetime(df["review_date"]).dt.date

    # Deteksi device dari teks (karena Play Store tidak menyediakan kolom device)
    df["device_mentioned"] = df["review_text"].apply(detect_device_from_text)

    st.info(
        "Kolom `device_mentioned` untuk data Play Store dibuat dari teks review dengan pencarian keyword brand device."
    )
    return df


# =============================
# Load sample data dari CSV
# =============================
@st.cache_data
def load_sample_data():
    # File contoh: strava_playstore_dummy_reviews_100rows.csv
    df = read_strava_csv("strava_playstore_dummy_reviews_100rows.csv")

    # Kalau nama kolom di file revisi beda, bisa di-rename di sini.
    # Contoh (sesuaikan kalau perlu):
    # df = df.rename(columns={
    #     "id_review": "review_id",
    #     "text_review": "review_text",
    #     "bintang": "rating",
    #     "tanggal": "review_date",
    # })

    expected = ["review_id", "review_text", "rating", "review_date"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(f"Kolom yang belum ada di sample data: {missing}")

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"]).dt.date

    return df


# =============================
# Fungsi: Hitung sentiment (multi-algoritma)
# =============================
def compute_sentiment(df: pd.DataFrame, algorithm: str = "vader") -> pd.DataFrame:
    """
    algorithm:
        - "vader"  : sentiment murni dari teks (VADER)
        - "rating" : sentiment dari rating saja (1-5)
        - "hybrid" : gabungan VADER + rating
    """
    df = df.copy()

    # rating-based score: mapping 1..5 ke [-1, 1]
    df["rating_based_score"] = df["rating"].apply(
        lambda r: (r - 3) / 2 if pd.notnull(r) else 0
    )

    # text-based score (VADER) jika perlu
    if algorithm in ("vader", "hybrid"):
        df["text_sentiment"] = df["review_text"].astype(str).apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
    else:
        df["text_sentiment"] = np.nan

    # combine
    if algorithm == "vader":
        df["sentiment_score"] = df["text_sentiment"]
    elif algorithm == "rating":
        df["sentiment_score"] = df["rating_based_score"]
    elif algorithm == "hybrid":
        df["sentiment_score"] = (
            0.5 * df["text_sentiment"].fillna(0)
            + 0.5 * df["rating_based_score"].fillna(0)
        )
    else:
        df["sentiment_score"] = df["text_sentiment"]

    # Label sentiment dari skor gabungan
    def label_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    # Flag apakah review membahas akurasi
    df["is_accuracy_related"] = df["review_text"].str.contains(
        "accur", case=False, na=False
    )

    return df


# =============================
# Agregasi per device
# =============================
def aggregate_by_device(df: pd.DataFrame) -> pd.DataFrame:
    if "device_mentioned" not in df.columns:
        df = df.copy()
        df["device_mentioned"] = df["review_text"].apply(detect_device_from_text)

    agg = (
        df.groupby("device_mentioned")
        .agg(
            n_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
            pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean() * 100),
            pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
        )
        .reset_index()
    )

    for col in [
        "avg_rating",
        "avg_sentiment",
        "pct_positive",
        "pct_neutral",
        "pct_negative",
    ]:
        agg[col] = agg[col].round(2)

    return agg


# =============================
# Agregasi waktu
# =============================
def aggregate_over_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["year_month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()

    if "device_mentioned" not in df.columns:
        df["device_mentioned"] = df["review_text"].apply(detect_device_from_text)

    time_agg = (
        df.groupby(["year_month", "device_mentioned"])
        .agg(
            n_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    time_agg["avg_rating"] = time_agg["avg_rating"].round(2)
    time_agg["avg_sentiment"] = time_agg["avg_sentiment"].round(2)
    return time_agg


# =============================
# STREAMLIT APP
# =============================
st.set_page_config(
    page_title="Strava Play Store Reviews - Aspect Based Sentiment",
    layout="wide",
)

st.title("📊 Aspect-Based Sentiment Analysis for Strava Reviews (Google Play)")

st.markdown(
    """
Aplikasi ini menganalisis **review Strava** dari Google Play Store dengan fokus pada **aspek akurasi** terhadap 
**device** (Garmin, Huawei Band, Suunto, dll).

**Fitur utama:**
- Ambil data langsung dari Google Play Store **atau** gunakan file CSV contoh / upload sendiri  
- Analisis sentimen dengan beberapa **algoritma yang bisa dipilih**  
- Aspect-based sentiment untuk review yang membahas **accuracy**  
- Ringkasan per device + grafik (bar chart & time series)
"""
)

# -----------------------------
# Sidebar: sumber data
# -----------------------------
st.sidebar.header("⚙️ Pengaturan Data")

data_source = st.sidebar.radio(
    "Sumber data",
    ("Contoh CSV bawaan", "Ambil dari Play Store", "Upload CSV sendiri"),
)

df = None

if data_source == "Contoh CSV bawaan":
    st.sidebar.info("Menggunakan file: strava_playstore_dummy_reviews_100rows.csv")
    df = load_sample_data()

elif data_source == "Ambil dari Play Store":
    n_reviews = st.sidebar.slider(
        "Jumlah review yang diambil",
        min_value=100,
        max_value=5000,
        step=100,
        value=1000,
        help="Untuk produksi bisa dinaikkan sampai ~100k, tapi perhatikan limit & waktu.",
    )

    if st.sidebar.button("🚀 Ambil data dari Play Store"):
        with st.spinner("Mengambil data dari Google Play..."):
            df = fetch_reviews_from_playstore(n_reviews)
            if not df.empty:
                st.success(f"Berhasil mengambil {len(df)} review dari Play Store.")

elif data_source == "Upload CSV sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV dengan kolom minimal: review_id, review_text, rating, review_date",
        type=["csv"],
    )
    if uploaded_file is not None:
        df = read_strava_csv(uploaded_file)
        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"]).dt.date

# Jika belum ada df
if df is None or df.empty:
    st.warning("Silakan pilih sumber data di sidebar untuk memulai.")
    st.stop()

# =============================
# Pastikan kolom wajib
# =============================
required_cols = ["review_id", "review_text", "rating", "review_date"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib berikut tidak ditemukan di data: {missing}")
    st.stop()

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])

df["review_date"] = pd.to_datetime(df["review_date"]).dt.date

# Kalau device_mentioned belum ada (misal dari Excel revisi), buat dari teks
if "device_mentioned" not in df.columns:
    df["device_mentioned"] = df["review_text"].apply(detect_device_from_text)

# -----------------------------
# Filter tanggal & rating
# -----------------------------
st.subheader("👀 Data Mentah")

with st.expander("Lihat sample data"):
    st.dataframe(df.head(20))

min_date = df["review_date"].min()
max_date = df["review_date"].max()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Filter Tanggal")

date_range = st.sidebar.date_input(
    "Rentang tanggal review",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask_date = (df["review_date"] >= start_date) & (df["review_date"] <= end_date)
df = df[mask_date]

st.sidebar.subheader("⭐ Filter Rating")
min_rating, max_rating = st.sidebar.slider(
    "Rentang rating",
    min_value=1,
    max_value=5,
    value=(1, 5),
)
df = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]

st.sidebar.markdown("---")

# -----------------------------
# Pilih algoritma sentiment
# -----------------------------
st.sidebar.subheader("🧠 Pilih Algoritma Sentiment")

algo_label = st.sidebar.selectbox(
    "Algoritma yang digunakan",
    (
        "VADER (berbasis teks)",
        "Rating saja",
        "Hybrid (VADER + Rating)",
    ),
)

algo_map = {
    "VADER (berbasis teks)": "vader",
    "Rating saja": "rating",
    "Hybrid (VADER + Rating)": "hybrid",
}
algo_key = algo_map[algo_label]

# Filter hanya review soal accuracy?
only_accuracy = st.sidebar.checkbox(
    "Hanya review yang menyebut **accuracy/akurasi**",
    value=True,
    help="Filter review yang mengandung kata 'accur' (accuracy, accurate, dll.)",
)

# -----------------------------
# Hitung sentiment
# -----------------------------
df = compute_sentiment(df, algorithm=algo_key)

if only_accuracy:
    df = df[df["is_accuracy_related"]]
    st.info(f"Filter: hanya review terkait akurasi. Jumlah baris: {len(df)}")

if df.empty:
    st.warning("Tidak ada data setelah filter dan algoritma diterapkan.")
    st.stop()

# -----------------------------
# Ringkasan umum
# -----------------------------
st.subheader("📌 Ringkasan Data Sesudah Filter")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jumlah Review", f"{len(df):,}")
with col2:
    st.metric("Rata-rata Rating", f"{df['rating'].mean():.2f}")
with col3:
    st.metric("Skor Sentiment Rata-rata", f"{df['sentiment_score'].mean():.2f}")
with col4:
    pos_pct = (df["sentiment_label"] == "positive").mean() * 100
    st.metric("% Review Positif", f"{pos_pct:.1f}%")

st.caption(f"Algoritma sentiment yang digunakan: **{algo_label}**")

# -----------------------------
# Aspect-based: per device
# -----------------------------
st.subheader("🎯 Aspect-Based Sentiment per Device (Akurasi)")

device_agg = aggregate_by_device(df)

st.markdown("**Tabel ringkasan per device:**")
st.dataframe(device_agg, use_container_width=True)

# Grafik rata-rata rating per device
st.markdown("#### 📈 Rata-rata Rating per Device")

chart_rating = (
    alt.Chart(device_agg)
    .mark_bar()
    .encode(
        x=alt.X("device_mentioned:N", title="Device"),
        y=alt.Y("avg_rating:Q", title="Rata-rata Rating"),
        tooltip=[
            alt.Tooltip("device_mentioned", title="Device"),
            alt.Tooltip("n_reviews", title="Jumlah Review"),
            alt.Tooltip("avg_rating", title="Rata-rata Rating"),
        ],
    )
    .properties(height=400)
)
st.altair_chart(chart_rating, use_container_width=True)

# Grafik distribusi sentiment per device (stacked bar)
st.markdown("#### 😊 Distribusi Sentiment per Device")

sent_cols = ["pct_positive", "pct_neutral", "pct_negative"]
sent_long = device_agg.melt(
    id_vars="device_mentioned",
    value_vars=sent_cols,
    var_name="sentiment",
    value_name="percentage",
)

sent_long["sentiment"] = sent_long["sentiment"].map(
    {
        "pct_positive": "Positive",
        "pct_neutral": "Neutral",
        "pct_negative": "Negative",
    }
)

chart_sent = (
    alt.Chart(sent_long)
    .mark_bar()
    .encode(
        x=alt.X("device_mentioned:N", title="Device"),
        y=alt.Y("percentage:Q", stack="normalize", title="Proporsi Review"),
        color=alt.Color("sentiment:N", title="Sentiment"),
        tooltip=["device_mentioned", "sentiment", "percentage"],
    )
    .properties(height=400)
)
st.altair_chart(chart_sent, use_container_width=True)

# -----------------------------
# Tren waktu
# -----------------------------
st.subheader("⏱️ Tren Waktu: Rating & Sentiment per Device")

time_agg = aggregate_over_time(df)

devices_available = list(time_agg["device_mentioned"].unique())
selected_devices = st.multiselect(
    "Pilih device untuk ditampilkan di grafik waktu",
    options=devices_available,
    default=devices_available[:3] if len(devices_available) > 3 else devices_available,
)

if selected_devices:
    time_filtered = time_agg[time_agg["device_mentioned"].isin(selected_devices)]

    tab1, tab2 = st.tabs(["Rata-rata Rating per Bulan", "Skor Sentiment per Bulan"])

    with tab1:
        chart_time_rating = (
            alt.Chart(time_filtered)
            .mark_line(point=True)
            .encode(
                x=alt.X("year_month:T", title="Bulan"),
                y=alt.Y("avg_rating:Q", title="Rata-rata Rating"),
                color="device_mentioned:N",
                tooltip=[
                    "year_month",
                    "device_mentioned",
                    "avg_rating",
                    "n_reviews",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_time_rating, use_container_width=True)

    with tab2:
        chart_time_sentiment = (
            alt.Chart(time_filtered)
            .mark_line(point=True)
            .encode(
                x=alt.X("year_month:T", title="Bulan"),
                y=alt.Y("avg_sentiment:Q", title="Skor Sentiment"),
                color="device_mentioned:N",
                tooltip=[
                    "year_month",
                    "device_mentioned",
                    "avg_sentiment",
                    "n_reviews",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_time_sentiment, use_container_width=True)
else:
    st.info("Pilih minimal satu device untuk melihat tren waktu.")

# -----------------------------
# Detail per review
# -----------------------------
st.subheader("🔍 Detail Review")

st.markdown(
    "Tabel ini menampilkan review satu per satu, beserta device, rating, dan label sentiment."
)

st.dataframe(
    df[
        [
            "review_id",
            "review_date",
            "device_mentioned",
            "rating",
            "sentiment_score",
            "sentiment_label",
            "review_text",
        ]
    ].sort_values("review_date", ascending=False),
    use_container_width=True,
)
