# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

# --- Sentiment (VADER) ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Pastikan lexicon terunduh
@st.cache_resource
def load_vader():
    nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

# -----------------------------
# Fungsi: Ambil data dari Play Store
# -----------------------------
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
            lang="en",          # bisa diganti "id" kalau ingin Indonesia
            country="us",       # atau "id"
            sort=Sort.NEWEST,
            count=min(2000, n_reviews - len(all_reviews)),
            continuation_token=token
        )
        all_reviews.extend(batch)
        if token is None:
            break

    df = pd.DataFrame(all_reviews)

    # Adaptasi ke format yang sama dengan file contoh
    # (cek key dari google_play_scraper: reviewId, content, score, at, device)
    expected_cols = ["reviewId", "content", "score", "at", "device"]
    for col in expected_cols:
        if col not in df.columns:
            st.warning(f"Kolom '{col}' tidak ditemukan di hasil scrap. Cek struktur data terbaru google_play_scraper.")
    df = df[expected_cols]

    df = df.rename(columns={
        "reviewId": "review_id",
        "content": "review_text",
        "score": "rating",
        "at": "review_date",
        "device": "device_mentioned"
    })

    # Konversi tipe
    df["review_date"] = pd.to_datetime(df["review_date"]).dt.date
    return df


# -----------------------------
# Fungsi: Load sample data bawaan
# -----------------------------
@st.cache_data
def load_sample_data():
    # Pastikan file ini ada di folder yang sama dengan app.py
    # Gunakan file yang kamu lampirkan: strava_playstore_dummy_reviews_100rows.csv
    df = pd.read_csv("strava_playstore_dummy_reviews_100rows.csv")
    df["review_date"] = pd.to_datetime(df["review_date"]).dt.date
    return df


# -----------------------------
# Fungsi: Hitung sentiment berbasis teks + rating
# -----------------------------
def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pastikan kolom yang dibutuhkan ada
    required_cols = ["review_text", "rating"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di DataFrame.")

    # Sentiment dari teks (VADER compound)
    df["sentiment_score"] = df["review_text"].astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    # Label sentiment dari teks
    def label_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    # Flag apakah review membahas akurasi (accuracy)
    df["is_accuracy_related"] = df["review_text"].str.contains(
        "accur", case=False, na=False
    )

    return df


# -----------------------------
# Fungsi: Agregasi aspect-based per device
# -----------------------------
def aggregate_by_device(df: pd.DataFrame) -> pd.DataFrame:
    if "device_mentioned" not in df.columns:
        df = df.copy()
        df["device_mentioned"] = "Unknown"

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

    # Biar enak dibaca
    for col in ["avg_rating", "avg_sentiment", "pct_positive", "pct_neutral", "pct_negative"]:
        agg[col] = agg[col].round(2)

    return agg


# -----------------------------
# Fungsi: Agregasi waktu
# -----------------------------
def aggregate_over_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["year_month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()

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


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(
    page_title="Strava Play Store Reviews - Aspect Based Sentiment",
    layout="wide"
)

st.title("📊 Aspect-Based Sentiment Analysis for Strava Reviews (Google Play)")

st.markdown(
    """
Aplikasi ini menganalisis **review Strava** dari Google Play Store dengan fokus pada **aspek akurasi** terhadap 
**device** yang digunakan (Garmin, Huawei Band, Suunto, Polar, dll).

**Fitur utama:**
- Ambil data langsung dari Google Play Store **atau** gunakan file CSV contoh / upload sendiri  
- Analisis sentimen berbasis teks & rating  
- Aspect-based sentiment untuk review yang membahas **accuracy**  
- Ringkasan per device + grafik (bar chart & time series)
"""
)

# -----------------------------
# Sidebar: pilihan sumber data
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
        help="Untuk produksi bisa dinaikkan sampai ~100k, tapi perhatikan limit & waktu."
    )

    if st.sidebar.button("🚀 Ambil data dari Play Store"):
        with st.spinner("Mengambil data dari Google Play..."):
            df = fetch_reviews_from_playstore(n_reviews)
            st.success(f"Berhasil mengambil {len(df)} review dari Play Store.")

elif data_source == "Upload CSV sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV dengan kolom: review_id, review_text, rating, review_date, device_mentioned",
        type=["csv"],
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"]).dt.date

# Jika belum ada df
if df is None:
    st.warning("Silakan pilih sumber data di sidebar untuk memulai.")
    st.stop()

# -----------------------------
# Filter dan praproses
# -----------------------------
st.subheader("👀 Data Mentah")

with st.expander("Lihat sample data"):
    st.dataframe(df.head(20))

# Pastikan kolom ada
required_cols = ["review_id", "review_text", "rating", "review_date"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib berikut tidak ditemukan di data: {missing}")
    st.stop()

# Konversi tipe
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])

# Range tanggal
df["review_date"] = pd.to_datetime(df["review_date"]).dt.date
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

if isinstance(date_range, tuple) or isinstance(date_range, list):
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

# Filter hanya review soal "accuracy"?
st.sidebar.markdown("---")
only_accuracy = st.sidebar.checkbox(
    "Hanya review yang menyebut **accuracy/akurasi**",
    value=True,
    help="Filter review yang mengandung kata 'accur' (accuracy, accurate, dll.)"
)

# Hitung sentiment
df = compute_sentiment(df)

if only_accuracy:
    df = df[df["is_accuracy_related"]]
    st.info(f"Filter: hanya review terkait akurasi. Jumlah baris: {len(df)}")

if df.empty:
    st.warning("Tidak ada data setelah filter diterapkan.")
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
    st.metric("% Review Positif (teks)", f"{pos_pct:.1f}%")

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
st.markdown("#### 😊 Distribusi Sentiment Teks per Device")

sent_cols = ["pct_positive", "pct_neutral", "pct_negative"]
sent_long = device_agg.melt(
    id_vars="device_mentioned",
    value_vars=sent_cols,
    var_name="sentiment",
    value_name="percentage",
)

sent_long["sentiment"] = sent_long["sentiment"].map({
    "pct_positive": "Positive",
    "pct_neutral": "Neutral",
    "pct_negative": "Negative"
})

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

# Pilih device untuk grafik waktu
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
                tooltip=["year_month", "device_mentioned", "avg_rating", "n_reviews"],
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
                y=alt.Y("avg_sentiment:Q", title="Skor Sentiment (teks)"),
                color="device_mentioned:N",
                tooltip=["year_month", "device_mentioned", "avg_sentiment", "n_reviews"],
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
    "Gunakan tabel ini untuk melihat review satu per satu, "
    "beserta device, rating, dan label sentiment-nya."
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
