import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="SPK Education | SAW",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==== CSS ====
st.markdown(
    """
<style>

header[data-testid="stHeader"]{
    display:none;
}


#MainMenu{
visibility:hidden;
}


footer{
visibility:hidden;
}


.stApp {
    background: #AF8260;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 1.5rem;
}

[data-testid="stSidebar"] {
    background: #9A8678;
    border-right: 0;
}

[data-testid="stSidebar"] * {
color: #111827;
}

.brand-box {
    color: #000000;
    padding: 0.25rem 0.25rem 1.3rem 0.25rem;
    margin-top: 0.15rem;
}

.brand-title {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
    color: #000000 !important;
}

.brand-subtitle {
    font-size: 0.95rem;
    opacity: 0.95;
    color: #000000 !important;
}
        
.metric-card {
    background: #E1DCC9;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(17,24,39,0.06);
    border: 1px solid #eceef5;
    padding: 1rem 1rem 0.9rem 1rem;
    height: 100%;
}

.metric-label {
    font-size: 0.92rem;
    color: #6b7280;
    margin-bottom: 0.1rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.05;
    margin: 0.1rem 0 0.2rem 0;
}
        
.metric-foot {
    font-size: 0.9rem;
    color: #6b7280;
}
        
.badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
}


.small-muted { color: #111844; font-size: 1 rem; }
.tab-title { font-weight: 900; color: #4f46e5; }
.stMetric label { color: #6b7280 !important; }
.stDataFrame { background: white; }
        
div[data-testid="stTabs"] button {
    font-weight: 700;
    }

div[data-testid="stMetric"] label {
    color:#111827 !important;
    font-weight:600 !important;
}

div[data-testid="stMetricValue"] {
    color:#111827 !important;
    font-weight:800 !important;
}

div[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
}

.best-course-box{
    display:flex;
    align-items:center;
    gap:18px;

    background:linear-gradient(
        135deg,
        #FFF8E7,
        #FCEFC7
    );

    padding:22px;

    border-radius:20px;

    border:1px solid #F4D27A;

    box-shadow:0 4px 14px rgba(0,0,0,0.05);

    margin-top:15px;
    margin-bottom:30px;
}

.best-course-icon{
    font-size:42px;
}

.best-course-title{
    font-size:18px;
    font-weight:800;
    color:#7C4A03;

    margin-bottom:4px;
}

.best-course-text{
    font-size:16px;
    color:#111827;

    line-height:1.5;
}

.best-course-score{
    margin-top:8px;

    display:inline-block;

    background:#F59E0B;

    color:white;

    padding:6px 12px;

    border-radius:999px;

    font-size:14px;

    font-weight:700;
}

hr {
    border-color: #eceef5;
        }
    </style>
    
    """,
    unsafe_allow_html=True,
)

# ==== Helpers ====
@st.cache_data
def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Remove duplicate data
    df = df.drop_duplicates()

    # Convert numeric columns
    numeric_cols = [
        "Duration (hours)",
        "Enrolled_Students",
        "Completion_Rate (%)",
        "Price ($)",
        "Rating (out of 5)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    return df

def normalize_benefit(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def normalize_cost(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / (s.max() - s.min())

def make_rank_label(rank: int) -> tuple[str, str]:
    if rank == 1:
        return "Terbaik", "badge-best"
    if rank == 2:
        return "Baik", "badge-good"
    if rank == 3:
        return "Cukup", "badge-fair"
    return "Terakhir", "badge-last"

def fmt_num(x, digits=4):
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)

def metric_card(label, value, foot, icon=None):
    icon_html = f"<div style='font-size:1.8rem;margin-bottom:0.2rem'>{icon}</div>" if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def current_timestamp():
    return datetime.now().strftime("%d %b %Y, %H:%M")

def aggregate_platforms(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("Platform", as_index=False)
        .agg(
            Total_Course=("Course_ID", "count"),
            Avg_Rating=("Rating (out of 5)", "mean"),
            Avg_Completion=("Completion_Rate (%)", "mean"),
            Avg_Enrolled=("Enrolled_Students", "mean"),
            Avg_Price=("Price ($)", "mean"),
            Avg_Duration=("Duration (hours)", "mean"),
        )
    )
    return agg

def compute_saw(platform_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = platform_df.copy()

    # Normalisasi
    out["n_rating"] = normalize_benefit(out["Avg_Rating"])
    out["n_completion"] = normalize_benefit(out["Avg_Completion"])
    out["n_enrolled"] = normalize_benefit(out["Avg_Enrolled"])
    out["n_price"] = normalize_cost(out["Avg_Price"])
    out["n_duration"] = normalize_cost(out["Avg_Duration"])

    # Bobot dinormalisasi ke 1.0
    raw = np.array(list(weights.values()), dtype=float)
    raw_sum = raw.sum() if raw.sum() != 0 else 1.0
    w = {k: v / raw_sum for k, v in weights.items()}

    out["Skor_SAW"] = (
        out["n_rating"] * w["Rating"]
        + out["n_completion"] * w["Completion"]
        + out["n_enrolled"] * w["Enrolled"]
        + out["n_price"] * w["Price"]
        + out["n_duration"] * w["Duration"]
    )

    out = out.sort_values("Skor_SAW", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)

    labels = [make_rank_label(r) for r in out["Rank"]]
    out["Keterangan"] = [x[0] for x in labels]
    out["Badge_Class"] = [x[1] for x in labels]
    return out

# ==== Load ====
DATA_PATH = "online_courses_uses.csv"
df = load_data(DATA_PATH)

# ==== Sidebar ====
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">SPK Education</div>
            <div class="brand-subtitle">Metode SAW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Pengaturan Filter")
    categories = ["Semua"] + sorted(df["Category"].dropna().unique().tolist())
    selected_category = st.selectbox("Kategori kursus", categories, index=0)

    st.caption("Data akan dihitung ulang berdasarkan filter kategori yang dipilih.")

    st.subheader("Pengaturan Bobot Kriteria")
    st.caption("Bobot otomatis dinormalisasi saat perhitungan.")

    w_rating = st.slider("Rating (Benefit)", 0, 100, 35)
    w_completion = st.slider("Completion Rate (Benefit)", 0, 100, 25)
    w_enrolled = st.slider("Enrolled Students (Benefit)", 0, 100, 20)
    w_price = st.slider("Price (Cost)", 0, 100, 15)
    w_duration = st.slider("Duration (Cost)", 0, 100, 5)

    total_bobot = w_rating + w_completion + w_enrolled + w_price + w_duration
    st.info(f"Total bobot input: {total_bobot}%")

# ===== VALIDASI TOTAL BOBOT =====
    if total_bobot > 100:
        st.warning(
        "⚠️ Total bobot melebihi 100%. "
        "Bobot tetap akan dinormalisasi otomatis."
        )

    elif total_bobot < 100:
        st.info(
        "ℹ️ Total bobot kurang dari 100%. "
        "Bobot akan dinormalisasi otomatis."
        )

    else:
        st.success(
        "✅ Total bobot sudah ideal (100%)."
        )

    st.markdown("---")
    st.markdown(
        """
        <div class="small-muted">
            <b>Keterangan kriteria</b><br>
            • Benefit: makin besar makin baik<br>
            • Cost: makin kecil makin baik
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(f"Last updated: {current_timestamp()}")

# ==== Filtering ====
filtered_df = df.copy()
if selected_category != "Semua":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category].copy()

# ===== VALIDASI DATA =====
# Validasi dataset kosong
if filtered_df.empty:
    st.error(
        "❌ Data tidak ditemukan pada kategori yang dipilih."
    )
    st.stop()

# Validasi missing value
missing_value = filtered_df.isnull().sum().sum()

if missing_value > 0:
    st.warning(
        f"⚠️ Terdapat {missing_value} missing value pada dataset."
    )

# Validasi duplicate data
duplicate_data = filtered_df.duplicated().sum()

if duplicate_data > 0:
    st.info(
        f"ℹ️ Terdapat {duplicate_data} data duplikat."
    )

# Validasi nilai negatif
numeric_cols = [
    "Duration (hours)",
    "Enrolled_Students",
    "Completion_Rate (%)",
    "Price ($)",
    "Rating (out of 5)"
]

negative_found = False

for col in numeric_cols:

    if (filtered_df[col] < 0).any():

        st.warning(
            f"⚠️ Ditemukan nilai negatif pada kolom {col}"
        )

        negative_found = True

# Validasi total bobot
if total_bobot == 0:

    st.error(
        "❌ Total bobot tidak boleh 0."
    )

    st.stop()

# Validasi jumlah platform
if filtered_df["Platform"].nunique() < 2:

    st.warning(
        "⚠️ Jumlah platform terlalu sedikit untuk perbandingan SAW."
    )

# Validasi completion rate > 100
if (filtered_df["Completion_Rate (%)"] > 100).any():

    st.warning(
        "⚠️ Terdapat Completion Rate lebih dari 100%."
    )

# Validasi rating > 5
if (filtered_df["Rating (out of 5)"] > 5).any():

    st.warning(
        "⚠️ Terdapat rating melebihi skala maksimum."
    )

# Validasi harga 0
if (filtered_df["Price ($)"] == 0).any():

    st.info(
        "ℹ️ Terdapat course gratis dengan harga 0."
    )

platform_df = aggregate_platforms(filtered_df)
weights = {
    "Rating": w_rating,
    "Completion": w_completion,
    "Enrolled": w_enrolled,
    "Price": w_price,
    "Duration": w_duration,
}
ranking_df = compute_saw(platform_df, weights)

best_platform = ranking_df.iloc[0]["Platform"] if len(ranking_df) else "-"
best_score = ranking_df.iloc[0]["Skor_SAW"] if len(ranking_df) else 0

# ==== Header ====
st.title("Sistem Pendukung Keputusan 🎓")
st.subheader("Pemilihan Platform Education Terbaik Menggunakan Metode SAW")

top_left, top_mid1, top_mid2, top_mid3, top_right = st.columns([1.15, 1.15, 1.15, 1.15, 1.15], gap="medium")
with top_left:
    metric_card("Total Platform", f"{platform_df['Platform'].nunique()}", "Alternatif", icon="📚")
with top_mid1:
    metric_card("Total Course", f"{len(filtered_df):,}".replace(",", "."), "Data", icon="🧾")
with top_mid2:
    metric_card("Platform Terbaik", best_platform, "Berdasarkan SAW", icon="🏆")
with top_mid3:
    metric_card("Skor Tertinggi", fmt_num(best_score, 4), "Nilai SAW", icon="📈")
with top_right:
    metric_card("Terakhir Diperbarui", datetime.now().strftime("%d %b %Y"), datetime.now().strftime("%H:%M WIB"), icon="⏱️")

st.write("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Ringkasan", "Dataset", "Normalisasi", "Perhitungan SAW", "Ranking by Platform","Ranking by Course", "Visualisasi"]
)

# ==== Ringkasan ====
with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🏅 Hasil Ranking SAW")
        show_rank = ranking_df[["Rank", "Platform", "Skor_SAW", "Keterangan"]].copy()
        show_rank["Skor_SAW"] = show_rank["Skor_SAW"].map(lambda x: f"{x:.4f}")
        st.dataframe(show_rank, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Visualisasi Ranking")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(ranking_df["Platform"], ranking_df["Skor_SAW"])
        ax.set_ylabel("Skor SAW")
        ax.set_xlabel("")
        ax.set_ylim(0, min(1.05, max(ranking_df["Skor_SAW"]) + 0.15))
        ax.grid(axis="y", alpha=0.25)

        for bar, score in zip(bars, ranking_df["Skor_SAW"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.4f}",
                    ha="center", va="bottom", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==== Dataset =====
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Dataset Awal")
    st.caption(f"Menampilkan 15 baris pertama dari data yang sudah difilter. Total baris: {len(filtered_df):,}".replace(",", "."))
    st.dataframe(filtered_df.head(15), use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kategori unik", filtered_df["Category"].nunique())
    c2.metric("Platform unik", filtered_df["Platform"].nunique())
    c3.metric("Rata-rata rating", f"{filtered_df['Rating (out of 5)'].mean():.2f}")
    c4.metric("Rata-rata completion", f"{filtered_df['Completion_Rate (%)'].mean():.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# ==== Normalisasi ====
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Normalisasi Data Alternatif")
    norm_df = platform_df[["Platform", "Avg_Rating", "Avg_Completion", "Avg_Enrolled", "Avg_Price", "Avg_Duration"]].copy()
    norm_df["n_rating"] = normalize_benefit(platform_df["Avg_Rating"])
    norm_df["n_completion"] = normalize_benefit(platform_df["Avg_Completion"])
    norm_df["n_enrolled"] = normalize_benefit(platform_df["Avg_Enrolled"])
    norm_df["n_price"] = normalize_cost(platform_df["Avg_Price"])
    norm_df["n_duration"] = normalize_cost(platform_df["Avg_Duration"])

    display_norm = norm_df[[
        "Platform", "n_rating", "n_completion", "n_enrolled", "n_price", "n_duration"
    ]].copy()
    for col in display_norm.columns[1:]:
        display_norm[col] = display_norm[col].map(lambda x: f"{x:.4f}")

    st.dataframe(display_norm, use_container_width=True, hide_index=True)
    st.caption("Nilai normalisasi memakai metode min-max. Benefit = makin besar makin baik, cost = makin kecil makin baik.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==== Perhitungan SAW ====
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Perhitungan SAW")
    weight_df = pd.DataFrame(
        {
            "Kriteria": ["Rating", "Completion", "Enrolled", "Price", "Duration"],
            "Bobot Input": [w_rating, w_completion, w_enrolled, w_price, w_duration],
            "Bobot Normalisasi": [
                w_rating / total_bobot if total_bobot else 0,
                w_completion / total_bobot if total_bobot else 0,
                w_enrolled / total_bobot if total_bobot else 0,
                w_price / total_bobot if total_bobot else 0,
                w_duration / total_bobot if total_bobot else 0,
            ],
            "Tipe": ["Benefit", "Benefit", "Benefit", "Cost", "Cost"],
        }
    )
    weight_df["Bobot Normalisasi"] = weight_df["Bobot Normalisasi"].map(lambda x: f"{x:.4f}")
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    calc_df = ranking_df[[
        "Platform", "Avg_Rating", "Avg_Completion", "Avg_Enrolled", "Avg_Price", "Avg_Duration",
        "n_rating", "n_completion", "n_enrolled", "n_price", "n_duration", "Skor_SAW"
    ]].copy()

    rename_map = {
        "Avg_Rating": "Rating",
        "Avg_Completion": "Completion",
        "Avg_Enrolled": "Enrolled",
        "Avg_Price": "Price",
        "Avg_Duration": "Duration",
        "n_rating": "N Rating",
        "n_completion": "N Completion",
        "n_enrolled": "N Enrolled",
        "n_price": "N Price",
        "n_duration": "N Duration",
        "Skor_SAW": "Skor SAW",
    }
    calc_df = calc_df.rename(columns=rename_map)
    for col in calc_df.columns[1:]:
        calc_df[col] = calc_df[col].map(lambda x: f"{x:.4f}")
    st.dataframe(calc_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==== Ranking =====
with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Ranking Akhir")
    rank_view = ranking_df[["Rank", "Platform", "Skor_SAW", "Keterangan"]].copy()
    rank_view["Skor_SAW"] = rank_view["Skor_SAW"].map(lambda x: f"{x:.4f}")
    st.dataframe(rank_view, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==== Visualisasi ====
with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    # ===== TITLE =====
    if selected_category == "Semua":
        st.markdown(
            "### 📚 Rank by Course (Semua Kategori)"
        )
    else:
        st.markdown(
            f"### 📚 Rank by Course - {selected_category}"
        )

    # ==== COPY DATA ====
    course_df = filtered_df.copy()

    # ==== NORMALISASI ====
    course_df["n_rating"] = normalize_benefit(
        course_df["Rating (out of 5)"]
    )

    course_df["n_completion"] = normalize_benefit(
        course_df["Completion_Rate (%)"]
    )

    course_df["n_enrolled"] = normalize_benefit(
        course_df["Enrolled_Students"]
    )

    course_df["n_price"] = normalize_cost(
        course_df["Price ($)"]
    )

    course_df["n_duration"] = normalize_cost(
        course_df["Duration (hours)"]
    )

    # ==== NORMALISASI BOBOT ====
    total = total_bobot if total_bobot != 0 else 1

    wr = w_rating / total
    wc = w_completion / total
    we = w_enrolled / total
    wp = w_price / total
    wd = w_duration / total

    # ==== HITUNG SKOR SAW ====
    course_df["Skor_SAW"] = (
        course_df["n_rating"] * wr +
        course_df["n_completion"] * wc +
        course_df["n_enrolled"] * we +
        course_df["n_price"] * wp +
        course_df["n_duration"] * wd
    )

    # ==== SORTING ====
    course_df = course_df.sort_values(
        by="Skor_SAW",
        ascending=False
    ).reset_index(drop=True)

    # ==== RANK ====
    course_df["Rank"] = np.arange(
        1,
        len(course_df) + 1
    )

    # ==== LABEL ====
    labels = [
        make_rank_label(r)
        for r in course_df["Rank"]
    ]

    course_df["Keterangan"] = [
        x[0] for x in labels
    ]

    # ==== TABEL ====
    show_course = course_df[
        [
            "Rank",
            "Course_Name",
            "Platform",
            "Category",
            "Skor_SAW",
            "Keterangan"
        ]
    ].copy()

    show_course["Skor_SAW"] = show_course[
        "Skor_SAW"
    ].map(lambda x: f"{x:.4f}")

    st.dataframe(
        show_course,
        use_container_width=True,
        hide_index=True
    )

    # ==== CAPTION ====
    if selected_category == "Semua":
        st.caption(
            "Menampilkan seluruh ranking course berdasarkan metode SAW."
        )
    else:
        st.caption(
            f"Menampilkan seluruh ranking course pada kategori {selected_category} berdasarkan metode SAW."
        )

    st.write("")

    

    # ==== SUMMARY ====
    best_course = course_df.iloc[0]
    st.markdown(
    f"""
        <div class="best-course-box">
        <div class="best-course-icon">
            🏆
        </div>
        <div>
            <div class="best-course-title">
                Course Terbaik
            </div>
            <div class="best-course-text">
                <b>{best_course['Course_Name']}</b>
                dari platform
                <b>{best_course['Platform']}</b>
            </div>
            <div class="best-course-score">
                Skor SAW:
                {best_course['Skor_SAW']:.4f}
            </div>
        </div>
    </div>
""",
unsafe_allow_html=True
    )

# ==== Visualisasi ====
with tab7:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Grafik Skor SAW")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(ranking_df["Platform"], ranking_df["Skor_SAW"])
        ax.set_ylabel("Skor SAW")
        ax.set_ylim(0, min(1.05, max(ranking_df["Skor_SAW"]) + 0.15))
        ax.grid(axis="y", alpha=0.25)
        for bar, score in zip(bars, ranking_df["Skor_SAW"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.4f}",
                    ha="center", va="bottom", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🥧 Bobot Kriteria")
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
        sizes = [w_rating, w_completion, w_enrolled, w_price, w_duration]
        labels = ["Rating", "Completion", "Enrolled", "Price", "Duration"]
        ax2.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <p style="
        color:#111827;
        font-weight:500;
        font-size:14px;
    ">
        SPK Education © 2025 | 
        Metode SAW (Simple Additive Weighting) 
        by Erlan & Roi
    </p>
    """,
    unsafe_allow_html=True
)
