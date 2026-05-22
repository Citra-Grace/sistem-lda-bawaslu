import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
import numpy as np
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Sistem Analisis Kendala Logistik Pemilu",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS CUSTOM PROFESIONAL
# ==========================
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }

    .main-header {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.1rem;
        line-height: 1.3;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1rem;
        font-style: italic;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #777;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #c62828;
        font-weight: 700;
    }

    .info-box {
        background-color: #f5f5f5;
        border-left: 4px solid #c62828;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 0.8rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .info-box h4 {
        margin-top: 0;
        color: #c62828;
    }

    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
        color: #e0e0e0 !important;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #333 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }

    .footer-text {
        text-align: center;
        color: #999;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER DENGAN LOGO
# ==========================
col_logo_left, col_logo_title, col_logo_right = st.columns([1, 3, 1])

with col_logo_left:
    if os.path.exists("logo_bawaslu.png"):
        st.image("logo_bawaslu.png", width=110)

with col_logo_title:
    st.markdown(
        '<p class="main-header">Sistem Analisis Kendala Distribusi Logistik Pemilu</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Berbasis Topic Modeling — Latent Dirichlet Allocation (LDA)</p>',
        unsafe_allow_html=True
    )

with col_logo_right:
    if os.path.exists("logo_unima.png"):
        st.image("logo_unima.png", width=110)

st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    # Logo kecil di sidebar (rapi)
    col_sb1, col_sb2, col_sb3 = st.columns([1, 1, 1])
    with col_sb1:
        if os.path.exists("logo_bawaslu.png"):
            st.image("logo_bawaslu.png", width=65)
    with col_sb3:
        if os.path.exists("logo_unima.png"):
            st.image("logo_unima.png", width=65)

    st.markdown("### Sistem Analisis LDA")
    st.markdown("**BAWASLU RI × UNIMA**")
    st.markdown("---")

    menu = st.radio(
        "Menu Navigasi",
        ["Dashboard", "Eksplorasi Topik", "Visualisasi Interaktif", "Tentang Sistem"]
    )

    st.markdown("---")
    st.markdown("**Status Sistem**")
    st.success("🟢 Online & Stabil")
    st.caption("Versi 1.0 — 2026")

# ==========================
# LABEL & DESKRIPSI TOPIK
# ==========================
TOPIC_LABELS = [
    "Pengawasan & Penegakan Hukum (Sumsel & Banten)",
    "Distribusi Logistik Wilayah Jawa & Sumatera",
    "Kelengkapan dan Kondisi Perlengkapan Logistik",
    "Kendala Geografis Sungai & Pedalaman",
    "Distribusi Logistik Wilayah Kepulauan & Pesisir",
    "Laporan Tanpa Temuan Kendala (Nihil)",
    "Proses Pengadaan dan Produksi Logistik",
    "Distribusi Logistik Kalimantan & Kepulauan",
    "Distribusi Pasca Putusan Mahkamah Konstitusi",
    "Distribusi Logistik Kalsel & Jawa Timur"
]

TOPIC_DESCRIPTIONS = {
    0: "Laporan pengawasan distribusi logistik terkait aspek penegakan hukum dan penanganan pelanggaran di wilayah Sumatera Selatan dan Banten.",
    1: "Pola distribusi logistik berdasarkan wilayah administratif di Pulau Jawa dan Sumatera.",
    2: "Pembahasan terkait ketersediaan, kelengkapan, dan kondisi perlengkapan logistik pemilu di lapangan.",
    3: "Kendala distribusi logistik di wilayah dengan akses geografis sulit seperti daerah aliran sungai dan pedalaman.",
    4: "Distribusi logistik di wilayah kepulauan dan pesisir yang berpotensi menimbulkan hambatan pengiriman.",
    5: "Laporan pengawasan distribusi logistik yang menyatakan tidak ditemukan kendala signifikan.",
    6: "Tahap pengadaan logistik melalui mekanisme lelang dan proses produksi perlengkapan pemilu.",
    7: "Distribusi logistik berdasarkan wilayah Kalimantan dan kepulauan dengan karakteristik geografis khusus.",
    8: "Distribusi logistik dalam konteks Pemungutan Suara Ulang (PSU) setelah putusan Mahkamah Konstitusi.",
    9: "Laporan distribusi logistik di wilayah Kalimantan Selatan dan Jawa Timur."
}

# ==========================
# LOAD DATA & MODEL
# ==========================
@st.cache_resource
def load_model():
    import pickle
    
    df = pd.read_csv("dataset_laporan_bawaslu_preprocessed.csv")
    
    with open("model_data/model_results.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    return df, model_data

df, model_data = load_model()


# ============================================================
# 1. HALAMAN DASHBOARD
# ============================================================
if menu == "Dashboard":

    st.markdown("### 📊 Ringkasan Dataset & Model")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dokumen", len(df))
    col2.metric("Jumlah Topik", 10)
    col3.metric("Kata Unik", len(dictionary))
    col4.metric("Coherence Score", "0.4040")

    st.markdown("---")

    # Grafik Coherence Score
    st.markdown("### 📈 Evaluasi Coherence Score")

    coherence_data = {
        "Jumlah Topik": list(range(2, 11)),
        "Coherence Score": [
            0.3092, 0.3432, 0.3979, 0.3560,
            0.3946, 0.4006, 0.3932, 0.4037, 0.4040
        ]
    }
    df_coh = pd.DataFrame(coherence_data)

    col_chart, col_explain = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            df_coh["Jumlah Topik"], df_coh["Coherence Score"],
            marker='o', linewidth=2.5, color='#c62828',
            markersize=8, markerfacecolor='white', markeredgewidth=2
        )
        ax.fill_between(
            df_coh["Jumlah Topik"], df_coh["Coherence Score"],
            alpha=0.12, color='#c62828'
        )
        # Highlight optimal
        ax.plot(10, 0.4040, 'o', markersize=14, color='#c62828',
                markeredgecolor='white', markeredgewidth=3)
        ax.annotate('Optimal\nk=10', xy=(10, 0.4040),
                    xytext=(9, 0.41), fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#c62828'),
                    color='#c62828')
        ax.set_xlabel("Jumlah Topik (k)", fontsize=11)
        ax.set_ylabel("Coherence Score", fontsize=11)
        ax.set_title("Grafik Coherence Score (k = 2–10)",
                     fontsize=13, fontweight='bold')
        ax.set_xticks(range(2, 11))
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

    with col_explain:
        st.markdown("""
        <div class="info-box">
            <h4>📌 Interpretasi</h4>
            <p>Nilai coherence tertinggi <b>0.4040</b> dicapai pada
            <b>k = 10</b> topik.</p>
            <ul>
                <li>Topik koheren secara semantik</li>
                <li>Kata-kata dalam topik saling berkaitan</li>
                <li>Model mampu membedakan tema dengan baik</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Distribusi Dokumen per Topik
    st.markdown("### 📋 Distribusi Dokumen per Topik Dominan")

    topic_dist = []
    for doc_bow in corpus:
        topic_probs = lda_model.get_document_topics(
            doc_bow, minimum_probability=0.0
        )
        dominant = max(topic_probs, key=lambda x: x[1])[0]
        topic_dist.append(dominant)

    topic_counts = pd.Series(topic_dist).value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_colors = plt.cm.Reds(np.linspace(0.35, 0.85, 10))
    bars = ax2.bar(
        [f"T{i}" for i in topic_counts.index],
        topic_counts.values,
        color=bar_colors,
        edgecolor='white',
        linewidth=0.8
    )
    ax2.set_xlabel("Topik", fontsize=11)
    ax2.set_ylabel("Jumlah Dokumen", fontsize=11)
    ax2.set_title("Jumlah Dokumen Berdasarkan Topik Dominan",
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2., h + 0.5,
            str(int(h)), ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    st.pyplot(fig2)

    # Ringkasan Temuan
    st.markdown("---")
    st.markdown("### 🔍 Ringkasan Temuan Utama")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="info-box">
            <h4>🗺️ Dominasi Wilayah</h4>
            <p>Topik didominasi nama wilayah administratif,
            mencerminkan struktur pelaporan BAWASLU berbasis geografis.</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="info-box">
            <h4>⛰️ Kendala Geografis</h4>
            <p>Kata <i>sungai, pulau, laut, hulu</i> menunjukkan
            hambatan distribusi di wilayah terpencil & kepulauan.</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="info-box">
            <h4>⚖️ Pasca Putusan MK</h4>
            <p>Topik terkait distribusi logistik pasca putusan
            Mahkamah Konstitusi (PSU) teridentifikasi secara jelas.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 2. HALAMAN EKSPLORASI TOPIK
# ============================================================
elif menu == "Eksplorasi Topik":

    st.markdown("### 🔎 Eksplorasi Detail Topik")

    # Selector + deskripsi
    topic_number = st.selectbox(
        "Pilih Topik untuk Dianalisis",
        list(range(10)),
        format_func=lambda x: f"Topik {x} — {TOPIC_LABELS[x]}"
    )

    st.markdown(f"""
    <div class="info-box">
        <h4>Topik {topic_number}: {TOPIC_LABELS[topic_number]}</h4>
        <p>{TOPIC_DESCRIPTIONS[topic_number]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Chart + Word Cloud side by side
    col_chart, col_cloud = st.columns([1, 1])

    topic_terms = model_data["topics_data"][topic_number][:10]
    words = [t[0] for t in topic_terms]
    weights = [t[1] for t in topic_terms]

    with col_chart:
        st.markdown("#### Distribusi Kata Dominan")
        fig, ax = plt.subplots(figsize=(7, 5))
        bar_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(words)))
        bars = ax.barh(words[::-1], weights[::-1], color=bar_colors)
        ax.set_xlabel("Bobot Probabilitas", fontsize=11)
        ax.set_title(
            f"Kata Dominan — Topik {topic_number}",
            fontsize=13, fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        for bar, w in zip(bars, weights[::-1]):
            ax.text(
                w + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va='center', fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)

    with col_cloud:
        st.markdown("#### Word Cloud")
        word_freq = {w: wt for w, wt in topic_terms}
        wc = WordCloud(
            width=700, height=450,
            background_color='white',
            colormap='Reds',
            max_words=50,
            contour_width=1,
            contour_color='#c62828'
        ).generate_from_frequencies(word_freq)

        fig_wc, ax_wc = plt.subplots(figsize=(7, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        plt.tight_layout()
        st.pyplot(fig_wc)

    # Tabel Detail
    with st.expander("📋 Lihat Detail Kata & Probabilitas"):
        df_terms = pd.DataFrame(topic_terms, columns=["Kata", "Probabilitas"])
        df_terms.index = range(1, len(df_terms) + 1)
        df_terms.index.name = "No"
        df_terms["Probabilitas"] = df_terms["Probabilitas"].round(4)
        st.dataframe(df_terms, use_container_width=True)

        # Tombol download
        csv_data = df_terms.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Data Topik (CSV)",
            data=csv_data,
            file_name=f"topik_{topic_number}_data.csv",
            mime="text/csv"
        )


# ============================================================
# 3. HALAMAN VISUALISASI INTERAKTIF
# ============================================================
elif menu == "Visualisasi Interaktif":

    st.markdown("### 🌐 Visualisasi Interaktif Model LDA")

    st.markdown("""
    <div class="info-box">
        <h4>📌 Tentang Visualisasi</h4>
        <p>Visualisasi di bawah menggunakan <b>pyLDAvis</b> untuk menampilkan
        hubungan antar topik secara interaktif. Klik lingkaran topik untuk
        melihat distribusi kata, dan gunakan slider λ untuk mengatur relevansi.</p>
    </div>
    """, unsafe_allow_html=True)

    # pyLDAvis
    pyldavis_loaded = False

    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis

        with st.spinner("⏳ Memuat visualisasi interaktif pyLDAvis..."):
            vis_data = gensimvis.prepare(
                lda_model, corpus, dictionary, sort_topics=False
            )
            html_string = pyLDAvis.prepared_data_to_html(vis_data)

        st.components.v1.html(html_string, width=1300, height=800, scrolling=True)
        pyldavis_loaded = True

        st.markdown("---")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("""
            **🟢 Cara Membaca Visualisasi:**
            - **Lingkaran** = Satu topik
            - **Ukuran** = Proporsi topik dalam korpus
            - **Jarak** = Topik yang mirip berdekatan
            - **Klik** lingkaran untuk melihat kata dominan
            """)
        with col_v2:
            st.markdown("""
            **🔵 Fitur Interaktif:**
            - Hover untuk highlight kata penting
            - Klik lingkaran untuk fokus pada satu topik
            - Slider **λ** untuk atur relevansi kata
            - Bandingkan kata antar topik
            """)

    except ImportError:
        st.warning("⚠️ Package `pyLDAvis` belum terinstall.")
        st.code("pip install pyldavis", language="bash")
        st.info("Setelah install, restart Streamlit: `streamlit run app.py`")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat pyLDAvis: {e}")

    # Fallback — hanya tampil jika pyLDAvis GAGAL
    if not pyldavis_loaded:
        st.markdown("---")
        st.markdown("### 📊 Fallback: Ringkasan Semua Topik")

        cols_per_row = 2
        for row_start in range(0, 10, cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, col in enumerate(cols):
                t_idx = row_start + idx
                if t_idx >= 10:
                    break
                with col:
                    terms = lda_model.show_topic(t_idx, topn=8)
                    w = [t[0] for t in terms]
                    p = [t[1] for t in terms]

                    fig_t, ax_t = plt.subplots(figsize=(5, 3))
                    ax_t.barh(
                        w[::-1], p[::-1],
                        color=plt.cm.Reds(np.linspace(0.4, 0.85, len(w)))
                    )
                    ax_t.set_title(f"Topik {t_idx}", fontsize=10, fontweight='bold')
                    ax_t.tick_params(axis='both', labelsize=8)
                    ax_t.grid(axis='x', alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig_t)
                    st.caption(TOPIC_LABELS[t_idx])


# ============================================================
# 4. HALAMAN TENTANG SISTEM
# ============================================================
elif menu == "Tentang Sistem":

    st.markdown("### ℹ️ Tentang Sistem")

    col_about, col_tech = st.columns([2, 1])

    with col_about:
        st.markdown("""
        <div class="info-box">
            <h4>📚 Deskripsi Sistem</h4>
            <p style="text-align:justify;">
            Sistem ini dikembangkan sebagai implementasi dari penelitian skripsi berjudul
            <b>"Topic Modeling Kendala Logistik Pemilu pada Laporan Distribusi Logistik Pemilu
            di BAWASLU RI Menggunakan Latent Dirichlet Allocation (LDA)"</b>.
            Sistem membantu BAWASLU RI menganalisis pola kendala distribusi logistik Pemilu
            secara otomatis dari ratusan laporan pengawasan berbentuk teks naratif.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Tujuan Sistem
        1. Mengidentifikasi topik-topik utama kendala logistik secara otomatis
        2. Mempercepat analisis big data laporan pengawasan
        3. Mendukung pengambilan keputusan berbasis data (*data-driven*)
        4. Menyajikan visualisasi interaktif untuk eksplorasi temuan
        """)

        st.markdown("### 📊 Parameter Model")
        param_data = {
            "Parameter": ["Jumlah Topik", "Coherence Score", "Passes",
                          "Random State", "Total Dokumen", "Kata Unik"],
            "Nilai": ["10", "0.4040", "10", "42", "254", "6.152"]
        }
        st.table(pd.DataFrame(param_data))

    with col_tech:
        st.markdown("""
        <div style="background:#f8f9fa; padding:1.5rem; border-radius:10px;
                    border:1px solid #dee2e6;">
            <h4>🛠️ Teknologi</h4>
            <ul>
                <li><b>Framework:</b> Streamlit</li>
                <li><b>Model:</b> Gensim LDA</li>
                <li><b>Visualisasi:</b> Matplotlib, pyLDAvis, WordCloud</li>
                <li><b>Preprocessing:</b> NLTK, Sastrawi</li>
                <li><b>Bahasa:</b> Python 3.x</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div style="background:#f8f9fa; padding:1.5rem; border-radius:10px;
                    border:1px solid #dee2e6; margin-top:1rem;">
            <h4>👤 Pengembang</h4>
            <ul>
                <li><b>Nama:</b> Citra Grace Kalangie</li>
                <li><b>NIM:</b> 22210133</li>
                <li><b>Prodi:</b> Teknik Informatika</li>
                <li><b>Fakultas:</b> Teknik</li>
                <li><b>Universitas:</b> Universitas Negeri Manado</li>
                <li><b>Tahun:</b> 2026</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("### 🏛️ Institusi")
        col_inst1, col_inst2 = st.columns(2)
        with col_inst1:
            if os.path.exists("logo_bawaslu.png"):
                st.image("logo_bawaslu.png", caption="BAWASLU RI", width=90)
        with col_inst2:
            if os.path.exists("logo_unima.png"):
                st.image("logo_unima.png", caption="UNIMA", width=90)

        st.markdown("""
        <div class="success-box" style="margin-top:1rem;">
            <b>✅ Status Sistem</b><br>
            Stabil dan siap digunakan
        </div>
        """, unsafe_allow_html=True)

    # Panduan Penggunaan
    st.markdown("---")
    with st.expander("📖 Panduan Penggunaan Sistem", expanded=False):
        st.markdown("""
        #### Langkah 1 — Dashboard
        - Lihat ringkasan statistik dataset dan model
        - Periksa grafik coherence score
        - Lihat distribusi dokumen per topik

        #### Langkah 2 — Eksplorasi Topik
        - Pilih topik dari dropdown
        - Lihat grafik kata dominan dan probabilitasnya
        - Analisis word cloud untuk memahami tema
        - Download data topik dalam format CSV

        #### Langkah 3 — Visualisasi Interaktif
        - Gunakan pyLDAvis untuk melihat hubungan antar topik
        - Klik lingkaran untuk fokus pada satu topik
        - Gunakan slider λ untuk atur relevansi kata

        #### Catatan Penting
        - Data: Laporan BAWASLU RI periode Pemilu 2024
        - Model: 10 topik optimal berdasarkan evaluasi coherence score
        - Preprocessing: case folding, stopword removal, tokenizing
        """)


# ==========================
# FOOTER
# ==========================
st.markdown("---")

col_f1, col_f2, col_f3 = st.columns([1, 3, 1])

with col_f1:
    if os.path.exists("logo_bawaslu.png"):
        st.image("logo_bawaslu.png", width=55)

with col_f2:
    st.markdown("""
    <div class="footer-text">
        © 2026 — Sistem Analisis Kendala Distribusi Logistik Pemilu<br>
        <b>BAWASLU RI × Universitas Negeri Manado</b><br>
        Skripsi Teknik Informatika — Citra Grace Kalangie (22210133)
    </div>
    """, unsafe_allow_html=True)

with col_f3:
    if os.path.exists("logo_unima.png"):
        st.image("logo_unima.png", width=55)
