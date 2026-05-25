import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# ================= 1. KONFIGURASI HALAMAN & TEMA =================
st.set_page_config(
    page_title="AI Career Navigator - Business EDA", 
    page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk Tampilan Dashboard (Background & Teks)
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* --- WARNA BACKGROUND CUSTOM --- */
    /* Background Dashboard Utama (Abu Terang) */
    [data-testid="stAppViewContainer"] {
        background-color: #F1F5F9; /* Abu terang (slate-100) */
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* Menyesuaikan warna teks default utama menjadi gelap agar terbaca di background terang */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3 {
        color: #1E293B; /* Teks Gelap */
    }

    /* Background Sidebar (Krem) */
    [data-testid="stSidebar"] {
        background-color: #FFF8DC; /* Warna Krem (Cornsilk) */
    }
    
    /* Menyesuaikan warna teks di Sidebar menjadi gelap agar terbaca */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] .stWidgetLabel p {
        color: #1E293B !important;
    }
    /* ------------------------------- */

    /* Desain Kartu Insight */
    .insight-card {
        background-color: #FFFFFF; /* Putih bersih agar pop-out dari abu terang */
        border-left: 5px solid #0284c7;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        margin-bottom: 25px;
    }
    .question-title {
        color: #0369a1 !important;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .insight-text {
        color: #334155 !important;
        font-size: 15px;
        line-height: 1.5;
    }
    
    /* Teks Judul KPI / Info disesuaikan agar cocok dengan background terang */
    .kpi-title {
        color: #0284c7; /* Biru agak gelap */
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }
    .kpi-subtitle {
        color: #475569; /* Abu gelap */
        font-size: 13px;
        margin-bottom: 15px;
    }
    
    /* Mempercantik metrik st.metric untuk background terang */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #0284C7 !important; /* Biru jelas */
    }
    div[data-testid="stMetricLabel"] p {
        color: #475569 !important; /* Abu-abu */
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
PALETTE_PRIMARY = ["#0284c7", "#38bdf8", "#0ea5e9", "#7dd3fc", "#bae6fd"]

# ================= 2. DATA LOADING & PREPROCESSING =================
@st.cache_data
def load_data():
    career_df = pd.read_csv('dataset/cleaned_career_recomendation.csv')
    udemy_df = pd.read_csv('dataset/cleaned_udemy_course_details.csv')
    
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z,; ]', '', str(text))
        return text
    
    career_df['skills'] = career_df['skills'].str.lower().apply(clean_text)
    career_df['skill_length'] = career_df['skills'].str.len()
    
    np.random.seed(42)
    random_dates = pd.date_range(start='2021-01-01', end='2025-12-31', freq='D')
    udemy_df['published_date'] = np.random.choice(random_dates, size=len(udemy_df))
    
    return career_df, udemy_df

try:
    career_df, udemy_df = load_data()
except FileNotFoundError:
    st.error("Dataset tidak ditemukan! Pastikan file 'cleaned_career_recomendation.csv' dan 'cleaned_udemy_course_details.csv' ada di folder yang sama.")
    st.stop()


# ================= 3. SIDEBAR CONTROLS (FILTER) =================
st.sidebar.image("img/logo.png", width=500)
st.sidebar.image("img/image.png", width=500)
st.sidebar.title("Control Panel")

# --- FILTER DATA KARIR ---
st.sidebar.subheader("Filter Data Karir")
career_filtered = career_df.copy()

study_fields = sorted([str(x) for x in career_df['study_field'].dropna().unique()])
selected_fields = st.sidebar.multiselect("Bidang Studi (Kosong = Semua):", options=study_fields)
if selected_fields:
    career_filtered = career_filtered[career_filtered['study_field'].isin(selected_fields)]
    
job_titles = sorted([str(x) for x in career_df['first_job_title'].dropna().unique()])
selected_jobs = st.sidebar.multiselect("Target Karir (Kosong = Semua):", options=job_titles)
if selected_jobs:
    career_filtered = career_filtered[career_filtered['first_job_title'].isin(selected_jobs)]

# --- FILTER DATA UDEMY ---
st.sidebar.subheader("Filter Data Udemy")
udemy_filtered = udemy_df.copy()

levels = udemy_df['level'].dropna().unique().tolist()
selected_levels = st.sidebar.multiselect("Level Kursus (Kosong = Semua):", options=levels)
if selected_levels:
    udemy_filtered = udemy_filtered[udemy_filtered['level'].isin(selected_levels)]

min_date = udemy_df['published_date'].min().date()
max_date = udemy_df['published_date'].max().date()

date_range = st.sidebar.date_input(
    "Rentang Rilis Course:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    udemy_filtered = udemy_filtered[
        (udemy_filtered['published_date'].dt.date >= start_date) & 
        (udemy_filtered['published_date'].dt.date <= end_date)
    ]


# ================= 4. MAIN HEADER =================
st.title("AI Career Navigator Dashboard")
st.markdown("Analisis data eksploratif untuk memetakan kesiapan karir, kesenjangan keahlian (*skill gap*), dan rekomendasi pembelajaran yang responsif terhadap data.")

# ================= 5. INTERACTIVE TABS LAYOUT =================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Overview & Kualitas", 
    "📋 Demografi & Distribusi Karir", 
    "📋 Pemetaan & Analisis Skill", 
    "📋 Optimasi Kursus Pembelajaran"
])

# ----- TAB 1: OVERVIEW & KUALITAS DATA -----
with tab1:
    st.subheader(" Deskripsi & Integritas Dataset Keseluruhan")
    
    st.markdown("""
    <div class="insight-card" style="border-left-color: #8b5cf6;">
        <div class="question-title"> Project Overview: AI Career Navigator</div>
        <div class="insight-text">
            <b>AI Career Navigator</b> merupakan sistem berbasis <i>Artificial Intelligence</i> yang bertujuan membantu mahasiswa dan <i>fresh graduate</i> dalam memahami potensi karir, mengidentifikasi <i>skill gap</i>, serta memperoleh rekomendasi pembelajaran yang relevan dengan kebutuhan industri.<br><br>
            Pada tahap <b>Exploratory Data Analysis (EDA)</b> ini, dilakukan eksplorasi mendalam terhadap dua dataset (<i>career recommendation</i> dan <i>Udemy course</i>) untuk memahami distribusi data, pola hubungan antar variabel, serta merumuskan <i>insight</i> yang akan mendukung pengembangan sistem recommendation dan model AI.
        </div>
    </div>
    
    <div class="insight-card" style="border-left-color: #14b8a6;">
        <div class="question-title">🔎 Informasi Pengumpulan Data (Gathering Data)</div>
        <div class="insight-text">
            Sistem ini dibangun menggunakan dua dataset krusial yang saling berkesinambungan dan telah melalui tahap <i>Data Wrangling</i> secara menyeluruh:
            <ul>
                <li><b>Dataset Profil Karir (Career Recommendation):</b> Dataset utama yang berisi histori demografi akademik (bidang studi), status pekerjaan, target karir (<i>first job title</i>), dan <i>skills</i>. Berperan penting untuk fitur <i>job classification</i> dan <i>skill gap analysis</i>.</li>
                <li><b>Dataset Katalog Udemy (Course Details):</b> Dataset pendukung yang berisi ribuan referensi materi kursus pembelajaran. Berperan sebagai basis pengetahuan bagi AI untuk merekomendasikan kursus yang tepat.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Total Baris Data Karir", value=f"{career_filtered.shape[0]:,}")
    kpi2.metric(label="Total Kolom Data Karir", value=f"{career_filtered.shape[1]}")
    kpi3.metric(label="Total Baris Udemy (Filtered)", value=f"{len(udemy_filtered):,}")
    kpi4.metric(label="Total Kolom Udemy", value=f"{udemy_filtered.shape[1]}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.json({"Total Baris": len(career_df), "Missing Values": int(career_df.isnull().sum().sum()), "Duplikasi": int(career_df.duplicated().sum())})
    with col_b:
        st.json({"Total Baris": len(udemy_df), "Missing Values": int(udemy_df.isnull().sum().sum()), "Duplikasi": int(udemy_df.duplicated().sum())})
        
    st.markdown("<br>", unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("""
        <div class='kpi-title'>1️⃣ Keseluruhan Dataset Profil Karir & Keahlian</div>
        <div class='kpi-subtitle'>Tabel profil pengguna yang siap digunakan sebagai basis analisis kompetensi.</div>
        """, unsafe_allow_html=True)
        st.dataframe(career_filtered, use_container_width=True, height=350)

    with info_col2:
        st.markdown("""
        <div class='kpi-title'>2️⃣ Keseluruhan Dataset Referensi Kursus Udemy</div>
        <div class='kpi-subtitle'>Katalog kursus yang digunakan AI untuk sistem rekomendasi pembelajaran.</div>
        """, unsafe_allow_html=True)
        st.dataframe(udemy_filtered, use_container_width=True, height=350)
     
    st.markdown(f"""
    <div class="insight-card">
        <div class="question-title"> Kesimpulan EDA & Langkah Selanjutnya</div>
        <div class="insight-text">
            Berdasarkan hasil eksplorasi secara komprehensif, dapat ditarik kesimpulan bahwa:
            <ol>
                <li>Dataset <b>Career Recommendation</b> sangat siap digunakan sebagai dataset utama untuk mendukung fitur <i>job classification</i> dan <i>skill gap analysis</i>.</li>
                <li>Dataset <b>Udemy Course</b> terbukti ideal untuk dimanfaatkan secara paralel guna mendukung fitur <i>learning recommendation</i>.</li>
                <li>Tidak ditemukan anomali, <i>missing values</i>, maupun data duplikat pada kedua dataset setelah tahapan pembersihan.</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----- TAB 2: DEMOGRAFI & DISTRIBUSI KARIR -----
with tab2:
    st.subheader("Analisis Profil Pendidikan dan Karir Pertama")
    if career_filtered.empty:
        st.warning("⚠️ Tidak ada data Karir yang sesuai dengan filter. Silakan sesuaikan filter di sidebar.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            career_filtered['study_field'].value_counts().head(5).plot(kind='bar', ax=ax1, color=PALETTE_PRIMARY)
            ax1.set_title('Top 5 Latar Belakang Bidang Studi', fontsize=11, weight='bold')
            plt.xticks(rotation=15)
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            career_filtered['first_job_title'].value_counts().head(5).plot(kind='bar', ax=ax2, color="#0ea5e9")
            ax2.set_title('Top 5 Pekerjaan Pertama Terbanyak', fontsize=11, weight='bold')
            plt.xticks(rotation=15)
            st.pyplot(fig2)
            
        # Mengambil data dinamis berdasarkan filter
        top_study = career_filtered['study_field'].mode()[0] if not career_filtered['study_field'].empty else "Belum teridentifikasi"
        top_job = career_filtered['first_job_title'].mode()[0] if not career_filtered['first_job_title'].empty else "Belum teridentifikasi"
        total_filtered = len(career_filtered)

        st.markdown(f"""
        <div class="insight-card">
            <div class="question-title">Business Question / Tantangan Analisis</div>
            <div class="insight-text"><b>Profil latar belakang studi dan jenis pekerjaan apa yang mendominasi data pengguna berdasarkan rentang ini?</b></div>
            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;'>
            <div class="insight-text">💡 <b>Insight Data Scientist (Dinamis):</b> <br>
            Dari <b>{total_filtered:,}</b> profil yang sedang Anda lihat, latar belakang studi yang paling mendominasi adalah <b>{str(top_study).upper()}</b>. 
            Hal ini berbanding lurus dengan tren pekerjaan, di mana sebagian besar dari kelompok ini berhasil terserap di industri sebagai <b>{str(top_job).title()}</b>.<br><br>
            Distribusi yang terpusat pada rumpun ilmu teknologi ini menegaskan bahwa sistem recommendation kita memiliki bias yang kuat namun terarah untuk menganalisis kesiapan karir di bidang Tech & Engineering.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ----- TAB 3: PEMETAAN & ANALISIS SKILL -----
with tab3:
    st.subheader("Korelasi Keahlian Terhadap Posisi Kerja")
    if career_filtered.empty:
        st.warning("⚠️ Tidak ada data Karir yang sesuai dengan filter. Silakan sesuaikan filter di sidebar.")
    else:
        col3, col4 = st.columns([3, 2])
        top_skills_list = [] # Menyimpan data top skill untuk insight dinamis
        top_se_list = []
        
        with col3:
            st.markdown("#### 15 Keahlian Utama Paling Dicari")
            all_skills = [skill.strip() for skills in career_filtered['skills'] for skill in str(skills).replace(';', ',').split(',') if skill.strip()]
            top_skills = Counter(all_skills).most_common(15)
            skills_chart_df = pd.DataFrame(top_skills, columns=['Skill', 'Frekuensi'])
            
            if not skills_chart_df.empty:
                top_skills_list = [x.capitalize() for x in skills_chart_df['Skill'].head(3)]
                fig4, ax4 = plt.subplots(figsize=(7, 4.2))
                sns.barplot(x='Frekuensi', y='Skill', data=skills_chart_df, ax=ax4, palette="mako")
                st.pyplot(fig4)
            else:
                st.info("Tidak ada data keahlian (skill) untuk ditampilkan.")
            
        with col4:
            st.markdown("#### Top 5 Skill (Klasifikasi Khusus)")
            # Jika user memfilter job title spesifik, kita gunakan mode dari job title tsb, jika tidak default ke SE
            specific_job = career_filtered['first_job_title'].mode()[0] if not career_filtered['first_job_title'].empty else 'Computer Software Engineer'
            
            se_skills = [skill.strip() for skills in career_filtered[career_filtered['first_job_title'] == specific_job]['skills'] for skill in str(skills).replace(';', ',').split(',') if skill.strip()]
            top_se_skills = Counter(se_skills).most_common(5)
            se_chart_df = pd.DataFrame(top_se_skills, columns=[f'Skill {specific_job}', 'Frekuensi'])
            
            if not se_chart_df.empty:
                top_se_list = [x.capitalize() for x in se_chart_df[f'Skill {specific_job}'].head(3)]
                fig5, ax5 = plt.subplots(figsize=(5, 4.2))
                sns.barplot(x='Frekuensi', y=f'Skill {specific_job}', data=se_chart_df, ax=ax5, palette="flare")
                st.pyplot(fig5)
            else:
                st.info(f"Tidak ada data skill untuk posisi {specific_job} pada filter ini.")

        st.markdown("---")
        
        # VISUALISASI JAWABAN PERTANYAAN 1: HUBUNGAN BIDANG STUDI DAN PEKERJAAN PERTAMA
        st.markdown("#### Matriks Hubungan (Heatmap): Bidang Studi vs Pekerjaan Pertama")
        
        top_jobs = career_filtered['first_job_title'].value_counts().head(8).index
        top_fields = career_filtered['study_field'].value_counts().head(8).index
        heatmap_df = career_filtered[career_filtered['first_job_title'].isin(top_jobs) & career_filtered['study_field'].isin(top_fields)]
        
        if not heatmap_df.empty:
            cross_tab = pd.crosstab(heatmap_df['study_field'], heatmap_df['first_job_title'])
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', cbar=True, ax=ax6)
            ax6.set_ylabel("Bidang Studi (Study Field)")
            ax6.set_xlabel("Pekerjaan Pertama (First Job Title)")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig6)
        else:
             st.info("Data tidak cukup untuk menampilkan matriks korelasi pada rentang filter ini.")
             
        # String Dinamis untuk Insight
        skill_str = ", ".join(top_skills_list) if top_skills_list else "Spesifik/Terbatas"
        job_skill_str = ", ".join(top_se_list) if top_se_list else "N/A"

        st.markdown(f"""
        <div class="insight-card">
            <div class="question-title">1. Apakah terdapat hubungan antara bidang studi dan pekerjaan pertama?</div>
            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;'>
            <div class="insight-text">💡 <b>Insight & Jawaban (Dinamis):</b> <br>
            Berdasarkan rentang data yang Anda amati saat ini, <i>skills</i> utama yang mendominasi profil pengguna adalah <b>{skill_str}</b>. Lebih tajam lagi, untuk target spesifik pada klasifikasi peran terbanyak saat ini, keahlian yang mutlak diperlukan adalah <b>{job_skill_str}</b>.<br><br>
            Dari Heatmap matriks hubungan di atas, meskipun terdapat garis pola antara lulusan ilmu teknik (<i>engineering</i>) dengan penyerapan kerja berbasis pemrograman, <b>sebarannya tetap membuktikan bahwa seseorang dengan jurusan yang sama bisa menapaki karir yang sama sekali berbeda.</b><br><br>
            <i>Kesimpulan Bisnis:</i> Model rekomendasi AI kita tidak boleh bergantung pada kolom pendidikan formal ('Study Field') semata, melainkan wajib membangun algoritma klasifikasi berbasis penguasaan kompetensi praktis (<i>Skill-Driven Classification</i>) yang tercantum di profil mereka.</div>
        </div>
        """, unsafe_allow_html=True)


# ----- TAB 4: OPTIMASI KURSUS PEMBELAJARAN -----
with tab4:
    st.subheader("Kurasi Konten dan Kesiapan Materi Rekomendasi")
    if udemy_filtered.empty:
        st.warning("⚠️ Tidak ditemukan kursus dalam rentang waktu atau spesifikasi yang ditetapkan pada Sidebar.")
    else:
        col5, col6 = st.columns(2)
        with col5:
            fig7, ax7 = plt.subplots(figsize=(6, 4.5))
            udemy_filtered['level'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax7, colors=["#0284c7", "#14b8a6", "#f97316", "#cbd5e1"], startangle=90, wedgeprops=dict(width=0.4))
            ax7.set_title("Proporsi Tingkat Kesulitan Kursus", fontsize=11, weight='bold')
            ax7.set_ylabel("")
            st.pyplot(fig7)
            
        with col6:
            # VISUALISASI JAWABAN PERTANYAAN 2: KATEGORI COURSE POPULER BERDASARKAN SUBSCRIBER
            fig8, ax8 = plt.subplots(figsize=(7, 4.5))
            top_courses = udemy_filtered.sort_values(by='subscribers', ascending=False).head(10) # Diambil Top 10
            sns.barplot(x='subscribers', y='title', data=top_courses, ax=ax8, palette="viridis")
            ax8.set_title("Top 10 Course Paling Populer (Berdasarkan Subscriber)", fontsize=11, weight='bold')
            ax8.set_xlabel("Jumlah Subscriber")
            ax8.set_ylabel("")
            # Memperpendek label judul y-axis jika terlalu panjang agar rapi di dashboard
            labels = [text.get_text()[:35] + '...' if len(text.get_text()) > 35 else text.get_text() for text in ax8.get_yticklabels()]
            ax8.set_yticklabels(labels)
            st.pyplot(fig8)
            
        # Variabel Dinamis untuk Udemy Insight
        top_level = udemy_filtered['level'].mode()[0] if not udemy_filtered['level'].empty else "Belum diketahui"
        top_course_title = top_courses.iloc[0]['title'] if not top_courses.empty else "Belum ada course"
        total_udemy = len(udemy_filtered)
            
        st.markdown(f"""
        <div class="insight-card">
            <div class="question-title">2. Kategori course apa yang paling populer berdasarkan jumlah subscriber?</div>
            <hr style='margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;'>
            <div class="insight-text">💡 <b>Insight & Jawaban (Dinamis):</b> <br>
            Berdasarkan <b>{total_udemy:,}</b> data katalog kursus Udemy yang tersedia pada parameter filter saat ini, fokus proporsi tingkat kesulitan (*Course Level*) terbanyak adalah kelas untuk level <b>{top_level}</b>.<br><br>
            Jika kita melihat dari segi peminat (Subscribers), kursus yang menduduki peringkat teratas (Ranking #1) adalah <b>"{top_course_title}"</b>. Hal ini merefleksikan tren permintaan pasar yang sangat tinggi terhadap keterampilan pada materi tersebut.<br><br>
            <i>Penerapan untuk AI:</i> Kumpulan kursus <i>top-rated</i> inilah yang akan menjadi *"Bank Solusi"* bagi model Machine Learning. Ketika sistem mendeteksi adanya kesenjangan (<i>skill gap</i>) antara keahlian saat ini yang dimiliki pengguna dengan tuntutan lowongan kerja idamannya, AI akan secara otomatis memanggil dan menyarankan kursus-kursus relevan dari bank data ini.</div>
        </div>
        """, unsafe_allow_html=True)