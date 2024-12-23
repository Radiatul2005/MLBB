import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules

# Judul dan Deskripsi Aplikasi
st.title("MLBB Hero Association Analysis App")
st.write("Aplikasi ini menyediakan preprocessing, peningkatan data, dan penambangan aturan asosiasi untuk menganalisis atribut hero.")

# Upload Dataset
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")

# Inisialisasi session state untuk data yang telah diproses
if 'data_cleaned' not in st.session_state:
    st.session_state['data_cleaned'] = None
    st.session_state['original_data'] = None

# Load dataset jika diunggah
if uploaded_file is not None:
    # Memuat dataset
    st.session_state['original_data'] = pd.read_csv(uploaded_file)

    # Tampilkan dataset lengkap
    st.subheader("Dataset Lengkap")
    st.dataframe(st.session_state['original_data'])

    # Preprocessing dataset
    features = [
        'defense_overall', 'offense_overall', 'skill_effect_overall',
        'difficulty_overall', 'movement_spd', 'magic_defense', 'mana',
        'hp_regen', 'physical_atk', 'physical_defense', 'hp',
        'attack_speed', 'mana_regen', 'win_rate', 'pick_rate', 'ban_rate'
    ]

    if st.button("Jalankan Preprocessing"):
        # Normalisasi fitur numerik
        scaler = MinMaxScaler()
        st.session_state['data_cleaned'] = st.session_state['original_data'].copy()
        st.session_state['data_cleaned'][features] = scaler.fit_transform(
            st.session_state['data_cleaned'][features]
        )

        # Binarisasi fitur untuk penambangan aturan asosiasi
        binarized_data = st.session_state['data_cleaned'][features] > 0.5
        binarized_data = binarized_data.astype(int)

        # Tambahkan kembali kolom non-numerik
        binarized_data['hero_name'] = st.session_state['data_cleaned']['hero_name']
        st.session_state['data_cleaned'] = binarized_data

        st.success("Preprocessing selesai!")
        st.subheader("Dataset yang Diproses")
        st.dataframe(st.session_state['data_cleaned'])

    # Pilih hero untuk melihat atribut
    st.subheader("Pilih Hero untuk Lihat Atribut")
    hero_names = st.session_state['data_cleaned']['hero_name'].unique()
    selected_hero = st.selectbox("Pilih Hero", hero_names)

    # Tampilkan atribut hero yang dipilih dalam bentuk grafik
    if selected_hero:
        hero_data = st.session_state['data_cleaned'][st.session_state['data_cleaned']['hero_name'] == selected_hero]
        hero_data = hero_data[features].iloc[0]

        st.subheader(f"Atribut Hero {selected_hero}")
        
        # Buat bar chart untuk atribut hero
        fig, ax = plt.subplots(figsize=(10, 6))
        hero_data.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f"Atribut Hero {selected_hero}")
        ax.set_ylabel("Nilai Atribut")
        ax.set_xlabel("Atribut")
        ax.set_xticklabels(hero_data.index, rotation=45, ha="right")
        st.pyplot(fig)

    # Penambangan Aturan Asosiasi
    st.subheader("Penambangan Aturan Asosiasi")
    if st.session_state['data_cleaned'] is not None:
        # Pastikan hanya data biner yang digunakan
        binary_data = st.session_state['data_cleaned'].drop(columns=['hero_name'], errors='ignore')

        # Jalankan algoritma Apriori
        min_support = st.slider("Minimum Support", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        frequent_itemsets = apriori(binary_data, min_support=min_support, use_colnames=True)

        # Filter frequent itemsets yang di bawah nilai min_support
        frequent_itemsets = frequent_itemsets[frequent_itemsets['support'] >= min_support]

        # Buat aturan asosiasi
        min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

        try:
            # Tambahkan num_itemsets sesuai kebutuhan
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

            # Hilangkan istilah teknis 'frozenset' untuk output yang lebih bersih
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

            # Tambahkan catatan untuk menjelaskan parameter dan output
            st.subheader("Catatan Penting")
            st.markdown("""
            - **Minimum Support**: Ini adalah nilai ambang untuk menentukan seberapa sering sebuah kombinasi item muncul dalam data. Misalnya, jika Anda memilih 0.1 (10%), hanya kombinasi yang muncul di setidaknya 10% data yang akan dipertimbangkan.
            - **Minimum Confidence**: Ini menunjukkan seberapa sering aturan ditemukan benar ketika item di bagian awal aturan (antecedent) ada. Contoh, confidence 0.5 berarti aturan tersebut benar dalam 50% kasus.
            - **Aturan Asosiasi**: Aturan ini menunjukkan hubungan antara atribut-atribut dalam data. Kolom 'antecedents' (sebelumnya) dan 'consequents' (konsekuensi) menunjukkan item yang berasosiasi.
            """)

            st.write("Aturan Asosiasi:")
            st.dataframe(rules)

            # Filter aturan untuk wawasan khusus
            if st.checkbox("Filter Aturan"):
                antecedent = st.text_input("Filter berdasarkan Antecedent (misalnya, 'defense_overall')")
                if antecedent:
                    filtered_rules = rules[rules['antecedents'].apply(lambda x: antecedent in str(x))]
                    st.write("Aturan yang Difilter")
                    st.dataframe(filtered_rules)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat menghasilkan aturan asosiasi: {str(e)}")

else:
    st.write("Silakan unggah file CSV untuk melanjutkan.")
