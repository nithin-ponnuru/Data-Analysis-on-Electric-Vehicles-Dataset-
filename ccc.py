import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="EV Data Analysis Dashboard", layout="wide")

st.title("⚡ Electric Vehicle Data Analysis Dashboard")

# === Upload Dataset ===
uploaded_file = st.file_uploader("Upload EV Dataset (Excel File)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # Data Cleaning
    df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
    df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

    # === 1. Top 10 Makes ===
    st.subheader("1️⃣ Top 10 Electric Vehicle Makes")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    make_counts = df['Make'].value_counts().head(10)
    sns.barplot(x=make_counts.index, y=make_counts.values, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # === 2. Electric Range Distribution ===
    st.subheader("2️⃣ Electric Range Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    # === 3. Pair Plot ===
    st.subheader("3️⃣ Pair Plot of Numeric Variables")
    numeric_cols = ['Electric Range', 'Base MSRP', 'Model Year']
    fig3 = sns.pairplot(df[numeric_cols].dropna())
    st.pyplot(fig3)

    # === 4. Box Plot by Vehicle Type ===
    st.subheader("4️⃣ Electric Range by Vehicle Type")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Electric Vehicle Type', y='Electric Range', data=df, ax=ax4)
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    # === 5. Correlation Heatmap ===
    st.subheader("5️⃣ Correlation Heatmap")
    fig5, ax5 = plt.subplots()
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    # === 6. Scatter Plot ===
    st.subheader("6️⃣ Electric Range vs Model Year")
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x='Model Year', y='Electric Range', 
                    data=df, hue='Electric Vehicle Type', ax=ax6)
    st.pyplot(fig6)

    # === Z-Test ===
    st.subheader("7️⃣ Z-Test: BEV vs PHEV Electric Range")

    bev_data = df[df['Electric Vehicle Type'] == 
                  'Battery Electric Vehicle (BEV)']['Electric Range'].dropna()

    phev_data = df[df['Electric Vehicle Type'] == 
                   'Plug-in Hybrid Electric Vehicle (PHEV)']['Electric Range'].dropna()

    bev_mean = bev_data.mean()
    phev_mean = phev_data.mean()
    bev_std = bev_data.std()
    phev_std = phev_data.std()
    bev_n = len(bev_data)
    phev_n = len(phev_data)

    z_score = (bev_mean - phev_mean) / np.sqrt((bev_std**2 / bev_n) + (phev_std**2 / phev_n))
    p_value = stats.norm.sf(abs(z_score)) * 2

    st.write(f"BEV Mean Range: {bev_mean:.2f}")
    st.write(f"PHEV Mean Range: {phev_mean:.2f}")
    st.write(f"Z-Score: {z_score:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value < 0.05:
        st.success("Statistically significant difference between BEVs and PHEVs.")
    else:
        st.warning("No statistically significant difference found.")

    # === 8. EV Count by Year ===
    st.subheader("8️⃣ EV Count by Model Year")
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    year_counts = df['Model Year'].value_counts().sort_index()
    sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax7)
    plt.xticks(rotation=45)
    st.pyplot(fig7)

    # === 9. Electric Range by Top Makes ===
    st.subheader("9️⃣ Electric Range by Top Makes")
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    top_makes = df['Make'].value_counts().head(10).index
    sns.boxplot(x='Make', y='Electric Range', 
                data=df[df['Make'].isin(top_makes)], ax=ax8)
    plt.xticks(rotation=45)
    st.pyplot(fig8)

    # === 10. CAFV Eligibility ===
    st.subheader("🔟 CAFV Eligibility Status")
    fig9, ax9 = plt.subplots()
    eligibility_counts = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
    sns.barplot(x=eligibility_counts.index, y=eligibility_counts.values, ax=ax9)
    plt.xticks(rotation=45)
    st.pyplot(fig9)

    st.success("Analysis Complete ✅")
