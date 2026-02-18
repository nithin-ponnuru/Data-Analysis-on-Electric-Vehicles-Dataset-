import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Page Config
st.set_page_config(page_title="EV Data Analysis Dashboard", layout="wide")

st.title("⚡ Electric Vehicle Data Analysis Dashboard")

# Load Dataset
try:
    df = pd.read_excel("pythonca2.xlsx")
except FileNotFoundError:
    st.error("Dataset file 'pythonca2.xlsx' not found. Make sure it is in the same folder as app.py.")
    st.stop()

# ================= DATA OVERVIEW =================
st.subheader("📊 Dataset Overview")
st.write(df.head())
st.write("Dataset Shape:", df.shape)

# ================= DATA CLEANING =================
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

# ================= PLOT 1 =================
st.subheader("1️⃣ Top 10 Electric Vehicle Makes")
fig1, ax1 = plt.subplots(figsize=(10, 5))
make_counts = df['Make'].value_counts().head(10)
sns.barplot(x=make_counts.index, y=make_counts.values, ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ================= PLOT 2 =================
st.subheader("2️⃣ Distribution of Electric Range")
fig2, ax2 = plt.subplots()
sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# ================= PLOT 3 =================
st.subheader("3️⃣ Pair Plot of Numeric Variables")
numeric_cols = ['Electric Range', 'Base MSRP', 'Model Year']
pair_fig = sns.pairplot(df[numeric_cols].dropna())
st.pyplot(pair_fig)

# ================= PLOT 4 =================
st.subheader("4️⃣ Electric Range by Vehicle Type")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='Electric Vehicle Type', y='Electric Range', data=df, ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)

# ================= PLOT 5 =================
st.subheader("5️⃣ Correlation Heatmap")
fig5, ax5 = plt.subplots()
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
st.pyplot(fig5)

# ================= PLOT 6 =================
st.subheader("6️⃣ Electric Range vs Model Year")
fig6, ax6 = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x='Model Year',
    y='Electric Range',
    data=df,
    hue='Electric Vehicle Type',
    ax=ax6
)
st.pyplot(fig6)

# ================= Z-TEST =================
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

st.write(f"BEV Mean Range: {bev_mean:.2f} miles")
st.write(f"PHEV Mean Range: {phev_mean:.2f} miles")
st.write(f"Z-Score: {z_score:.4f}")
st.write(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    st.success("Statistically significant difference between BEVs and PHEVs.")
else:
    st.warning("No statistically significant difference found.")

# ================= PLOT 8 =================
st.subheader("8️⃣ EV Count by Model Year")
fig8, ax8 = plt.subplots(figsize=(10, 5))
year_counts = df['Model Year'].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax8)
plt.xticks(rotation=45)
st.pyplot(fig8)

# ================= PLOT 9 =================
st.subheader("9️⃣ Electric Range Distribution by Top 10 Makes")
fig9, ax9 = plt.subplots(figsize=(10, 5))
top_makes = df['Make'].value_counts().head(10).index
sns.boxplot(
    x='Make',
    y='Electric Range',
    data=df[df['Make'].isin(top_makes)],
    ax=ax9
)
plt.xticks(rotation=45)
st.pyplot(fig9)

# ================= PLOT 10 =================
st.subheader("🔟 CAFV Eligibility Status Count")
fig10, ax10 = plt.subplots()
eligibility_counts = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
sns.barplot(x=eligibility_counts.index, y=eligibility_counts.values, ax=ax10)
plt.xticks(rotation=45)
st.pyplot(fig10)

st.success("✅ Analysis Complete!")
