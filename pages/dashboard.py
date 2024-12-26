import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
import plotly.express as px

# Load dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Menambahkan kolom target untuk warna scatterplot
df['species'] = [iris.target_names[i] for i in iris.target]

# Hitung rata-rata setiap kolom numerik
mean_values = df.iloc[:, :-1].mean()

# Fungsi untuk menampilkan statistik ke Streamlit
def display_statistics(mean_values):
    st.title("Dashboard - Iris Dataset Statistics")
    st.subheader("Summary of Feature Averages")

    # Membuat kolom untuk menampilkan statistik
    col1, col2, col3, col4 = st.columns(4)

    # Menampilkan rata-rata setiap fitur menggunakan st.metric
    col1.metric(label="Sepal Length (cm)", value=f"{mean_values[0]:.2f}")
    col2.metric(label="Sepal Width (cm)", value=f"{mean_values[1]:.2f}")
    col3.metric(label="Petal Length (cm)", value=f"{mean_values[2]:.2f}")
    col4.metric(label="Petal Width (cm)", value=f"{mean_values[3]:.2f}")

# Fungsi untuk membuat barplot dengan Plotly
def create_barplot(mean_values):
    plot_data = pd.DataFrame({
        "Feature": mean_values.index,
        "Average Value": mean_values.values
    })

    fig = px.bar(
        plot_data,
        x="Feature",
        y="Average Value",
        title="Average Values of Iris Features",
        labels={"Feature": "Feature Name", "Average Value": "Mean Value"},
        color="Feature"
    )
    return fig

# Fungsi untuk membuat scatterplot dengan Plotly
def create_scatterplot(df):
    fig = px.scatter(
        df,
        x=df.columns[0],  # Sepal Length
        y=df.columns[2],  # Petal Length
        color='species',  # Species as color
        title="Scatterplot: Sepal Length vs Petal Length",
        labels={
            df.columns[0]: "Sepal Length (cm)",
            df.columns[2]: "Petal Length (cm)",
            "color": "Species"
        }
    )
    return fig

# Menampilkan statistik
display_statistics(mean_values)

# Menampilkan barplot
st.subheader("Interactive Barplot")
barplot_fig = create_barplot(mean_values)
st.plotly_chart(barplot_fig)

# Menampilkan scatterplot
st.subheader("Interactive Scatterplot")
scatterplot_fig = create_scatterplot(df)
st.plotly_chart(scatterplot_fig)
