import streamlit as st


# Halaman Home
def home_page():
    # Judul dan deskripsi
    st.title("Iris Dataset Exploration and Classification")
    st.subheader("Welcome to the Iris Dataset Application!")

    st.markdown("""
    This application is designed to explore and classify the famous Iris dataset. 
    It consists of three main sections:

    - **Home**: Overview of the application and dataset.
    - **Dashboard**: Interactive visualizations and data analysis tools.
    - **Model**: Predictive model for classifying Iris species.

    ---
    ### About the Iris Dataset
    The Iris dataset is a classic dataset in machine learning and statistics. It contains:

    - **150 samples** of iris flowers.
    - **4 features**: Sepal Length, Sepal Width, Petal Length, and Petal Width.
    - **3 species**: Setosa, Versicolor, and Virginica.

    The dataset is widely used for demonstrating classification algorithms and data visualization.
    """)

    # Gambar ilustrasi atau dataset Iris (opsional)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_dataset_scatterplot.svg",
        caption="Iris Dataset Scatterplot",
        use_column_width=True
    )

    # Footer
    st.info("Navigate through the app using the sidebar to explore more!")


# Menjalankan halaman jika file di-run langsung
if __name__ == "__main__":
    home_page()
