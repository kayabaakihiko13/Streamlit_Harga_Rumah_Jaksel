import streamlit as st
import json
from load_model import load_model_regression, load_model_classification
import os


def display_house_category_image(predicted_label):
    # Mendapatkan nama file gambar berdasarkan predicted_label
    image_filename = predicted_label.lower() + "_house_categories.jpg"

    # Mendapatkan path lengkap ke file gambar
    image_path = os.path.join("image", image_filename)

    # Menampilkan gambar
    st.image(
        image_path, caption=predicted_label + " House Categories", use_column_width=True
    )


# Load the JSON files
with open("model/tipe_kamar_tidur.json") as f:
    try:
        tipe_kamar_tidur = json.load(f)
    except json.JSONDecodeError:
        st.error("Failed to load tipe_kamar_tidur JSON file.")
        st.stop()

with open("model/tipe_kamar_mandi.json") as f:
    try:
        tipe_kamar_mandi = json.load(f)
    except json.JSONDecodeError:
        st.error("Failed to load tipe_kamar_mandi JSON file.")
        st.stop()

with open("model/label_kategori.json") as f:
    try:
        label_categories = json.load(f)
    except json.JSONDecodeError:
        st.error("Failed to load label_categories JSON file.")
        st.stop()

# Load regression and classification models
model_regression = load_model_regression("model/best_regression_model.pkl")
model_classification = load_model_classification("model/best_classification_model.pkl")

# Load models
model_regression.load_model()
model_classification.load_model()

# Set the page configuration
st.set_page_config(page_title="Predict House price in Jaksel", layout="wide")

# Create a container to center the form
with st.container():
    # Create columns to center the form elements
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Create a form
        with st.form("my_form"):
            st.header("Calculation House Price in Jaksel")

            # Form elements
            luas_tanah = st.number_input("Luas Tanah", min_value=0)
            luas_bangunan = st.number_input("Luas Bangunan", min_value=0)

            # Extract available bedroom options
            kamar_tidur_options = [str(item) for item in sorted(tipe_kamar_tidur)]
            jumlah_kamar_tidur = st.selectbox(
                "Jumlah Kamar Tidur", options=kamar_tidur_options
            )

            # Derive the corresponding options for jumlah_kamar_mandi based on jumlah_kamar_tidur
            jumlah_kamar_mandi_options = [
                str(item) for item in sorted(tipe_kamar_mandi)
            ]
            jumlah_kamar_mandi = st.selectbox(
                "Jumlah Kamar Mandi", options=jumlah_kamar_mandi_options
            )

            ada_garasi = st.checkbox("Ada Garasi")

            # Form submission button
            submit_button = st.form_submit_button("Submit")

            if submit_button:
                # Ensure all elements are converted to numeric values
                new_value_regression = [
                    float(luas_tanah),
                    float(luas_bangunan),
                    int(jumlah_kamar_tidur),
                    int(jumlah_kamar_mandi),
                    int(ada_garasi),
                ]
                predict_price = model_regression.predict(new_value_regression)

                # Ensure all elements are converted to numeric values
                new_value_classification = [
                    float(luas_tanah),
                    float(luas_bangunan),
                    int(jumlah_kamar_tidur),
                    int(jumlah_kamar_mandi),
                    round(predict_price[0], 4),
                    int(ada_garasi),
                ]
                predict_classification = model_classification.prediction(
                    new_value_classification
                )[0]
                reverse_label_categories = {v: k for k, v in label_categories.items()}
                # Map the predicted classification back to its corresponding label
                predicted_label = reverse_label_categories.get(
                    predict_classification, "Unknown"
                )

                st.write(
                    f"The predicted house price is: **Billion Rp {predict_price[0]:,.4f}**"
                )
                st.write(f"The predicted house category is: **{predicted_label}**")
                display_house_category_image(predicted_label)
