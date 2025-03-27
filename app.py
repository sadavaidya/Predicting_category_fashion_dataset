# Save this code in a file named `app.py` and run `streamlit run app.py`

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('fashion_category_classifier.h5')

# Class indices (mapping of numerical labels to class names)
class_indices = {'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4, 'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Cufflinks': 8, 'Deodorant': 9, 'Dresses': 10, 'Dupatta': 11, 'Earrings': 12, 'Flats': 13, 'Flip Flops': 14, 'Formal Shoes': 15, 'Handbags': 16, 'Heels': 17, 'Innerwear Vests': 18, 'Jackets': 19, 'Jeans': 20, 'Kajal and Eyeliner': 21, 'Kurtas': 22, 'Kurtis': 23, 'Leggings': 24, 'Lip Gloss': 25, 'Lipstick': 26, 'Nail Polish': 27, 'Necklace and Chains': 28, 'Night suits': 29, 'Nightdress': 30, 'Other': 31, 'Pendant': 32, 'Perfume and Body Mist': 33, 'Ring': 34, 'Sandals': 35, 'Sarees': 36, 'Scarves': 37, 'Shirts': 38, 'Shorts': 39, 'Skirts': 40, 'Socks': 41, 'Sports Shoes': 42, 'Sunglasses': 43, 'Sweaters': 44, 'Sweatshirts': 45, 'Ties': 46, 'Tops': 47, 'Track Pants': 48, 'Trousers': 49, 'Trunk': 50, 'Tshirts': 51, 'Tunics': 52, 'Wallets': 53, 'Watches': 54}

reverse_class_indices = {v: k for k, v in class_indices.items()}  # Reverse mapping

def predict_image(image_path):
    # Load and preprocess the image
    image_size = (224, 224)  # MobileNetV2 input size
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = reverse_class_indices[predicted_class_index]

    return predicted_class_name

# Streamlit App Interface
st.title("Fashion Product Category Classifier")
st.write("Upload an image to predict its product category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save uploaded file temporarily and predict its class
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_class_name = predict_image("temp.jpg")
    
    st.write(f"Predicted Class: {predicted_class_name}")
