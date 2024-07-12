import streamlit as st
import numpy as np
import cv2
import os
from tempfile import NamedTemporaryFile
from descriptor import glcm, bitdesc
from distances import retrieve_similar_image

def load_signatures(descriptor_type):
    if descriptor_type == "GLCM":
        return np.load('signatures_glcm.npy', allow_pickle=True)
    elif descriptor_type == "BIT":
        return np.load('signatures_bit.npy', allow_pickle=True)
    else:
        return None

def main():
    st.set_page_config(page_title="Récupération d'images basée sur le contenu (CBIR)", page_icon='Smile robot', layout="wide")
    st.title("🔍 Récupération d'images basée sur le contenu (CBIR)")

    st.sidebar.header("🔧 Descriptors & Distances")
    with st.sidebar:
        descriptor_options = ["GLCM", "BIT"]
        selected_descriptor = st.radio("Descriptor", descriptor_options)

        distance_options = ["Manhattan", "Euclidean", "Chebyshev", "Canberra"]
        selected_distance = st.radio("Distance", distance_options)
        max_distance = st.slider("Distance maximale", min_value=0.0, max_value=100.0, value=50.0)

    signatures = load_signatures(selected_descriptor)
    if signatures is not None:
        total_images = len(signatures)

        st.write("Veuillez téléverser votre image:")
        uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_image_path = temp_file.name

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(uploaded_file, caption='Image téléversée.', use_column_width=False)

            with col2:
                st.write(f"Descripteur sélectionné : {selected_descriptor}")
                st.write(f"Distance sélectionnée : {selected_distance}")
                st.write(f"Distance maximale : {max_distance}")

            features = None
            if selected_descriptor == "GLCM":
                features = glcm(temp_image_path)[:6]
            elif selected_descriptor == "BIT":
                features = bitdesc(temp_image_path)[:14]

            sorted_results = retrieve_similar_image(signatures, features, selected_distance.lower(), total_images)

            filtered_results = [result for result in sorted_results if result[1] <= max_distance]

            st.sidebar.write(f"Nombre total d'images similaires trouvées : {len(filtered_results)}")
            num_res_options = list(range(1, len(filtered_results) + 1))
            selected_num_res = st.sidebar.slider("Num Res", min_value=0, max_value=len(num_res_options), value=min(5, len(num_res_options)))



            st.write(f"Top {selected_num_res} résultats les plus proches :")
            cols = st.columns(3)
            for i, result in enumerate(filtered_results[:selected_num_res]):
                col = cols[i % 3]
                col.write(f"Image : {result[0]}, Distance : {result[1]:.6f}, Label : {result[2]}")
                similar_image = cv2.imread(result[0])
                col.image(similar_image, caption=f"Similar Image (Distance: {result[1]:.6f})", use_column_width=True)

            os.remove(temp_image_path)

if __name__ == '__main__':
    main()
