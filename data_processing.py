import cv2, os
from descriptor import glcm, bitdesc
import numpy as np
 
def extract_features(image_path, descriptor):
    print(f"Reading image: {image_path}")
    try:
        img = cv2.imread(image_path, 0)
        if img is not None:
            features = descriptor(img)
            print(f"Extracted features from {image_path}: {features}")
            return features
        else:
            print(f"Failed to read image: {image_path}")
            return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
 
descriptors = [glcm, bitdesc]
 
def process_datasets(root_folder):
    features_glcm = []
    features_bit = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_rel_path = os.path.join(root, file)
                print(f"Processing file: {image_rel_path}")
                if os.path.isfile(image_rel_path):
                    try:
                        folder_name = os.path.basename(os.path.dirname(image_rel_path))
                        features = glcm(image_rel_path)
                        features1=bitdesc(image_rel_path)
                        if features and features1 is not None:
                            features = features + [folder_name, image_rel_path]
                            features1=features1+[folder_name, image_rel_path]
                            features_glcm.append(features)
                            features_bit.append(features1)
                    except Exception as e:
                        print(f"Error processing file {image_rel_path}: {e}")
                else:
                    print(f"File does not exist: {image_rel_path}")
    print(f"All features: {features_glcm}")
    signatures_glcm = np.array(features_glcm)
    np.save('signatures_glcm.npy', signatures_glcm)
    print('Successfully stored!')
    print(f"All features: {features_bit}")
    signatures_bit = np.array(features_bit)
    np.save('signatures_bit.npy', signatures_bit)
    print('Successfully stored!')
# Usage example
process_datasets('Projet1_Dataset')