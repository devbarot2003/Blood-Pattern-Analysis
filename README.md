# Bloodstain Pattern Analysis – Variant B (ML Pipeline)

This project contains a notebook implementing a **Bloodstain pattern analysis pipeline** for classifying images as **Gunshot** or **Impact**.

The workflow includes image loading, stain segmentation, feature extraction, model training, and final inference with a fallback method for low-confidence results.

Images that are being used will be ran through a verification code to check if they are too big in size or the file is corrupted before running through feature extraction.

## What the Notebook Does

- Loads very large images safely using PIL 
- Segments stains using **HSV-based Variant B segmentation**  
- Extracts stain-level and pattern-level features  
- Builds a feature dataset for ML training  
- Trains a **XGBoost classifier**  
- Saves the trained model + feature columns  
- Runs **single image prediction** with:
  - Main Variant B path  
  - Automatic contingency (backup) segmentation if confidence is low  

##  Inputs & Outputs

**Inputs**
- High-resolution images in category folders (GS, IS, HP, C)
- Notebook cells (feature extraction, training, inference)

**Outputs**
- Feature dataset (`features.csv`)
- Trained model bundle (`variantB_xgb.joblib`)
- Per-image prediction (label + confidence)
- Optional segmentation masks for visualization

## How to Use

1. Place your dataset inside the project folder  
2. Run the notebook top-to-bottom to build features and train  
3. Use the last section (“Test on one image”) to classify new images  
4. The model automatically switches to contingency mode if needed


## To do 

Eventhough the current pipeline gives us output in a descent way there are some errors which needs to be focused for which we are planning to do some improvements to boost accuracy and robustness which includes: 
  - Increase dataset to have more features for traininga and refining some within directories.
  - Creating a dual-model architecture where separate XGBoost Classifiers are trained for Gunshot and Impact classes. 
  - Their outputs are combined to make more reliable predictions. Feature extraction has been expanded and optimized(added some more features to be extracted), and all images will definietly downscale to a safe maximum resolution during processing, improving speed without losing stain-level detail. 
  - The notebook also includes the multi-pass HSV segmentation to capture even tiny droplets,  plus a refined contingency superpixel method that activates when confidence drops below a threshold(right now we have some erroes that needs to be fixed). 
  - Together, these changes we believe that the model generalize better, reduce misclassifications, and handle challenging or low-quality images more effectively.
