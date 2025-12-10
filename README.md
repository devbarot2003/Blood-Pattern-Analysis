# Bloodstain Pattern Analysis – Variant B (ML Pipeline)

This notebook implements a bloodstain pattern classification pipeline for predicting whether an image represents a Gunshot or Impact pattern.

The workflow includes safe image loading, stain segmentation, grayscale coated masking, feature extraction, model training, probability calibration, and a contingency fallback for low-confidence results.

All images are checked for size validity before processing to avoid failures with extremely large or corrupted files.

## What the Notebook Does

- Loads very large images safely and downsizes them only when necessary
- Segments stains using the final combined RGB + HSV segmentation
- Generates both a highlight image and a grayscale coated mask
- Extracts geometric, color, and spatial distribution features
- Builds a feature dataset for ML training
- Trains a CatBoost classifier on the extracted features
- Applies probability calibration (Isotonic + Beta)
- Performs single-image prediction with: Main calibrated model and Automatic contingency fallback when confidence or segmentation quality is low

##  Inputs & Outputs

**Inputs**
- High-resolution images in category folders (GS, IS, HP, C)
- Notebook cells (feature extraction, training, inference)

**Outputs**
- features.csv – Extracted feature dataset
- model_bundle.joblib – Saved CatBoost model + calibrators + metadata
- Highlight mask image (optional)
- Coated grayscale mask (optional)
- Final prediction (label + confidence + contingency flag)

## How to Use

1. Place your dataset inside the project folder
2. Run the notebook from top to bottom if its's the first time(Can Ignore the building dataset and training if U run once cause it will be saved)
3. A feature dataset and a model bundle will be generated
4. Use the final section (“Test on one image”) to classify new images
5. The model automatically switches to the contingency path when confidence is low or segmentation quality is weak


## Future Improvements 

Although the current pipeline produces stable predictions, several improvements are planned to further increase robustness and accuracy:
- Expand the dataset and refine class folders for better coverage
- Improve segmentation using modern models (e.g., SAM or U-Net) to reduce mask errors
- Enhance feature extraction with more spatial and texture-based descriptors
- Optimize thresholds for confidence and contingency activation dynamically
- Add multi-image consistency checks for scenes with multiple angles
- Improve fallback segmentation for extremely low-stain or noisy images

These refinements aim to improve generalization, reduce misclassification risk, and handle challenging or low-quality images more effectively.
