# Skin Cancer Detection with SHAP and LIME

## Overview

This project aims to build a Convolutional Neural Network (CNN) model for **skin cancer detection** using the **HAM10000 dataset**. The trained model is then explained using two popular model-agnostic explanation techniques: **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**, both of which were manually implemented in this project. The custom implementation allowed for greater control and understanding of the inner workings of these explainability methods, rather than relying on pre-existing libraries.

The goal of using SHAP and LIME is to interpret the model's predictions by identifying the important features (superpixels) in the images that influence the classification decisions. This provides transparency and helps understand how the model makes decisions, which is crucial in sensitive areas like healthcare.

## Key Components

1. **Data Preprocessing**:

   - Load the HAM10000 skin cancer dataset.
   - Resize and preprocess images for input into the CNN model.
   - Encode labels for model training.
2. **Model Training**:

   - The model used is **EfficientNetB0**, a pre-trained CNN, fine-tuned on the HAM10000 dataset.
   - Image augmentation is applied to improve model generalization.
   - The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.
3. **Model Interpretability**:

   - **SHAP**: Uses a game-theoretic approach to assign importance values to each pixel (superpixel) in the image based on how much it contributes to the final prediction.
   - **LIME**: Fits a local surrogate model to approximate the decision of the CNN and identifies which superpixels are most influential for a given prediction.
4. **Comparison of SHAP and LIME**:

   - SHAP and LIME explanations are visualized and compared using Jaccard index, which measures the overlap of the important features identified by both methods.

## Files in this Repository

- `SHAP_LIME_IMPL_PROJECT.ipynb`: Jupyter notebook that contains the full implementation of SHAP and LIME for skin cancer detection. It includes data loading, model training, and interpretability steps.
- `Skin_cancer_detection_with_SHAP_and_LIME.md`: Detailed report on the implementation, covering methodology, results, and comparison of SHAP and LIME.
- `efficientnet_advanced.h5`: The trained model saved after fine-tuning EfficientNetB0 on the HAM10000 dataset.
- `lime_explanations.png`: Example visualization of LIME explanations showing the important superpixels for skin cancer detection.
- `shap_explanations.png`: Example visualization of SHAP explanations showing the important superpixels for skin cancer detection.

## Setup and Requirements

To run the code, make sure to install the following dependencies:

```bash
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install scikit-image
```

## Running the Code

1. Clone the repository or download the notebook (`SHAP_LIME_IMPL_PROJECT.ipynb`).
2. Ensure the dataset is available or point to the correct directory containing the images. The dataset can be downloaded from Kaggle: [Skin Cancer MNIST: HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=HAM10000_images_part_1)
3. Open the Jupyter notebook and run the cells sequentially to train the model and generate SHAP and LIME explanations.
4. The results, including visualizations of the important superpixels for each method, will be displayed in the notebook.

## Explanation Visualizations

- **SHAP Explanations**: The SHAP method highlights the superpixels that are most influential in the model's decision, with a heatmap overlay.
- **LIME Explanations**: LIME identifies the superpixels that have the highest importance for a given prediction by fitting a local surrogate model to the CNN.

## Results

The **SHAP Faithfulness Test** measures the impact of removing the most important superpixels on the model's prediction. A significant drop in the prediction probability suggests that SHAP has provided a faithful explanation.

The **Jaccard index** between the SHAP and LIME masks helps quantify the agreement between the superpixels identified by both methods. A low Jaccard index (e.g., 0.03) may suggest that the two methods highlight different features in the image as important.

## Conclusion

This project demonstrates the explainability techniques like SHAP and LIME, providing transparency into deep learning models. By interpreting the model's decisions, we can better understand how the model detects skin cancer and ensure that it is making decisions based on relevant and meaningful features in the images.
