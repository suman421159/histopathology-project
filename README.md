# Histopathologic Cancer Detection

## Project Details
- **Core Code**: Located in the `src` folder.
- **Results and Visualizations**: Found in the `results` folder.
- **Submission Output**: `submission.csv` located in the root directory.
- **Large Files**: The `models` and `data` folders are excluded due to size constraints. The model directory, in particular, contains multiple models generated from 5-fold cross-validation, contributing to its large size.

## Introduction
This project uses deep learning to detect metastatic cancer in lymph node sections. By leveraging a pre-trained model, it processes image patches to identify early metastasis sites, enhancing cancer diagnostics' accuracy and reliability.

## Exploratory Data Analysis
Visualizations focus on the distribution of labels and differences between metastatic and non-metastatic samples. These analyses set the groundwork for the model training, ensuring well-prepared input data.

## Model Architecture
The VGG16 architecture was chosen for its robustness and capability in complex medical image analysis, proving effective for feature extraction and cancer detection.

## Training and Results
- **Training Methodology**: Stratified K-Fold cross-validation was used to ensure thorough evaluation.
- **Performance Metrics**: The model achieved an ROC AUC of 0.93.
- **Kaggle Performance**: Scored 0.9030 on Kaggle, demonstrating strong generalization.

## Conclusion
The project highlights the potential of using deep learning for enhancing diagnostic accuracy in medical imaging. It provided significant insights into model training and validation, demonstrating robust performance across metrics.

## References
1. [Kaggle: Histopathologic Cancer Detection](#)
2. [Fashion-MNIST with tf.keras](#)
3. [Boost Your Image Classification Model with pretrained VGG-16](#)
4. [How to Use ROC Curves and Precision-Recall Curves for Classification in Python](#)
