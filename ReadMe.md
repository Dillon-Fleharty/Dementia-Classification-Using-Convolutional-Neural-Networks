
# Alzheimer's Detection Using OASIS
**Dillon Fleharty**

## Project Summary

Alzheimer's disease (AD), a form of dementia, is an advanced brain disorder that can cause damage to memory and promote
tissue loss in the brain. With no current cure, early detection is critical for slowing the disease and improving
the quality of life for individuals affected. For my research, I intend to build a convolutional neural network
that detects Alzheimer's among four different ordinal classes ranging in severity from none to moderate.
The project aims to develop a model that performs better than currently available tools. Further,
I would like to evaluate the performance between ordinal and categorical classifications by keeping ordinal
information in place of one-hot encoding. I will also experiment with handling class imbalance and
evaluate each approach.

## Problem Statement

The use of Convolutional Neural Networks (CNNs) for detecting Alzheimer's through MRI scans is a significant
breakthrough in medical imaging and diagnostics. CNNs, a class of deep learning algorithms are incredibly efficient
in processing visual imagery, making them particularly suited for analyzing complex patterns in
MRI scans. By automatically detecting and learning features such as brain volume, cortical thickness, and the
presence of amyloid plaques—hallmarks of Alzheimer's disease—CNNs can identify subtle brain changes with high
precision. This capability not only streamlines the diagnostic process but also enhances the early detection
of Alzheimer's, facilitating earlier intervention and management strategies. Consequently, the use of CNNs in
analyzing MRI scans represents a powerful tool in the fight against Alzheimer's, offering hope for improved
outcomes through advanced, data-driven diagnostics.

## Dataset

Derived from the Open Access Series of Imaging Studies (OASIS), this dataset encompasses 80,000 MRI brain
scans across a diverse age group of 416 subjects, ranging from 18 to 96 years old. Each subject contributed
3 to 4 T1-weighted MRI scans within a single session, categorizing the dataset into four distinct dementia
stages: non-demented, very mild demented, mild demented, and moderate demented. The distribution of scans
includes 63,220 for non-demented, 13,700 for very mild demented, 5002 for mild demented, and 488 for moderate
demented cases. Despite the comprehensive nature of the dataset, it presents a substantial challenge due to its
imbalanced distribution among the different stages of dementia, highlighting the need for careful consideration
in research and analysis methodologies.


## Exploratory Data Analysis

The exploratory data analysis focused on examining the pixel intensity distribution across different classes of MRI scans to observe potential correlations with dementia stages. Key statistics, such as the mean pixel value and standard deviation, were computed for each class: non-demented, very mild demented, mild demented, and moderate demented.

### Key Findings:

Mean Pixel Values: The analysis revealed that non-demented images typically have higher mean pixel values, suggesting greater brain matter presence compared to more severe dementia stages. This could indicate less tissue loss and fewer amyloid plaque formations in non-demented subjects.
Standard Deviation: The standard deviation in pixel values provides insights into the variability of brain tissue density across different scans. A lower standard deviation in more advanced dementia stages could suggest more uniform tissue degradation.

### Visual Analysis:
Image Overlay: To visually compare the differences in brain structure between the non-demented and moderate demented stages, the mean pixel values from each class were overlaid to create composite images. These composite images starkly illustrate the contrast in brain volume and density, with non-demented images displaying more extensive and dense brain matter. This visual method helps highlight the potential areas of the brain most affected by the progression of Alzheimer's.
The EDA not only provides a quantitative foundation for the subsequent model development but also offers a visual reinforcement of the pathological impacts of Alzheimer's on the brain, as observed through MRI scans.


## Data Preprocessing

### 1. Image Loading and Preprocessing
Image Loading: Load the MRI brain scan images from the dataset directory.
Image Resizing: Resize the images if needed, I used various image sizing, (124x124), (224x224), and the original image size (498x248) to a consistent size suitable for input to the model. 


### 2. Data Balancing (Optional based on which model was used) 
Addressing Class Imbalance: Since there is severe data imbalance, for model 1 I reduced the size of the image dataset for non-demented and very mild demented. I didn't want to resample any images for fear of overfitting. For Model 3, binary classification, because I combine all demented images I have a larger dataset for demented images and chose to use 7000 images from each non-demented and demented. 

### 3. Train-Validation-Test Split
Splitting the Data: Split the dataset into training, validation, and test sets. Validation was 20%. 

### 4. Encoding Labels
One-Hot Encoding: Encode the categorical labels (dementia stages) into one-hot vectors. This is necessary for multi-class classification tasks and ensures that the model can interpret the categorical labels correctly during training and evaluation.


## Machine Learning Approaches

### Model 1: Multi-Class Classification CNN
Implemented a Convolutional Neural Network (CNN) for multi-class classification.
Utilized a sequential model architecture consisting of convolutional, batch normalization, max pooling, dropout, and dense layers.
Trained on the preprocessed MRI brain scan dataset, categorized into four dementia stages: non-demented, very mild demented, mild demented, and moderate demented.
Applied one-hot encoding to the class labels and performed a train-test split to evaluate model performance.

### Model 2: Pretrained Model with EfficientNetB0
Employed a pretrained EfficientNetB0 model for feature extraction and transfer learning.
Fine-tuned the model on the MRI brain scan dataset to adapt it to the task of dementia classification.
Leveraged EfficientNetB0's prelearned representations to improve generalization and efficiency.

### Model 3: Binary Classification CNN
Designed a CNN for binary classification to detect dementia presence or absence.
Adopted a similar architecture to Model 1 but with a different output layer for binary classification.
Trained on a subset of the MRI brain scan dataset, categorizing subjects into dementia-positive and dementia-negative classes.

## Experiments

### Hyperparameter Tuning:
Employed extensive hyperparameter tuning to optimize model performance.
Varied learning rates, including adjusting the learning rate schedule using linear decay.
Utilized Adam optimizer with various learning rates to find the optimal balance between convergence speed and model stability.
Explored the impact of L1 and L2 regularization on model generalization and overfitting.
Investigated multiple kernel sizes to match the aspect ratio of the input images, enhancing feature extraction capabilities.
### Model Comparison:
Compared the performance of different models with varying architectures and training approaches.
Evaluated the effectiveness of multi-class classification versus binary classification for dementia detection.
Analyzed the benefits of transfer learning using a pretrained model (EfficientNetB0) versus training from scratch.

### Handling Class Imbalance:
Explored strategies to address class imbalance, particularly for multi-class classification tasks.
Experimented with data balancing techniques, including downsampling the majority class and adjusting class weights during training.
Evaluated the impact of class distribution on model performance and generalization.

### Evaluation Metrics:
Utilized standard evaluation metrics such as accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC) to assess model performance.
Conducted cross-validation and examined performance on both training and validation datasets to ensure robustness and generalization.

### Optimized Architecture: 
The CNN architecture, comprising convolutional layers with kernel sizes matching the aspect ratio of input images, demonstrated superior feature extraction capabilities, capturing subtle patterns indicative of dementia progression.
### Regularization Techniques: 
Incorporating L2 regularization into the convolutional layers helped mitigate overfitting and improve model generalization. Additionally, dropout layers were instrumental in preventing co-adaptation of neurons and enhancing model robustness.
### Learning Rate Schedule: 
Employing linear decay for the learning rate schedule facilitated steady convergence during training, preventing abrupt changes and ensuring smoother optimization.
### Model Evaluation: 
The model's performance was evaluated using standard evaluation metrics such as accuracy and loss, alongside techniques such as early stopping to prevent overfitting and restore the best-performing weights.

### Detailed Analysis of Binary Classification Model (Most Accurate Model)

**Model Overview:**
This binary classification model was used for analysis of this project.

**Model Specifications and Configurations:**

1. **Preprocessing and Data Management:**
   - The model employs binary classification, grouping all forms of dementia into a single 'demented' category and 
   contrasting it with 'non-demented'.
   - Original image dimensions (498x248) were maintained to preserve the integrity of the image data, 
   - avoiding potential information loss from downsizing. This approach proved crucial as it allowed the model to utilize detailed textural and structural information from the MRI scans, which are significant for accurate dementia detection.

2. **Model Architecture:**
   - The network consists of five convolutional layers, each followed by batch normalization and max-pooling. 
   The filter sizes are designed proportionally to the images' aspect ratio (5x2), enhancing the model's ability to 
   capture relevant features aligned with the natural structure of the brain images.
   - Regularization techniques such as dropout and L2 regularization were applied to prevent overfitting.
   Specifically, dropout rates were set at 0.5 post certain convolutional layers, and L2 regularization was applied 
   to the final convolutional and dense layers.
   - An Adam optimizer with a learning rate of 0.0001, using a linear decay learning rate scheduler, was utilized to
   ensure stable convergence over training epochs.

3. **Performance Optimization:**
   - Early stopping was configured to monitor accuracy with a patience of 5 epochs, 
   ensuring training cessation when improvements became negligible, thus saving computational resources and preventing overfitting.
   - The GPU memory management setup was adjusted to allow TensorFlow to only allocate as much GPU memory as
   needed, preventing crashes due to memory overflows.

4. **Kernel Aspect Ratio Justification:**
   - The choice to use kernels with an aspect ratio of 5x2 is justified by the original dimensions of the images. 
   This decision was crucial for capturing the elongated features of the brain structures in the MRI scans,
   which are more effectively highlighted with horizontally oriented kernels compared to square kernels.

5. **Evaluation and Validation:**
   - The model achieved robust performance metrics, demonstrated through various evaluations including precision, 
   recall, F1-score, and a confusion matrix analysis. K-fold cross-validation further validated the model's 
   consistency and generalization across different subsets of data.

# Next Steps

To further enhance the robustness and applicability of the binary classification model, the following steps are what I 
plan to do:

1. **Testing on External Image Sets:**
   - Currently, the model has been trained and validated exclusively on images from the OASIS dataset. This would 
   ensure that the model generalizes well across different imaging conditions and patient demographics, 
   it is crucial to test it on external image sets from other studies or institutions. This will help
   identify any biases or limitations imposed by the OASIS dataset and refine the model's ability to 
   real-world scenarios.

2. **Expansion to Multiclass Classification:**
   - While binary classification provides a foundational understanding and detection capability for 
   dementia, expanding the model to classify various stages of dementia (e.g., very mild, mild, moderate)
   could offer more nuanced insights and diagnostic utility. This step would involve adjusting the model 
   architecture to handle multiclass outputs and revisiting the preprocessing steps to accommodate more detailed categorizations.

4. **Continuous Model Improvement:**
   - Continuous monitoring of the model's performance should be established, 
   with mechanisms to update and retrain the model as more data becomes available or as patient characteristics 
   and imaging technologies evolve. This ongoing improvement will help maintain the model's accuracy and relevance over time.

### Conclusion:
After conducting extensive experimentation and analysis, I found that for the architectures I built the most effective
strategy was using binary classification. This is likely due to available data as in the classification model I have a 
very imbalanced dataset. The most impactful insight I found was using a kernal size that was proportional to the 
image size. So instead of resizing the image, and risking lose of information in the interpolation 
process adjusting the kernal size to match the image gave the best results. This along with lots of trail and 
error with other paramters yeilded a result I was happy with.

The model's architecture and specific configurations, including the maintenance of original image sizes and 
the strategic use of kernel aspect ratios, were key in achieving high accuracy and reliability in dementia 
detection. This approach could serve as a benchmark for future studies in medical image analysis, emphasizing the 
importance of tailored preprocessing and model configuration to the data characteristics.

Overall, the research underscores the efficacy of CNNs in diagnosing Alzheimer's disease from MRI brain 
scans and highlights the importance of architectural design, regularization techniques, and hyperparameter
optimization in achieving optimal performance. The findings contribute to the ongoing efforts in leveraging deep 
learning for early detection and management of Alzheimer's disease, offering hope for improved diagnostic accuracy and patient outcomes.







