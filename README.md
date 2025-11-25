# 🌽 Corn Leaf Disease Detection  
Deep learning–powered identification and interpretation of major corn leaf diseases using Transfer Learning, Custom CNNs, and LIME.

---

## 📌 Project Introduction  
A deep learning system that classifies corn leaf diseases using a hybrid approach of transfer learning and custom CNNs, enhanced through image augmentation and LIME-based interpretability to help understand *why* the model predicts a disease. Designed to improve crop health monitoring with accurate and explainable predictions.

---

## 🧰 Tech Stack  
- **Language:** Python  
- **Frameworks & Libraries:** PyTorch, Torchvision, NumPy, Matplotlib  
- **Modeling Approaches:** Custom CNN, Transfer Learning (ResNet18), LIME  
- **Tools:** GPU acceleration, Data Augmentation pipelines  

---

## 🚀 Proposed Approach  
This project integrates three complementary components:

### 1️⃣ Transfer Learning (ResNet18)  
A pretrained ResNet18 acts as the feature extractor. Its final classification layer is fine-tuned to detect **Blight, Common Rust, Gray Leaf Spot, and Healthy** classes.  
- Leverages ImageNet-trained representations  
- Reduces training cost  
- Improves generalization with limited agricultural data  

### 2️⃣ Custom Convolutional Neural Network  
A lightweight CNN is also implemented to:  
- Experiment with architecture variations  
- Compare training convergence and accuracy  
- Demonstrate end-to-end feature learning from scratch  

The CNN uses stacked convolutional layers, ReLU activations, max pooling, and fully connected decision layers—capturing leaf texture, vein patterns, and disease spots.

### 3️⃣ LIME-Based Interpretability  
LIME provides pixel-level insight into the model's decision-making:  
- Generates perturbed variations of the input image  
- Identifies superpixels contributing to the prediction  
- Highlights disease-affected regions  
- Helps validate model behavior against agricultural domain knowledge  

This makes the solution **explainable**, bridging the gap between deep learning outputs and actionable insights for farmers.

---

## 🧹 Data Preprocessing Pipeline  
The project uses a well-structured preprocessing setup designed for robustness and better generalization.

### 🔧 Training Transformations  
- Resize to **256×256**  
- Random flips (horizontal + vertical) to simulate varied leaf orientations  
- Gaussian blur for noise and lighting variation  
- Convert to tensors  
- Normalize using ImageNet means & std  

These ensure the model is more resilient to real-world field conditions.

### 🔧 Validation & Test Transformations  
- Resize → Tensor → Normalize  
- No augmentations (ensures unbiased evaluation) 

---

## 🧠 Feature Extraction – What the Model Learns  
Both the custom CNN and ResNet18 extract structured patterns from leaf images:

- **Edges & contours:** early layers  
- **Texture patches:** mid-level layers  
- **Disease-specific marks (spots, rust textures, lesions):** deeper layers  
- **Global leaf structure:** fully connected layers  

This hierarchical representation enables robust classification even for subtle disease patterns.

---

## 🏋️ Training, Validation & Testing  
- Custom CNN trained with **Stochastic Gradient Descent** + Cross-Entropy Loss  
- Transfer learning model fine-tuned using same pipeline  
- Evaluation performed on a held-out test set  
- Metrics captured:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

## 📊 Results  
### ✔ Accuracy & Loss  
Both models show stable convergence with decreasing loss curves and high validation accuracy.

### ✔ Class-wise Metrics  
- **Blight:** High precision, recall, F1  
- **Common Rust:** High precision, recall, F1  
- **Green Leaf Spot:** Strong performance with minor confusion  
- **Healthy:** Near-perfect classification  

### ✔ LIME Visualizations  
Disease-affected regions are clearly marked with highlighted superpixels, confirming that predictions rely on biologically relevant features.

---

## 🔍 Why LIME Matters in Agriculture  
LIME builds trust by showing:  
- Which areas of the leaf triggered the prediction  
- Whether the model focuses on disease spots instead of background noise  
- How confidently the model makes distinctions between visually similar diseases  

This interpretability is crucial for deploying AI in real-world agricultural decision systems.

---

## 🧭 Project Flow Overview  
1. **Raw Dataset → Preprocessed Images**  
2. **Train/Val/Test Split Applied**  
3. **Training on Custom CNN + Fine-tuning ResNet18**  
4. **Evaluation on Unseen Data**  
5. **LIME Applied for Interpretability**  
6. **Predictions + Visual Explanations Delivered**  

The workflow combines accuracy, efficiency, and transparency—making it ideal for scalable agricultural disease detection.

---

## 🏁 Conclusion  
This project demonstrates a powerful and interpretable approach to corn leaf disease classification using modern deep learning:  
- Strong accuracy through transfer learning & refined CNNs  
- Increased robustness using targeted augmentations  
- Explainable decisions using LIME  
- Practical applicability for supporting farmers with AI-driven insights  

It showcases how deep learning and interpretability can work together to solve real agricultural challenges.  

---
