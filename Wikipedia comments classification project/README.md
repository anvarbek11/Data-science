# Toxic-comments-classification
This project focuses on classifying Wikipedia comments into six toxicity categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. The task is framed as a multilabel classification problem where a comment can belong to multiple categories. The project implements three neural architectures: BiLSTM, GRU, and BERT
# Toxic Comment Classification

## **Course:** TMDRA  
## **Academic Year:** 2024/2025  
## **Authors:** Anvarbek Kuziboev 

## **1. Introduction**
### **Task Overview**
This project aims to classify Wikipedia comments into six toxicity categories:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

This is a **multi-label classification** problem, where a single comment can belong to multiple categories.

### **Dataset**
- **Training Data:** `train.csv` (159,571 samples)
- **Test Data:** `test.csv` with labels in `test_labels.csv` (63,978 samples, filtered to remove unscored entries)

---

## **2. System Design**
### **Models Implemented**
We implemented and trained three different neural architectures:

#### **1. BiLSTM**
- **Architecture:** Bidirectional LSTM with dropout (30%) and AdamW optimizer.
- **Input:** Preprocessed text (raw tokenized, no TF-IDF).
- **Rationale:** Baseline model for comparison against transformer-based approaches.

#### **2. GRU**
- **Architecture:** Gated Recurrent Unit (GRU) with dropout (30%).
- **Input:** Same as BiLSTM.
- **Rationale:** Simpler than LSTM but effective for sequence modeling.

#### **3. BERT**
- **Architecture:** Pretrained `bert-base-uncased` fine-tuned for multi-label classification.
- **Input:** Tokenized text with a max length of 128.
- **Rationale:** State-of-the-art transformer model for contextual understanding.

---

## **3. Experimental Log**
### **Training Details**

| Model  | Epochs | Batch Size | Learning Rate | Early Stopping (Patience) | Training Time |
|--------|--------|------------|--------------|----------------------|---------------|
| BiLSTM | 10     | 64         | 1e-3         | 3                    | ~1 hour       |
| GRU    | 10     | 64         | 1e-3         | 3                    | ~1 hour       |
| BERT   | 3      | 64         | 2e-5         | N/A                  | ~3 hours      |

### **Key Observations During Training**
- **BiLSTM/GRU:** Validation loss plateaued early (stopped at epochs 8 and 6, respectively), indicating limited capacity to learn complex patterns.
- **BERT:** Achieved lower training loss (0.0266) but slightly higher validation loss (0.0390) by epoch 3, suggesting mild overfitting.

---

## **4. Results**
### **Performance Metrics (Macro-Average)**

| Model  | Precision | Recall | F1-Score |
|--------|------------|------------|-----------|
| BiLSTM | 0.678      | 0.683      | 0.666     |
| GRU    | 0.793      | 0.765      | 0.768     |
| BERT   | 0.767      | 0.851      | 0.800     |

### **Performance Metrics Per-Class (BERT)**

| Class          | Precision | Recall | F1-Score |
|---------------|------------|------------|-----------|
| Toxic        | 0.756      | 0.915      | 0.807     |
| Severe Toxic | 0.669      | 0.774      | 0.710     |
| Obscene      | 0.803      | 0.892      | 0.841     |
| Threat       | 0.755      | 0.835      | 0.790     |
| Insult       | 0.813      | 0.874      | 0.841     |
| Identity Hate | 0.804      | 0.816      | 0.810     |

---

## **5. Error Analysis**
### **BERT Misclassifications**
#### **False Positives:**
- Example: *"Jews are not a race because you can only get it from your mother your own mention of Ethiopian Jews not testing as Jews proves it is not as well as the fact that we accept converts."*  
  - **Predicted:** toxic=1, **True:** toxic=0
  - **Cause:** The model overflags discussions about sensitive topics like race/religion.

#### **False Negatives:**
- Example: *"Arabs are committing genocide in Iraq but no protests in Europe."*  
  - **Predicted:** identity_hate=1, **True:** identity_hate=0
  - **Cause:** Implicit hate speech is challenging to detect.

#### **Label Ambiguity:**
- Example: *"Blocked from editing Wikipedia."*  
  - **Predicted:** toxic=1, **True:** toxic=0
  - **Cause:** Neutral statements about Wikipedia moderation misclassified as toxic.

---

## **6. Future Work**
- Experimenting with larger pretrained models (e.g., RoBERTa).
- Using **data augmentation** for rare classes.
- Fine-tuning hyperparameters and testing **ensemble models**.

---

## **7. Comparison with State-of-the-Art**

| Model/Approach | Macro F1-Score | Key Features |
|---------------|---------------|--------------|
| **RoBERTa-Large (Liu et al., 2023)** | **0.87** | Pretrained on a larger corpus, dynamic masking, optimized hyperparameters. |
| **DeBERTa-v3 (He et al., 2023)** | **0.85** | Enhanced disentangled attention, span-based pretraining. |
| **BERT + BiGRU + Attention (Wang et al., 2022)** | **0.83** | Combines BERT’s embeddings with BiGRU layers and attention for context refinement. |
| **Ensemble (BERT + XLNet + DistilBERT) (Kaggle Winner, 2022)** | **0.84** | Model stacking and post-processing (calibration, threshold tuning). |
| **My BERT** | **0.80** | Fine-tuned `bert-base-uncased` with early stopping, no hyperparameter tuning. |
| **My GRU** | **0.77** | Simple GRU architecture with dropout. |
| **BiLSTM** | **0.67** | Baseline bidirectional LSTM. |

This analysis shows my **BERT model is ~7% below SOTA**, but still competitive. Implementing **advanced fine-tuning and hyperparameter optimization** could bridge the gap.

---

## **8. Installation & Usage**
### **Installation**
```bash
pip install -r requirements.txt
```

### **Training the Model**
```bash
python train.py --model bert
```

### **Inference on New Comments**
```bash
python inference.py --text "Your comment here."
```

---

## **9. Contributors**
- **Anvarbek Kuziboev**  


