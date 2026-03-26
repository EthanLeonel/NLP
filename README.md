# 🧠 Banking Intent Classification (NLP)

NLP project for classifying banking customer queries using the **BANKING77** dataset.

---

## 📌 Description

Five models were trained and compared to classify text into **77 different banking intents**.

The goal is to automate intent detection and improve query routing.

---

## 📊 Dataset

* Name: **BANKING77**
* 13,083 samples
* 77 intent classes
* Short queries (~11 words)

Source: PolyAI

---

## 🤖 Models Used

* Logistic Regression
* SVM (LinearSVC)
* Random Forest
* BiLSTM (with GloVe)
* DistilBERT (fine-tuned)

---

## 📈 Results

| Model          | Accuracy  |
| -------------- | --------- |
| Logistic Reg.  | 85.6%     |
| SVM            | 89.4%     |
| Random Forest  | 86.6%     |
| BiLSTM         | 89.9%     |
| **DistilBERT** | **93.2%** |

👉 DistilBERT achieved the best performance across all models.

---

## 🧠 Conclusions

* Traditional models perform well but have limitations
* Context is key in NLP
* Transformer models (BERT) outperform other approaches

---

## 📂 Files

* `proyecto_final.ipynb` → main notebook
* `proyecto_final.py` → script version
* Diagrams and presentation included

---

## 🚀 Usage

```bash
pip install -r requirements.txt
```

Run:

```bash
jupyter notebook proyecto_final.ipynb
```

---

## 👤 Author

Ethan Leonel

---
