# 🧠 Banking Intent Classification (NLP)

Proyecto de NLP para clasificar la intención de consultas bancarias usando el dataset **BANKING77**.

---

## 📌 Descripción

Se entrenaron y compararon 5 modelos para clasificar textos en **77 intenciones diferentes** de clientes bancarios.

El objetivo es automatizar la detección de intención y mejorar el enrutamiento de consultas.

---

## 📊 Dataset

* Nombre: **BANKING77**
* 13,083 ejemplos
* 77 clases de intención
* Consultas cortas (~11 palabras)

Fuente: PolyAI 

---

## 🤖 Modelos utilizados

* Logistic Regression
* SVM (LinearSVC)
* Random Forest
* BiLSTM (con GloVe)
* DistilBERT (fine-tuned)

---

## 📈 Resultados

| Modelo         | Accuracy  |
| -------------- | --------- |
| Logistic Reg.  | 85.6%     |
| SVM            | 89.4%     |
| Random Forest  | 86.6%     |
| BiLSTM         | 89.9%     |
| **DistilBERT** | **93.2%** |

👉 DistilBERT fue el mejor modelo, superando a los demás en todas las métricas 

---

## 🧠 Conclusiones

* Los modelos tradicionales tienen buen desempeño pero se quedan cortos
* El contexto es clave en NLP
* Los transformers (BERT) dominan en tareas complejas

---

## 📂 Archivos

* `proyecto_final.ipynb` → notebook principal
* `proyecto_final.py` → versión en script
* Diagramas y presentación incluidos

---

## 🚀 Uso

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
jupyter notebook proyecto_final.ipynb
```

---

## 👤 Autor

Ethan Leonel

---
