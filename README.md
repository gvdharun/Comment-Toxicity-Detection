# 🚨 Deep Learning for Comment Toxicity Detection with Streamlit 🚨

Welcome to the **Comment Toxicity Detection** project — a deep learning-powered system to automatically identify toxic comments in online discussions, fostering safer and healthier communities.

---

## 🚀 Project Overview

Online platforms struggle with toxic comments such as harassment, hate speech, and offensive language. This project develops a **deep learning model** to detect and flag such comments in real-time.

- Built with **Python**, **TensorFlow**, **NLP**, and **Streamlit**
- Supports multiple toxicity categories:  
  - Toxic  
  - Severe Toxic  
  - Obscene  
  - Threat  
  - Insult  
  - Identity Hate  

---

## 🛠️ Features

- 🔍 Real-time toxicity prediction in an interactive web app  
- 📁 Bulk comment toxicity analysis via CSV upload  
- 📊 Visual display of data insights & model performance  
- 🤖 Deep learning models using CNN, LSTM, and BERT architectures  
- ⚙️ Easy deployment with detailed setup guide

---

## 📂 Project Structure

```
├── models/ # Saved trained models
├── data/ # Dataset files and samples
├── toxic_app.py # Streamlit application script
├── comment_toxicity.ipynb # jupyter notebook (Preprocessing, Model training)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```


---

## 📈 Model Performance Summary

| Model      | Training Accuracy | Validation Accuracy | Remarks                               |
|------------|-------------------|---------------------|-------------------------------------|
| CNN        | ~99.4%            | ~99.4%              | Fast, good baseline                  |
| LSTM       | (TBD)             | (TBD)               | Better context modeling              |
| BERT       | >95% (typical)    | >95% (typical)      | State-of-the-art contextual model   |

---

## 🖥️ Getting Started

1. Clone this repository  
`git clone https://github.com/yourusername/comment-toxicity-detection.git`


2. Install dependencies  
`pip install -r requirements.txt`


3. Run the Streamlit app  
`streamlit run toxic_app.py`

---

## Usage 🎯

- Input single comments via the Streamlit text box for toxicity prediction.
- Upload CSV files with a `comment_text` column for batch predictions.
- View model metrics and sample test data insights on the dashboard.
- Download prediction results as CSV.

---

## Project Deliverables 📦

- Deep learning model files for toxicity detection
- Complete Streamlit application source code
- Deployment guide with instructions
- Supplementary documentation and video demonstration

---

## Conclusion 🎉

This project delivers a **robust and scalable solution** for automating the detection of toxicity in user-generated comments. By leveraging advanced deep learning architectures and an intuitive Streamlit interface, it empowers social media platforms, forums, educational sites, and media outlets to **maintain healthier online environments** through efficient content moderation.

The modular design facilitates easy extension, enabling future integration of state-of-the-art transformer models and continuous improvements in accuracy and usability.

---
