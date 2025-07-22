# **AI-Based Early Detection of Alzheimer’s from Patient Speech**



## 🧠 Project Overview

This project implements an AI-powered system for the **early detection of Alzheimer’s Disease** based on analysis of a patient's spontaneous speech. By evaluating both linguistic and acoustic patterns, the tool aims to enable non-invasive, scalable, and accessible Alzheimer's screening, particularly in low-resource or multilingual settings.

---

## ✨ Features

* 🎹 Input: Raw or transcribed spontaneous speech samples
* 🧬 Extracts linguistic (syntax, semantics) and acoustic (pitch, pause, energy) features
* 🤖 Applies NLP + machine learning models for Alzheimer’s prediction
* 📈 Visualization of key features for interpretability
* 🌍 Multilingual-ready architecture for wider reach

---

## 🛠️ Tech Stack

| Component         | Technology                                            |
| ----------------- | ----------------------------------------------------- |
| NLP               | SpaCy, NLTK                                           |
| Acoustic Analysis | Librosa, Praat-parselmouth                            |
| ML Models         | Scikit-learn, XGBoost, SVM                            |
| Interface         | Streamlit / Flask                                     |
| Datasets          | Pitt Corpus (DementiaBank), ADReSS Challenge datasets |

---

## 🚀 Installation and Setup

### Prerequisites

* Python 3.9+
* Access to DementiaBank / ADReSS datasets

### Steps

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/alzheimers-speech-ai.git
cd alzheimers-speech-ai
```

2. **Install Required Packages**

```bash
pip install -r requirements.txt
```

3. **Run the Web Interface**

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
alzheimers-speech-ai/
├── app.py                    # Streamlit interface
├── data/                     # Raw audio + transcripts
├── features/
│   ├── extract_acoustic.py
│   └── extract_linguistic.py
├── models/
│   ├── train_model.py
│   └── predict.py
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

* `spacy`, `nltk`, `librosa`, `parselmouth`
* `scikit-learn`, `xgboost`, `matplotlib`, `pandas`
* `streamlit` or `flask`

---

## 🔧 Customization

* Change feature selection criteria in `features/`
* Tune models in `models/train_model.py`
* Replace or augment datasets in `data/`

---

## 🐞 Known Issues

* Requires dataset license (e.g., DementiaBank access)
* Audio clarity can affect acoustic features
* Not a clinical diagnostic tool (for screening support only)

---

## 🚧 Future Improvements

* Add real-time voice recording input
* Expand multilingual support
* Integrate with telemedicine platforms
* Add longitudinal tracking for progression

---

## 📄 License

Licensed under the **Apache 2.0 License**.

---

## 🙌 Acknowledgments

* [DementiaBank](https://dementia.talkbank.org/) for data
* [ADReSS Challenge](https://www.aclweb.org/anthology/2020.interspeech-1.302/) for benchmark standards
* Research contributors in computational neurolinguistics
* Open-source libraries for enabling accessible healthcare innovation
