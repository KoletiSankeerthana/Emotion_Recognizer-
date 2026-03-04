# Emotion Intelligence System

An AI-powered web application that analyzes text to detect **emotions and potential hate speech** using **Transformer-based Natural Language Processing models**.

The application predicts emotional tone, identifies sentiment, and detects harmful language using an interactive **Streamlit web interface**.

---

## Live Demo

Try the application here:

**Live App:**  
https://ioahfzgzr2bard8nngyxhh.streamlit.app/

---

## Project Overview

Human communication often contains emotional signals. This project analyzes text input and identifies emotional states using NLP techniques.

The system allows users to:

- Detect the **primary emotion** expressed in text
- Identify **overall sentiment**
- View **emotion probability distribution**
- Calculate **emotional metrics**
- Detect **hate speech or toxic language**
- Receive **emotion-based insights**

---

## Features

### Emotion Detection

The system predicts emotional states using a transformer-based NLP model.

Supported emotions include:

- Joy  
- Sadness  
- Anger  
- Fear  
- Surprise  
- Disgust  
- Neutral  

The model returns the **dominant emotion and probability scores**.

---

### Sentiment Analysis

The system also identifies the **overall sentiment** of the text.

Sentiment categories include:

- Positive
- Negative
- Neutral

---

### Emotional Metrics

Additional metrics help interpret the emotional strength of the text.

**Confidence Score**  
Represents the model's confidence in the predicted emotion.

**Dominance Score**

Dominance = Top Emotion Probability − Second Emotion Probability

**Balance Index**

Balance = 1 − Dominance

These metrics help understand whether the emotion is **strong or mixed**.

---

### Emotion Visualization

Emotion probabilities are displayed using charts to show the **distribution of emotions in the text**.

This helps users easily understand which emotions are present.

---

### Hate Speech Detection

The system analyzes text for **potentially harmful language**.

Detected categories include:

- Neutral
- Aggressive language
- Targeted harassment
- Hate speech

The output includes:

- Top category
- Probability score
- Risk interpretation

---

### Emotion-Based Suggestions

The application generates helpful insights based on detected emotions.

Example:

If frustration or anger is detected, the system may suggest taking a pause before reacting.

---

## Technologies Used

**Programming Language**

- Python

**Machine Learning / NLP**

- Hugging Face Transformers  
- PyTorch  

**Data Processing**

- NumPy  
- Pandas  

**Visualization**

- Plotly  
- Matplotlib  

**Web Application**

- Streamlit  

---

## Project Structure

```
Emotion_Recognizer
│
├── .gitignore           # Git ignored files
├── streamlit_app.py     # Main Streamlit application
├── emotion_model.py     # Emotion detection model logic
├── hate_model.py        # Hate speech detection model
├── utils.py             # Helper functions
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## Installation

Clone the repository:

```
git clone https://github.com/KoletiSankeerthana/Emotion_Recognizer.git
```

Navigate to the project folder:

```
cd Emotion_Recognizer
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit app:

```
streamlit run streamlit_app.py
```

The application will open in your browser.

---

