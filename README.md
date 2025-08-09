🚨 Mental Crisis Detection App
A Streamlit-based web app that uses a fine-tuned Hugging Face Transformer model to detect potential mental health crisis risk levels from text input.
This tool can classify user messages into Low Risk, Neutral, or High Risk, with probability breakdowns and visual charts.

📌 Live Demo
https://huggingface.co/spaces/manaswini1216/mental-crisis
Mental Crisis Detection App — Project Overview
The Mental Crisis Detection App is a user-friendly, web-based tool built using Streamlit that detects potential mental health crisis risk levels in free-text input. This app classifies messages into three risk categories — Low Risk, Neutral, or High Risk — leveraging a fine-tuned Transformer model from Hugging Face’s 🤗 ecosystem.

How I Built This App
1. Data Preparation & Dataset
I curated and prepared a balanced dataset consisting of user text samples labeled as Low Risk, Neutral, and High Risk.
For the crisis data, I used labeled samples from the GoEmotions dataset focusing on nine specific emotions relevant to mental distress, such as sadness, nervousness, anger, fear, remorse, disappointment, embarrassment, grief, and confusion.
The neutral and low-risk classes contained normal conversational text or emotionally neutral data.
The dataset was cleaned, balanced, and preprocessed for text classification.

2. Model Selection & Fine-Tuning
I selected a pretrained transformer model (e.g., BERT or DistilBERT) from the Hugging Face Model Hub suitable for text classification.
The model was fine-tuned on the curated dataset for 3-class classification (Low Risk, Neutral, High Risk).
Training involved typical NLP steps: tokenization with the Hugging Face AutoTokenizer, attention masks, and cross-entropy loss function.
Fine-tuning was done on GPU-enabled hardware to speed up training, but inference runs efficiently on CPU.

3. Model Integration in Streamlit
The fine-tuned model and tokenizer were saved and loaded in the Streamlit app for inference.
The app takes user input text and tokenizes it, then feeds it into the model to get predicted class probabilities.
Instead of showing raw label codes (e.g., LABEL_0), the app maps predictions to meaningful risk labels: Low Risk, Neutral, or High Risk.
The model output includes the risk classification and a probability breakdown for all three classes.

4. Crisis Keyword Override Logic
To increase safety and capture urgent cases, I implemented a crisis keyword override feature.
If the user input contains predefined keywords strongly associated with crisis or emergency (e.g., "suicide," "harm," "help me"), the app forces the output to High Risk regardless of model confidence.
This helps catch critical cases that might have been misclassified or have low confidence in the prediction.

5. Interactive Probability Visualization
The app displays an interactive bar chart showing the model’s confidence scores for each risk category.
This visual breakdown allows users or clinicians to understand how confident the model is about the prediction.
The chart dynamically updates based on input, using Streamlit’s built-in charting functions.

6. Lightweight & Accessible UI
I designed the UI with simplicity and accessibility in mind.
Using Streamlit’s easy layout components, the app supports text input, displays prediction results clearly, and presents probability visuals without clutter.
The app is lightweight and runs entirely on CPU, making it suitable for deployment on common hosting platforms without GPU access.

7. Deployment
The app is deployed on Hugging Face Spaces as a Streamlit application.
This allows free, public access with easy sharing via a live demo link.
The hosted app demonstrates real-time mental crisis risk detection from user text input.

Summary
This project combines state-of-the-art NLP techniques, ethical considerations, and practical deployment to create a helpful mental health tool. The mental crisis detection app:
Leverages Hugging Face Transformers fine-tuned on emotional crisis datasets
Implements safety overrides for urgent risk cases

Offers clear, interpretable risk categories with probability scores

Runs smoothly on CPU with an intuitive Streamlit interface

Is publicly accessible via an interactive web demo
