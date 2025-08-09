import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    id2label = {
        0: "Low Risk",
        1: "Neutral",
        2: "High Risk"
    }

    return tokenizer, model, id2label

st.title("🚨 Mental Crisis Detection")

tokenizer, model, id2label = load_model()

text_input = st.text_area("Enter text for classification:", height=150)

if st.button("Detect Risk Level"):
    if text_input.strip():
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        prob_dict = {id2label[i]: round(p * 100, 2) for i, p in enumerate(probs)}

        high_risk_keywords = [
            "i want to die", "kill myself", "suicide", 
            "end my life", "can't go on", "no reason to live"
        ]
        if any(phrase in text_input.lower() for phrase in high_risk_keywords):
            risk_level = "High Risk"
        else:
            if prob_dict.get("High Risk", 0) >= 40:
                risk_level = "High Risk"
            elif prob_dict.get("Neutral", 0) >= 40:
                risk_level = "Neutral"
            else:
                risk_level = "Low Risk"

        st.subheader(f"**Predicted Risk:** {risk_level}")
        st.write("### Probability Breakdown")
        st.bar_chart(prob_dict)  
        st.json(prob_dict)

        if 35 <= prob_dict.get("High Risk", 0) < 40:
            st.warning("⚠ Borderline case — consider manual review.")

    else:
        st.warning("Please enter some text.")
