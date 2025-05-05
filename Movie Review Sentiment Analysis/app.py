from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

# Load model and tokenizer
model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    confidence = float(prediction[0][0]) if sentiment == "Positive" else 1 - float(prediction[0][0])
    return f"Sentiment: {sentiment} (Confidence: {confidence:.2%})"

title = "MOVIE REVIEW SENTIMENT ANALYSIS"
description = "Enter a movie review to analyze its sentiment (Positive/Negative)"

app = gr.Interface(
    fn=predictive_system,
    inputs=gr.Textbox(label="Movie Review", lines=5),
    outputs=gr.Textbox(label="Result"),
    title=title,
    description=description,
    examples=[
        ["This movie was fantastic! The acting was superb."],
        ["Terrible film with bad acting and weak plot."],
        ["It was okay, not great but not awful either."]
    ]
)

app.launch()