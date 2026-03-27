import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Cyberbullying Detector", layout="centered")

# Load model (cached)
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="unitary/toxic-bert")

classifier = load_model()

# --- Smarter Explanation Function ---
def explain_toxicity(score):
    if score > 0.9:
        return "This message contains strong toxic or abusive language with high confidence."
    
    elif score > 0.75:
        return "This message likely contains offensive or harmful language."
    
    elif score > 0.6:
        return "This message may include mild toxicity, sarcasm, or negative tone."
    
    else:
        return "No significant signs of toxicity detected."

# --- Title ---
st.title("💬 Cyberbullying Detector")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Input ---
user_input = st.text_input("Type a message:")

# --- Send Button ---
if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.messages.append(("user", user_input))

        # Model prediction
        result = classifier(user_input)[0]
        score = result['score']
        toxicity_percent = round(score * 100, 2)

        threshold = 0.6

        # Decision logic
        if score > threshold:
            explanation = explain_toxicity(score)
            response = f"🚨 Toxic ({toxicity_percent}%)\n\n{explanation}"
        else:
            explanation = explain_toxicity(score)
            response = f"✅ Non-toxic ({toxicity_percent}%)\n\n{explanation}"

        st.session_state.messages.append(("bot", response))

# --- Display Chat ---
for role, message in st.session_state.messages:
    if role == "user":
        st.markdown(
            f"""
            <div style='text-align: right; background-color: #262730; padding: 10px; border-radius: 10px; margin: 5px;'>
                <b>You:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='text-align: left; background-color: #1e3a2f; padding: 10px; border-radius: 10px; margin: 5px;'>
                <b>AI:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )