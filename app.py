import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(page_title="Spam Mail Classifier", page_icon="✉️", layout="centered")
st.title("✉️ Spam Mail Classifier")
st.markdown("Enter an email or text message below to analyze whether it is **Ham** (Safe) or **Spam**.")

@st.cache_resource
def load_and_train_model():
    url = "https://raw.githubusercontent.com/Anshika1517/spam-filter-project/refs/heads/main/mail_data.csv"
    df = pd.read_csv(url)
    data = df.where((pd.notnull(df)), '')
    
    data.loc[data['Category'] == 'spam', 'category'] = 0
    data.loc[data['Category'] == 'ham', 'category'] = 1
    
    X = data['Message']
    Y = data['category'].astype(int)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    
    return feature_extraction, model

with st.spinner("Initializing AI Model..."):
    feature_extraction, model = load_and_train_model()

user_input = st.text_area(label="Message Content", placeholder="Paste your email text here...", height=150)

if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please type or paste a message first!")
    else:
        input_data_features = feature_extraction.transform([user_input])
        prediction = model.predict(input_data_features)
        
        st.write("---")
        if prediction[0] == 1:
            st.success("### ✅ Ham Mail\nThis message looks perfectly safe!")
        else:
            st.error("### 🚨 Spam Mail\nCaution! This looks like a spam message.")
