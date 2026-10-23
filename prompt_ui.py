
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 500,
        "do_sample": True,
    },
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
