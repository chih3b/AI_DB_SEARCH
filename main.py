from langchain_helper import few_shots_db_chain
import streamlit as st
st.title("T shirts store: Database Q&A 👕")
question=st.text_input("Question:")
if question:
    chain=few_shots_db_chain()
    answer=chain.run(question)
    st.header("Answer:")
    st.write(answer)