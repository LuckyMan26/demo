# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
import dspy  # Ensure this is the library where your RAG class is defined
from langchain.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
# Initialize Chroma
client = chromadb.Client(Settings(persist_directory="db/"))
collection = client.create_collection("profile_summarization")
model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
langchain_chroma = Chroma(
    client=client,
    collection_name="profile_summarization",
    embedding_function=model
)

lm = dspy.GROQ(model='mixtral-8x7b-32768', api_key ="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",max_tokens=1000 )
dspy.configure(lm=lm)
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = langchain_chroma
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        self.num_of_profiles = num_passages

    def forward(self, question):
        context = self.retrieve.similarity_search(question, k=5)
        answer = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=answer.answer)


# Instantiate the RAG model
rag_model = RAG()

# Streamlit app
st.title("RAG-based Question Answering")

# Input box for the question
question = st.text_input("Enter your question:")

# Button to trigger the RAG model
if st.button("Generate Answer"):
    if question:
        # Get the answer from the RAG model
        prediction = rag_model.forward(question)

        # Display the context and the answer
        st.subheader("Context:")
        for i, context in enumerate(prediction.context):
            st.write(f"{i + 1}. {context}")

        st.subheader("Answer:")
        st.write(prediction.answer)
    else:
        st.write("Please enter a question.")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
