# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import dspy  # Ensure this is the library where your RAG class is defined
from langchain.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
# Initialize Chroma
client = chromadb.PersistentClient(path="db/")
collection = client.get_or_create_collection("profile_summarization")
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

promotion_keywords = f"""Technologies and skills taught during the course:

1. Antibody-Drug Conjugates (ADCs) mechanism of action and pharmacology.
2. GMP ADC manufacturing processes and technical challenges.
3. Regulatory landscape, including FDA and EMA requirements for ADCs.
4. Quality by Design (QbD) principles applied to ADCs.
5. Change management and tech transfer for ADCs.
6. Good manufacturing practices (GMP) procedures, specifically for ADCs.
7. Critical control points for CMC compliance in ADC manufacturing.

Categories of people interested in this course:

1. Quality professionals.
2. Quality assurance and control professionals.
3. CMC managers and heads.
4. Compliance professionals.
5. Regulatory affairs professionals.
6. QA/QC managers.
7. Analytical and development scientists.
8. Manufacturing, process development, upstream/downstream professionals.
9. R&D scientists.
10. Operations professionals.
11. Venture capitalists and investors in biologics."""
date = "10th September, 2024"
question = f"""From the given list of key technologies and list of person who might be interested in the course:{promotion_keywords}
           Extract individuals from the context, who may be interested in the attending the course. Generate for each individual personal invitation
           Invitation must contain the date of this event:{date} and the and following discount offer: `hurry up to buy a ticket with a discount of up to 25% until 28th June 2024`
           Response must contain full name, explanation of why you have chosen that person and personal invitation
            """
# Button to trigger the RAG model
if st.button("Generate Answer"):
    if question:
        # Get the answer from the RAG model
        prediction = rag_model.forward(question)


        st.subheader("Answer:")
        st.write(prediction.answer)
    else:
        st.write("Please enter a question.")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
