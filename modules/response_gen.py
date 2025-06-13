import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers.pipelines import pipeline

EMBEDDINGS_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FAISS_INDEX_PATH = "./embeddings"

genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_pipeline = pipeline("text2text-generation", model=genai.GenerativeModel("gemini-2.0-flash"), device=0)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        print("Answers:", result["result"])
        print("\n")
