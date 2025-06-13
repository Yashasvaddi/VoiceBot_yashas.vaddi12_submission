import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

EMBEDDINGS_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FAISS_INDEX_PATH = "./embeddings"

genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
model = genai.GenerativeModel("gemini-1.5-flash")

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""Context:
                    {context}
                    Question: {query}
                    Answer:"""

        response = model.generate_content(prompt)
        print("Answer:", response.text)
