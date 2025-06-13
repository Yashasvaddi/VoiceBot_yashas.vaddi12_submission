import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

EMBEDDINGS_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
FAISS_INDEX_PATH = "./embeddings"

genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
model = genai.GenerativeModel("gemini-2.0-flash")
remember="You are a customer service executive, whose only job is to give the best response to the query. Give only the response, do not give anything before or after the actual answer"
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
                    Remember: {remember} 
                    Question: i{query}"""
                    
        ans = model.generate_content(f"From this extract only that part which a sales executive would say, {prompt}")
        response = model.generate_content(ans.text)
        print("Answer:", response.text)
