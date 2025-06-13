import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import json
import PyPDF2

def load_text_file(dir):
    all_text = []
    
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_text.append(json.dumps(data))
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_text.append(f.read())
            elif ext == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pdf_text = ''
                    for page in reader.pages:
                        pdf_text += page.extract_text() or ''
                    all_text.append(pdf_text)
    return "\n".join(all_text)


def create_documents_from_text(text):
    return [Document(page_content=text)]


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def embed_and_store(docs, persist_directory="./embeddings"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_directory)


if __name__ == "__main__":
    data_path = "./data/TrainingSet"
    output_path = "./embeddings"

    text = load_text_file(data_path)
    documents = create_documents_from_text(text)
    split_docs = split_documents(documents)
    embed_and_store(split_docs, output_path)

    print(f"FAISS index saved to: {output_path}")
