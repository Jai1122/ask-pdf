import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import requests
import faiss
import numpy as np
import os


class CharacterTextSplitter:
    def __init__(self, separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text):
        chunks = []
        start = 0
        text_length = self.length_function(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks


def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        def get_embeddings(text):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            embeddings  = outputs.last_hidden_state.mean(dim=1)
            return embeddings.detach().numpy()

        embeddings = [get_embeddings(chunk) for chunk in chunks]
        embeddings = np.vstack(embeddings)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        user_question = st.text_input("Ask your question about PDF")

        if user_question:
            question_embedding  =   get_embeddings(user_question)
            _, indices = index.search(question_embedding, k=5)
            docs = [chunks[i] for i in indices[0]]

            api_key =  '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'
            url = 'https://api.anthropic.com/v1/complete'
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            prompt = f"Answer the question based on the following context:\n{docs}\n\nQuestion: {user_question}"
            payload = {
                'prompt': prompt,
                'max_tokens': 150
            }

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                st.write(data['completion'])
            else:
                st.write(f'Error: {response.status_code}, {response.text}')


if __name__ == "__main__":
    main()

