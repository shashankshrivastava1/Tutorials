import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import VectorDBQA, OpenAI
import pinecone

pinecone.init(api_key="b9ea9e9a-10ad-47d2-8cf8-11a838e44fcf", environment="gcp-starter")

if __name__ == "__main__":
    print("Hello Vectors in pinecone and langchain")
    loader = TextLoader("Gandhi.txt", "utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    # print(texts[95])

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="tutorial-langchain"
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    query = "Give a numbered list of all the awards that Gandhi won"
    result = qa({"query": query})
    print(result)
