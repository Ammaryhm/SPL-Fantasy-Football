from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings

def get_embedder():
    return OpenAIEmbeddings()  # You can swap to Azure or HuggingFace later