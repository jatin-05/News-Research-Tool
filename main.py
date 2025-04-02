from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" , temperature=0.9)



from langchain_community.document_loaders import UnstructuredURLLoader

# loader = UnstructuredURLLoader(urls= ['https://www.bbc.com/news/live/c2dep0r5ygnt' , 'https://www.bbc.com/news/articles/c99pvmxg0e3o'])

# data =loader.load()
# print(len(data))
# print(data[0].page_content)



from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap = 200 
# )

# docs = text_splitter.split_documents(data)

# print(len(docs))

# print(docs[5])


from sentence_transformers import SentenceTransformer


# embeding = SentenceTransformer("C:/Users/SSS/MACHINE LEARNING/NewsChat/all-MiniLM-L6-v2")


# a =model.encode('hello')
# print(a)
from langchain_huggingface import HuggingFaceEmbeddings
# embedding_model = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )
from langchain_huggingface import HuggingFaceEndpointEmbeddings
embedding_model = HuggingFaceEndpointEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')


from langchain.vectorstores import FAISS
import pickle

# vector_index = FAISS.from_documents(docs ,embeding)
# file_path = "vector_index.pkl"
# with open(file_path,"wb") as f :
#     pickle.dump(vector_index,f)

from langchain.chains import RetrievalQAWithSourcesChain

# RetrievalQAWithSourcesChain(llm= model ,retriever=vectorIndex.as)



file_path = 'faiss_store.pkl'

import streamlit as st
st.title("Newss Research Toool")

st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url =st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_onclick = st.sidebar.button("Process URLs")
main_palceholder = st.empty()
if(process_url_onclick):
    loader = UnstructuredURLLoader(urls= urls)
    main_palceholder.text("data is loading.......")
    data =loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n' , '\n' , '.' , ','],
        chunk_size = 1000
    )
    main_palceholder.text("data splitting started......")
    docs =text_splitter.split_documents(data)

    main_palceholder.text("data embedding started......")
    vector_store =FAISS.from_documents(docs , embedding_model)

    # with open(file_path,"wb") as f :
    #     pickle.dump(vector_store,f)
    st.session_state.vector_store = vector_store

# queries =[]
if "queries" not in st.session_state:
    st.session_state.queries = []


query = main_palceholder.text_input("question ?")

placeholder = st.empty()
import os
if query :
    # if os.path.exists(file_path):
    #     with open(file_path,"rb") as f :
    if "vector_store" in st.session_state:
        vector_store = st.session_state.vector_store
        placeholder.text('loading...')
        # vector_store = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm= model ,retriever=vector_store.as_retriever())
        result = chain.invoke({"question" : query}, return_only_outputs= True)
        # queries.append([query , result])
        st.session_state.queries.append([query, result])

        st.header("Answer")
        st.write(result["answer"])
        placeholder.text("")
        st.subheader('Sources')
        sources = result['sources']
        source_list =sources.split("\n")
        for s in source_list:
            st.write(s)

st.header(" ")

st.subheader("History") ; 

for i in range (len(st.session_state.queries) -1):
    st.write(f"Query :{st.session_state.queries[i][0]}")
    st.write(f"Answer:{st.session_state.queries[i][1]['answer']}")
    st.write(f"Source:{st.session_state.queries[i][1]['sources']}")
    st.write('---')









