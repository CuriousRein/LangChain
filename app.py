
from msilib.schema import TextStyle
from langchain import FAISS, QAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.llms.openai import OpenAIChat


llm = OpenAI(openai_api_key="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")
llm = ChatOpenAI()




# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('test.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
retriever = FAISS.from_documents(documents, OpenAIEmbeddings()).as_retriever()
handler =  StdOutCallbackHandler()
docs = retriever.get_relevant_documents("how old is mike enriquez?")
print(docs[0].page_content)
qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)
response = qa_with_sources_chain({"query":"how old is mike enriquez?"})

print(response['result'])
