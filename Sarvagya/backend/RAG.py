import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

def rag_preprocess(json_data_list):
    user_assistant_dict = {}
    for json_data in json_data_list:

        mappings = json_data.get("mapping", {})

        for key, value in mappings.items():
            if value.get("message") and value["message"]["author"]["role"] == "user":
                user_content = value["message"]["content"]["parts"]
                children_list = value.get("children", [])
                next_author_key = children_list[0] if children_list else None
                next_author_content = mappings.get(next_author_key, {}).get("message", {}).get("content", {}).get("parts", [])
                user_assistant_dict[user_content[0]] = next_author_content;
    items_list = list(user_assistant_dict.items())
    return items_list

def generate_output_file(items_list):

    with open("output.txt", "w") as file:
        # Iterate over the list and write each element to the file
        for user_content, assistant_content in items_list:
            file.write(f"User Content: {user_content}" + "\n")
            file.write(f"Assistant Content: {assistant_content}" + "\n")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_model():
    loader = TextLoader("output.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI()
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, answer it without using the context. Keep the answer concise.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain