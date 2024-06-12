__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

##### LLAMAPARSE #####
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain

from groq import Groq
from langchain_groq import ChatGroq

import joblib
import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

llamaparse_api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
groq_api_key = os.environ.get("GROQ_API_KEY")

class Query(BaseModel):
    question: str
    session_id: str

# Custom prompt template
custom_prompt_template = """You are Zilla, an AI assistant created by Humexid (Founded by Damilare Odueso). Your role is to provide helpful solutions about Humexid's products, Crendly and Trustbreed, while also offering general assistance to the customers. You have access to specific information about these products from provided documents, as well as extensive general knowledge.

When answering queries about Crendly or Trustbreed, use the following context:
{context}

Always cite your sources precisely:
1. For Crendly or Trustbreed info, cite as "Crendly PDF, Page X" or "Trustbreed PDF, Page X."
2. For external info, provide the full, direct URL (e.g., "https://www.example.com/page").
3. If combining sources, list all with their specific citations.

Previous conversation:
{chat_history}

Current question: {question}

Always Maintain professionalism and don't reintroduce yourself when you had already done that.


Helpful Answer:"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question', 'chat_history']
    )
    return prompt

prompt = set_custom_prompt()


def load_or_parse_data(pdf_dir="pdf/"):
    data_file = "data/parsed_data.pkl"
    if os.path.exists(data_file):
        parsed_data = joblib.load(data_file)
    else:
        parsingInstructionTrustbreed = """The provided document is a contains information about Trustbreed Inc, a platform that helps
        people escalate their complaints to the service providers for faster resolution, and Crendly - an I-driven P2P lending service.
        Try to be precise while answering the questions. Always include the page number where you found the information."""
        parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsingInstructionTrustbreed, max_timeout=5000,)
         # Get all PDF file paths
        
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        print(f"Found PDF files: {pdf_files}")
        
        # Pass all file paths to LlamaParse at once
        llama_parse_documents = parser.load_data(pdf_files)
        
        # Extract and store page numbers
        for doc in llama_parse_documents:
            page_num = doc.metadata.get('page_number')
            if page_num:
                doc.metadata['source'] = f"Trustbreed PDF, Page {page_num}"

        joblib.dump(llama_parse_documents, data_file)
        parsed_data = llama_parse_documents
    return parsed_data

parsed_data = load_or_parse_data()

# def create_vector_database_remove():
#     with open('data/output.md', 'w') as f:  # Open the file in write mode ('w') to overwrite
#         for doc in parsed_data:
#             f.write(doc.text + '\n')

#     markdown_path = "data/output.md"
#     loader = UnstructuredMarkdownLoader(markdown_path)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
#     docs = text_splitter.split_documents(documents)

#     print(f"length of documents loaded: {len(documents)}")
#     print(f"total number of document chunks generated :{len(docs)}")

#     embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

#     vs = Chroma.from_documents(
#         documents=docs,
#         embedding=embed_model,
#         persist_directory="chroma_db_llamaparse1",
#         collection_name="equitech"
#     )

#     print('Vector DB created successfully!')
#     return vs, embed_model

def create_vector_database():
    persist_directory = "chroma_db_llamaparse1"
    collection_name = "equitech"
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print('Loaded existing Vector DB.')

    return vectorstore, embed_model

# Initialize Groq for acceleration
groq_accelerator = Groq()

# Initialize Groq Chat model
chat_model = ChatGroq(
    temperature=0.2,
    model_name='llama3-70b-8192',
    api_key=groq_api_key,
)

vs, embed_model = create_vector_database()
vectorstore = Chroma(embedding_function=embed_model, persist_directory="chroma_db_llamaparse1", collection_name="equitech")
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})  # Adjusted to top 5 for more precise results

# Dictionary to store conversations
conversations = {}

def get_qa_chain(session_id):
    if session_id not in conversations:
        memory = ConversationBufferMemory(
            input_key="question",
            memory_key="chat_history",
            output_key="answer"  # Explicitly set the output key
        )
        conversations[session_id] = memory
    else:
        memory = conversations[session_id]

    # Create a base chain with the right input/output keys
    doc_chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    
    # Now create the retrieval chain
    qa_chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=doc_chain,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
    )
    
    return qa_chain

def get_answer(question, session_id):
    qa_chain = get_qa_chain(session_id)
    
    try:
        # Query the RAG system
        rag_response = qa_chain({"question": question})
        rag_answer = rag_response['answer']
        sources = rag_response.get('sources', [])
        source_documents = rag_response.get('source_documents', [])
        
        # # Initialize lists for internal and external sources
        # trustbreed_sources = []
        # external_sources = []
        
        # # First, try to use the provided sources
        # if sources:
        #     for source in sources:
        #         if 'Trustbreed PDF' in source:
        #             trustbreed_sources.append(source)
        #         else:
        #             # Validate and format the URL
        #             valid_url = validate_and_format_url(source)
        #             if valid_url:
        #                 external_sources.append(valid_url)
        #             else:
        #                 # If it's not a valid URL, ask for a correct one
        #                 source_query = f"The source '{source}' is not a valid direct URL. Please provide a correct, full URL that supports this information."
        #                 new_url = chat_model.invoke(source_query).content.strip()
        #                 valid_new_url = validate_and_format_url(new_url)
        #                 external_sources.append(valid_new_url or "Source URL not available")
        
        # # If no sources were provided or they were insufficient, use source_documents
        # if not (trustbreed_sources or external_sources) and source_documents:
        #     for doc in source_documents:
        #         source = doc.metadata.get('source', 'Unknown Source')
        #         if 'Trustbreed PDF' in source:
        #             trustbreed_sources.append(source)
        #         else:
        #             # Ask the model to find a relevant external source
        #             source_query = f"Provide a reputable external source with a direct, full URL that supports this information: {doc.page_content}"
        #             external_url = chat_model.invoke(source_query).content.strip()
                    
        #             # Validate and format the provided URL
        #             valid_url = validate_and_format_url(external_url)
        #             if valid_url:
        #                 external_sources.append(valid_url)
        #             else:
        #                 # If it's not a valid URL, ask again
        #                 retry_query = f"The previous URL '{external_url}' is not valid. Please provide a correct, direct, full URL for this information."
        #                 new_url = chat_model.invoke(retry_query).content.strip()
        #                 valid_new_url = validate_and_format_url(new_url)
        #                 external_sources.append(valid_new_url or "Source URL not available")
        
        # # Format the response with both internal and external sources
        response = f"{rag_answer}\n\nSources: {sources}\n"
        
        # if trustbreed_sources:
        #     response += "From Trustbreed Document:\n"
        #     for source in trustbreed_sources:
        #         response += f"- {source}\n"
        
        # if external_sources:
        #     response += "\nExternal Sources:\n"
        #     for source in external_sources:
        #         response += f"- {source}\n"
        
        # # If no relevant documents found, use general knowledge
        # if not (trustbreed_sources or external_sources):
        #     # Ask for a comprehensive answer with multiple reliable sources
        #     prompt = f"""You don't have specific information about that in your Trustbreed documents. 
        #     Please provide a comprehensive answer to this question based on your general knowledge: {question}

        #     Include at least two reputable external sources. For each source, provide a direct, full URL that supports your answer. Ensure each URL is valid and starts with http:// or https://."""
        #     general_knowledge = chat_model.invoke(prompt).content
            
        #     # Validate the URLs in the general knowledge response
        #     lines = general_knowledge.split('\n')
        #     validated_lines = []
        #     for line in lines:
        #         if 'http://' in line or 'https://' in line:
        #             url_start = line.find('http')
        #             url_end = line.find(' ', url_start)
        #             url_end = url_end if url_end != -1 else len(line)
        #             url = line[url_start:url_end]
        #             valid_url = validate_and_format_url(url)
        #             if valid_url:
        #                 line = line.replace(url, valid_url)
        #             else:
        #                 line += " (URL validation failed)"
        #         validated_lines.append(line)
            
        #     general_knowledge = '\n'.join(validated_lines)
        #     response = f"I don't have specific information about that in my Trustbreed documents. However, based on my general knowledge:\n\n{general_knowledge}"
        
        return response
    except Exception as e:
        print(f"Error in get_answer: {e}")
        return f"I encountered an error while trying to answer your question. Please try again: {e}"

@app.post("/ask")
async def ask_question(query: Query):
    try:
        answer = get_answer(query.question, query.session_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add an endpoint to clear a session's history
@app.post("/clear_session")
async def clear_session(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
        return {"message": "Session history cleared"}
    else:
        return {"message": "No such session found"}
    

import re

def validate_and_format_url(url):
    # Basic URL validation regex
    pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(pattern, url):
        # Ensure it starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    return None