__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings
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
from langchain_anthropic import ChatAnthropic

import joblib
import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()
# Add CORS middleware if needed
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

llamaparse_api_key = os.environ.get('LLAMA_CLOUD_API_KEY')
groq_api_key = os.environ.get("GROQ_API_KEY")
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom prompt template
custom_prompt_template = """You are an advanced AI Customer Care agent for Trustbreed, a platform that helps users escalate complaints and review brands. Your primary goal is to assist users in resolving their issues efficiently and effectively. You have access to specific information about complaints, resolutions, and company FAQs from provided documents, as well as extensive general knowledge.

When answering queries or addressing complaints, use the following context:
{context}


Previous conversation:
{chat_history}

Current query or complaint: {question}

Maintain a professional, empathetic tone throughout. Don't reintroduce yourself if you've already done so. Follow these guidelines:

1. Analyze the query or complaint thoroughly.
2. If you can provide a resolution based on the context, do so clearly and concisely.
3. Always Speak as a representative of Trustbreed, using "we" or "I" instead of "they" when interacting with customers.
4. If more information is needed, ask targeted follow-up questions.
5. If you cannot fully resolve the issue, guide the user on submitting a formal complaint through Trustbreed, and do not try to make up an answer.
6. Tailor responses to the specific company or brand mentioned when applicable.
7. Prioritize user privacy and avoid asking for sensitive information unless necessary.
8. After providing assistance, ask for feedback on the solution or guidance offered.
9. When necessary, always mention the specific company name you're in the response you're providing to a customer.


Remember to maintain context in multi-turn conversations and handle unclear inputs gracefully. Always strive to exceed the level of service provided by an expert human customer care representative.

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
        parsingInstructionTrustbreed = """
    The provided document contains information about Trustbreed, a platform that helps users escalate complaints and review brands. There is also a document that contains questions, answers, complaints, and resolutions related to various companies.
    - Extract all the information about Trustbreed
    - Extract all questions and their corresponding answers.
    - Extract all complaints and their associated resolutions.
    - Identify and extract the company names associated with each set of information.

  
    - For each company, create a separate section with the company name as the heading.
    - Under each company heading, clearly label and separate:
        - Questions and Answers
        - Complaints and Resolutions
    - Maintain the original order of information as it appears in the document.

    - For each extracted piece of information (question, answer, complaint, or resolution), indicate the page number from which it was extracted.
    - Format page numbers as: [Page X] at the end of each extracted item.

    - If a single item (question, answer, complaint, or resolution) spans multiple pages, indicate the range:
    [Pages X-Y]

    - Maintain any crucial formatting (e.g., bullet points, numbered lists) within the extracted text.

    - f it's unclear which company a piece of information belongs to, place it under a separate "Unspecified Company" heading.

    - Ensure all relevant information from the document is extracted and properly categorized.
    
    - Please process the entire document according to these instructions, ensuring accuracy in extraction and clear organization of the output.
        """
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

def create_vector_database_remove():
    with open('data/output.md', 'w') as f:  # Open the file in write mode ('w') to overwrite
        for doc in parsed_data:
            f.write(doc.text + '\n')

    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",
        collection_name="trustbreed"
    )

    print('Vector DB created successfully!')
    return vs, embed_model

def create_vector_database():
    persist_directory = "chroma_db_llamaparse1"
    collection_name = "trustbreed"
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    #embed_model = OpenAIEmbeddings(model='text-embedding-3-large')
    vectorstore = Chroma(
        #embedding_function=embed_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print('Loaded existing Vector DB.')

    return vectorstore, embed_model

# Initialize Groq for acceleration
#groq_accelerator = Groq()

# Initialize Groq Chat model
# chat_model = ChatGroq(
#     temperature=0.2,
#     model_name='llama3-70b-8192',
#     api_key=groq_api_key,
# )

chat_model = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=anthropic_api_key, max_tokens=400)

vs, embed_model = create_vector_database()
#embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(embedding_function=embed_model, persist_directory="chroma_db_llamaparse1", collection_name="trustbreed")
retriever = vs.as_retriever(search_kwargs={'k': 5})  # Adjusted to top 5 for more precise results

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
        
        
        response = f"{rag_answer}"
        
        return response
    except Exception as e:
        print(f"Error in get_answer: {e}")
        return f"I encountered an error while trying to answer your question. Please try again: {e}"

# @app.post("/ask")
# async def ask_question(query: Query):
#     try:
#         answer = get_answer(query.question, query.session_id)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Optional: Add an endpoint to clear a session's history
# @app.post("/clear_session")
# async def clear_session(session_id: str):
#     if session_id in conversations:
#         del conversations[session_id]
#         return {"message": "Session history cleared"}
#     else:
#         return {"message": "No such session found"}
    

class Query(BaseModel):
    question: str
    session_id: str

class APIResponse(BaseModel):
    status: str
    status_code: int
    message: str
    data: Optional[dict] = None
    timestamp: str


@app.post("/ask", response_model=APIResponse)
async def ask_question(query: Query):
    """
    Process a user's question and provide an AI-generated response.

    This endpoint takes a user's question/chat and session ID, processes the query using
    an AI model, and returns a contextually relevant answer.

    Parameters:
    - question (str): The user's question or complaint.
    - session_id (str): A unique identifier for the user's session to maintain conversation context.

    Returns:
    - APIResponse: A structured response containing:
        - status (str): 'success' or 'error'
        - status_code (int): HTTP status code
        - message (str): A brief description of the result
        - data (dict): Contains the 'answer' key with the AI-generated response
        - timestamp (str): ISO format timestamp of the response

    Raises:
    - HTTPException: 500 Internal Server Error if processing fails
    """
    try:
        answer = get_answer(query.question, query.session_id)
        return APIResponse(
            status="success",
            status_code=status.HTTP_200_OK,
            message="Question answered successfully",
            data={"answer": answer},
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/clear_session", response_model=APIResponse)
async def clear_session(session_id: str):
    """
    Clear the conversation history for a specific session.

    This endpoint removes all stored conversation context for the given session ID,
    effectively starting a new conversation.

    Parameters:
    - session_id (str): The unique identifier of the session to be cleared.

    Returns:
    - APIResponse: A structured response containing:
        - status (str): 'success'
        - status_code (int): HTTP status code (200)
        - message (str): Confirmation of session clearance or notification if session wasn't found
        - timestamp (str): ISO format timestamp of the response

    Note:
    - This operation is idempotent; calling it on a non-existent session is not an error.
    """
    if session_id in conversations:
        del conversations[session_id]
        return APIResponse(
            status="success",
            status_code=status.HTTP_200_OK,
            message="Session history cleared successfully",
            timestamp=datetime.utcnow().isoformat()
        )
    else:
        return APIResponse(
            status="success",
            status_code=status.HTTP_200_OK,
            message="No such session found",
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/health", response_model=APIResponse)
async def health_check():
    """
    Check the health status of the API.

    This endpoint can be used for monitoring and ensuring the API is operational.
    It doesn't check the health of dependent services (e.g., database, AI model),
    but confirms that the API server itself is running and responsive.

    Returns:
    - APIResponse: A structured response containing:
        - status (str): 'success' indicating the API is operational
        - status_code (int): HTTP status code (200)
        - message (str): Confirmation that the API is healthy and running
        - timestamp (str): ISO format timestamp of the response

    Note:
    - This endpoint can be used by monitoring tools or load balancers to check service availability.
    """
    return APIResponse(
        status="success",
        status_code=status.HTTP_200_OK,
        message="API is healthy and running",
        timestamp=datetime.utcnow().isoformat()
    )

# Error handler for unhandled exceptions
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return APIResponse(
        status="error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        data={"error_details": str(exc)},
        timestamp=datetime.utcnow().isoformat()
    )