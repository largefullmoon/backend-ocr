import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from pinecone import (
    Pinecone,
)
import asyncio
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langdetect import detect
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import pdf2image
import base64
from io import BytesIO
import tempfile
from flask_socketio import SocketIO
import time


# Load environment variables
load_dotenv()

openai_client = OpenAI()

app = Flask(__name__)
CORS(app)
chat_history = []
# Set up Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE')

# Initialize Pinecone client
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
# Get the index instance
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Text splitter for chunking content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize OpenAI LLM
llm = ChatOpenAI(temperature=0.2, model='gpt-4')

# Add MongoDB connection after other initializations
# Initialize MongoDB client
mongo_client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB URI
db = mongo_client['chat_documents']  # Your database name
chat_collection = db['chat_history']
documents_collection = db['documents']
topics_collection = db['topics']

# Initialize Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

def convert_pdf_to_images(pdf_path):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of PIL Image objects
    """
    try:
        # Convert PDF to list of images
        images = pdf2image.convert_from_path(pdf_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return None

def encode_image_to_base64(image):
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image_with_vision(image):
    """
    Process an image using OpenAI's Vision API.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        
        # Call OpenAI Vision API
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Please extract all the text from this document image. Maintain the layout and formatting as much as possible."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )
        
        return response.output_text
    except Exception as e:
        print(f"Error processing image with Vision API: {str(e)}")
        return None

def process_pdf_with_vision(file_path):
    """
    Process a PDF file using OpenAI Vision API for better text extraction with layout preservation.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content with preserved layout
    """
    try:
        # Convert PDF to images
        images = convert_pdf_to_images(file_path)
        if not images:
            return None
            
        # Process each image with Vision API
        full_text = ""
        for i, image in enumerate(images):
            page_text = process_image_with_vision(image)
            if page_text:
                full_text += page_text + "\n---PAGE BREAK---\n"
                
        return full_text.strip()
    except Exception as e:
        print(f"Error processing PDF with Vision API: {str(e)}")
        return None

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    # Check if the post request has the file part dmn-ugfd-jai
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']
    
    # Check if filename is empty
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Save the uploaded file temporarily
        audio_path = os.path.join('temp_uploads', audio_file.filename)
        os.makedirs('temp_uploads', exist_ok=True)
        audio_file.save(audio_path)
        
        # Use OpenAI's Whisper for transcription
        with open(audio_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        # Clean up temporary file
        os.remove(audio_path)
        
        return jsonify({
            "message": transcription.text
        })
    
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/documents/upload', methods=['POST'])
def upload_documents():
    try:
        # Emit event to notify upload has started
        socketio.emit('start_upload')
        print("Upload started")
        # Check if files are in the request
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        uploaded_files = request.files.getlist('files')
        processed_documents = []
        
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join('upload', 'documents')  # Generic documents folder
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in uploaded_files:
            if file.filename == '':
                continue
            # Save the file
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            
            socketio.emit('start_scanning')
            print("Scanning started")
            # Process PDF using vision
            document_content = process_pdf_with_vision(file_path)
            
            socketio.emit('start_embedding')
            print("Embedding started")
            # If vision fails, fallback to doctr
            if document_content is None:
                pdf_reader = PdfReader(file_path)
                document_content = ""
                for page in pdf_reader.pages:
                    document_content += page.extract_text() + "\n---PAGE BREAK---\n"
            # Split document into smaller chunks
            documents = text_splitter.create_documents([document_content])
            langchain_docs = [
                Document(page_content=doc.page_content, metadata={
                    "source": file.filename,
                    "page_number": i + 1
                }) 
                for i, doc in enumerate(documents)
            ]
            # Store in Pinecone using OpenAI embeddings
            PineconeVectorStore.from_documents(
                documents=langchain_docs, 
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME, 
                namespace=PINECONE_NAMESPACE
            )
            # Store document information in MongoDB
            socketio.emit('start_querying')
            print("Querying started")
            time.sleep(10)
            answer = get_response(file.filename)
            document_info = {
                'file_name': file.filename,
                'file_path': file_path,
                'document_content': document_content, 
                'answer': answer
            }
            socketio.emit('upload_completed')
            print("Upload completed")
            result = documents_collection.insert_one(document_info)
            print(f"Document {file.filename} processed and stored with ID: {result.inserted_id}")
            # Convert ObjectId to string for JSON serialization
            document_info['_id'] = str(result.inserted_id)
            processed_documents.append(document_info)

        # Emit event to notify upload has completed
        return jsonify({
            "status": 'ok',
            "message": "Documents uploaded and processed successfully",
            "documents": processed_documents
        })
        
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_response/<document_id>', methods=['GET'])
def get_response_endpoint(document_id):
    # Convert string ID to ObjectId
    doc_id = ObjectId(document_id)
    # Find the document in MongoDB
    document = documents_collection.find_one({"_id": doc_id})
    response = get_response(document['file_name'])
    return jsonify({"response": response})

def get_response(document_name):
    try:
        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE
        )
        # Create a retriever with a filter for user_id
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 25,
                "filter": {
                    "source": document_name
                }
            }
        )
        
        # Modify the custom prompt to include conversation history
        custom_prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Your task is to extract important information from the provided context.
            Context: {context}
            Current Question: {question}
            Please provide the extracted information in a clear, structured Markdown format in Spainish:
            """,
            input_variables=["context", "question"]
        )

        # Initialize LLM and QA chain with chat history
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": custom_prompt,
                "verbose": True
            }
        )

        query = "Hola necesito que me resumas esta plusvalia con los datos mas importantes para hacer su liquidacion"
        
        result = qa_chain.invoke({
            "query": query,
        })
        
        # Extract the result text
        response = result.get('result', 'No response generated')

        return response  # Return the response as a string

    except Exception as e:
        import traceback
        print(f"Error getting response: {str(e)}")
        print(traceback.format_exc())
        return "Failed to get response: {}".format(str(e))  # Return a string error message

# Add new endpoint to view a document by ID
@app.route("/api/documents/<document_id>/view", methods=["GET"])
def view_document(document_id):
    try:
        # Convert string ID to ObjectId
        doc_id = ObjectId(document_id)
        
        # Find the document in MongoDB
        document = documents_collection.find_one({"_id": doc_id})
        
        if not document:
            return jsonify({"error": "Document not found"}), 404
        
        # Get the file path from the document
        file_path = document.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404
        
        # Send the file as a response
        return send_file(file_path, mimetype='application/pdf')
        
    except Exception as e:
        import traceback
        print(f"Error viewing document: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to view document: {str(e)}"}), 500

# Add this new endpoint to get all documents
@app.route("/api/documents", methods=["GET"])
def get_all_documents():
    try:
        # Retrieve all documents from MongoDB
        documents = list(documents_collection.find({}, {'document_content':0}))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc['_id'] = str(doc['_id'])
        return jsonify({"documents": documents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add new endpoint to delete a document
@app.route("/api/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    try:
        # Convert string ID to ObjectId
        doc_id = ObjectId(document_id)
        
        # Find the document in MongoDB
        document = documents_collection.find_one({"_id": doc_id})
        
        if not document:
            return jsonify({"error": "Document not found"}), 404
        
        # Get the file path
        file_path = document.get('file_path')
        
        # Delete the physical file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete document from MongoDB
        documents_collection.delete_one({"_id": doc_id})
        
        # Delete document embeddings from Pinecone
        # Note: This assumes the document's filename is used as the metadata source
        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE
        )
        
        # Delete vectors with matching metadata
        pinecone_index.delete(
            namespace=PINECONE_NAMESPACE,
            filter={"source": document.get('file_name')}
        )
        
        return jsonify({"message": "Document deleted successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app with SocketIO
if __name__ == '__main__':
    socketio.run(app, debug=True, port=8021, host='0.0.0.0', allow_unsafe_werkzeug=True)
