import os
import telebot
import re
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

# ===== CONFIGURATION =====
# Get tokens from environment variables (safer for deployment)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    print("❌ Please set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY environment variables")
    exit(1)

# ===== INITIALIZE COMPONENTS =====
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables for RAG components
embedding_model = None
qdrant = None

# ===== RAG HELPER FUNCTIONS =====
def extract_text_from_markdown(md_path: str) -> str:
    """Extract text from a Markdown (.md) file."""
    all_text = ""
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            all_text = file.read()
            print(f"📄 Successfully loaded content from {md_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        all_text = f"Error: {e}"
    return all_text

def clean_text(text: str) -> str:
    """Clean Markdown or OCR text."""
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\|\s*-+\s*\|(\s*-+\s*\|)*', '', text)
    text = re.sub(r'^\s*(\|\s*){2,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'!', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text_from_string(text: str):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", "।", ".", " ", ""]
    )
    documents = [{"content": text, "source": "hindi_teacher_guide"}]
    chunks = []
    for doc in documents:
        split_docs = text_splitter.create_documents(
            [doc["content"]],
            metadatas=[{"source": doc["source"]}]
        )
        chunks.extend(split_docs)
    
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
    return chunks

def initialize_rag_system():
    """Initialize the RAG system with the teacher training content."""
    global embedding_model, qdrant
    
    try:
        # Try different possible file paths
        possible_paths = [
            "ocr_result.md"
        ]
        
        md_path = None
        for path in possible_paths:
            if os.path.exists(path):
                md_path = path
                break
        
        if not md_path:
            print("❌ ocr_result.md file not found in any expected location")
            return False
        
        print(f"🔄 Loading content from: {md_path}")
        markdown_text = extract_text_from_markdown(md_path)
        
        if not markdown_text or markdown_text.startswith("Error:"):
            print("❌ Failed to load markdown file")
            return False
            
        cleaned_text = clean_text(markdown_text)
        print(f"📄 Cleaned text length: {len(cleaned_text)} characters")
        
        chunks = chunk_text_from_string(cleaned_text)
        print(f"📊 Created {len(chunks)} text chunks")
        
        if len(chunks) == 0:
            print("❌ No chunks created - check your content")
            return False
        
        # Initialize embedding model
        print("🔄 Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="AkshitaS/bhasha-embed-v0")
        print("✅ Embedding model loaded")
        
        # Create vector database
        print("🔄 Creating vector database...")
        qdrant = Qdrant.from_documents(
            chunks,
            embedding_model,
            location=":memory:",
            collection_name="teacher_training_documents",
        )
        print("✅ Vector database created")
        
        # Test the system
        test_query = "गणित पाठ योजना"
        test_embedding = embedding_model.embed_query(test_query)
        test_results = qdrant.similarity_search_by_vector(test_embedding, k=3)
        
        if test_results:
            print(f"✅ RAG system working! Found {len(test_results)} test results")
            return True
        else:
            print("⚠️ Vector database created but no test results found")
            return False
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_rag_system(query: str):
    """Query the RAG system and get response."""
    global embedding_model, qdrant
    
    if not embedding_model or not qdrant:
        return "❌ RAG system not initialized. Please contact administrator."
    
    try:
        print(f"🔍 Processing query: {query}")
        query_embedding = embedding_model.embed_query(query)
        top_matches = qdrant.similarity_search_by_vector(query_embedding, k=7)
        print(f"📊 Found {len(top_matches)} matching contexts")
        
        if not top_matches:
            return "क्षमा करें, मुझे आपके प्रश्न से संबंधित जानकारी नहीं मिली। कृपया अपना प्रश्न अलग तरीके से पूछें।"
        
        response = pass_to_llm_for_teacher_training(query, top_matches)
        return response
        
    except Exception as e:
        print(f"❌ Error in RAG query: {e}")
        return f"क्षमा करें, एक तकनीकी समस्या आई है। कृपया बाद में पुनः प्रयास करें।"

def pass_to_llm_for_teacher_training(query, top_matches):
    """Use Gemini to generate response based on retrieved context."""
    system_prompt = """
आप एक सहायक हैं जो शिक्षकों की प्रशिक्षण गाइड के आधार पर हिंदी में उत्तर देता है।

आपका उत्तर इन सिद्धांतों पर आधारित होना चाहिए:
1. अस्पष्ट या अधूरी पूछी गई बातों को समझकर स्पष्ट सुझाव दें।
2. सप्ताह, दिन, और विषय के आधार पर structured तरीके से जानकारी दें।
3. उपयोगकर्ता द्वारा माँगा गया आउटपुट दें — जैसे चिट-शीट, remedial संस्करण, या किसी गतिविधि का विश्लेषण।
4. हमेशा दिए गए संदर्भ (context) से ही जानकारी दें। अगर जानकारी मौजूद नहीं है, तो साफ़-साफ़ कहें कि वह गाइड में नहीं है।
5. उत्तर केवल हिंदी में दें और स्रोत का सही-सही संदर्भ दें (उदाहरण: सप्ताह 3, दिन 2)।
6. उत्तर के अंत में हमेशा "धन्यवाद! यदि आपका कोई और प्रश्न है तो बेझिझक पूछें।" जोड़ें।
"""

    context_text = "\n\n---\n\n".join(match.page_content for match in top_matches)
    full_prompt = f"""{system_prompt}

संदर्भ:
{context_text}

प्रश्न:
{query}

उत्तर:"""

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "क्षमा करें, मैं इस समय आपकी सहायता नहीं कर पा रहा। कृपया बाद में पुनः प्रयास करें।"

# ===== TELEGRAM BOT HANDLERS =====
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Handle start and help commands."""
    user_name = message.from_user.first_name or "शिक्षक जी"
    
    welcome_message = f"""
नमस्ते {user_name}! 🙏

मैं Central Square Foundation Bot हूँ। आज मैं आपकी तीसरी कक्षा के गणित शिक्षण में कैसे सहायता कर सकता हूँ? 

मैं निम्नलिखित में आपकी मदद कर सकता हूँ:
📚 गणित पाठ योजनाओं को समझने में
📖 शिक्षक गाइड में विशिष्ट सामग्री खोजने में  
🎯 गणित अवधारणाओं के लिए शिक्षण रणनीतियों को लागू करने में
💡 पाठ योजना और गतिविधियों के सुझाव देने में

कृपया अपना प्रश्न हिंदी में पूछें। मैं आपकी सहायता के लिए यहाँ हूँ! 😊
"""
    
    bot.reply_to(message, welcome_message)

@bot.message_handler(func=lambda message: True)
def handle_teacher_query(message):
    """Handle all text messages from teachers."""
    user_name = message.from_user.first_name or "शिक्षक जी"
    user_query = message.text.strip()
    
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        response = query_rag_system(user_query)
        bot.reply_to(message, response)
        
    except Exception as e:
        error_message = f"क्षमा करें {user_name}, मुझे एक तकनीकी समस्या आई है। कृपया बाद में पुनः प्रयास करें।"
        bot.reply_to(message, error_message)
        print(f"❌ Error handling message: {e}")

# ===== MAIN EXECUTION =====
def main():
    """Main function to start the bot."""
    print("🚀 Starting Central Square Foundation Teacher Training Bot...")
    
    if not initialize_rag_system():
        print("❌ Failed to initialize RAG system. Bot cannot start.")
        return
    
    print("✅ Bot is ready to serve teachers!")
    print("🔄 Starting bot polling...")
    
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        print(f"❌ Bot polling error: {e}")

if __name__ == "__main__":
    main()