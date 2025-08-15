# pdf_qa_tool.py
import pdfplumber
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ===================== CONFIG =====================
OPENAI_API_KEY

MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 500
TOP_K = 3
# ===================================================

client = OpenAI(api_key=OPENAI_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

chunks = []
chunk_embeddings = []

# 1️⃣ Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 2️⃣ Split into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 3️⃣ Process PDF
def process_pdf(pdf_file):
    global chunks, chunk_embeddings
    text = extract_text_from_pdf(pdf_file.name)
    chunks = chunk_text(text, CHUNK_SIZE)
    chunk_embeddings = embedder.encode(chunks)
    return "✅ PDF processed successfully.", gr.update(visible=False), gr.update(visible=True)

# 4️⃣ Search chunks
def search_relevant_chunks(question, top_k=3):
    q_embedding = embedder.encode([question])
    similarities = cosine_similarity(q_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# 5️⃣ Ask with enhancement
def ask_question(question):
    if not chunks:
        return "⚠️ Please upload and process a PDF first."
    
    relevant_chunks = search_relevant_chunks(question, TOP_K)
    book_answer = "\n".join(relevant_chunks)

    prompt = f"""
    Here is some content from a textbook or PDF that may answer the question:

    {book_answer}

    Task: Based on the above, give a clear, complete, and well-structured answer to the following question, enhancing clarity but keeping factual accuracy from the text.

    Question: {question}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 6️⃣ Close PDF
def close_pdf():
    global chunks, chunk_embeddings
    chunks = []
    chunk_embeddings = []
    return "", gr.update(visible=True), gr.update(visible=False)

# 7️⃣ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF Question Answering Tool")

    # Initial view (before upload)
    with gr.Column(visible=True) as upload_column:
        pdf_input = gr.File(type="filepath", label="Upload your PDF")
        pdf_status = gr.Textbox(label="Status", interactive=False)

    # Q&A view (after upload)
    with gr.Column(visible=False) as qa_column:
        status_box = gr.Textbox(label="Status", interactive=False)
        close_btn = gr.Button("Close PDF")
        question_input = gr.Textbox(label="Ask a Question")
        ask_btn = gr.Button("Enter")
        answer_output = gr.Textbox(label="Answer", lines=8)

    # Events
    pdf_input.upload(process_pdf, pdf_input, [pdf_status, upload_column, qa_column])
    ask_btn.click(ask_question, question_input, answer_output)
    question_input.submit(ask_question, question_input, answer_output)
    close_btn.click(close_pdf, None, [pdf_status, upload_column, qa_column])

demo.launch()

