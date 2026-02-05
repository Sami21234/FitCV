from flask import Flask, request, render_template
import os
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    # 1. Check file
    if "resume" not in request.files:
        return "No file part"

    file = request.files["resume"]

    if file.filename == "":
        return "No selected file"

    # 2. Get Job Description text
    jd_text = request.form.get("job_description", "")

    # 3. Save file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 4. Extract resume text
    resume_text = ""

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        for page in reader.pages:
            resume_text += page.extract_text() or ""

    elif filename.lower().endswith(".docx"):
        doc = Document(filepath)
        for para in doc.paragraphs:
            resume_text += para.text + "\n"

    else:
        return "Unsupported file format. Please upload PDF or DOCX."

    # 6. Create embeddings and compute similarity
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)

    similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    match_percentage = round(similarity * 100, 2)

    # 7. Show result
    return f"""
    <h2>FitCV Result</h2>
    <p><strong>Match Score:</strong> {match_percentage}%</p>
    <hr>
    <h3>Resume Text (preview):</h3>
    <pre>{resume_text[:2000]}</pre>
    <hr>
    <h3>Job Description Text:</h3>
    <pre>{jd_text}</pre>
    """


if __name__ == "__main__":
    app.run(debug=True)
