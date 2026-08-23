import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

from resume_parser import extract_text
from matcher import extract_structured_data, score_match
from category_classifier import predict_category
from db import init_db, save_candidate, get_all_candidates

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "job_description is required"}), 400

    resume_file = request.files["resume"]
    file_path = os.path.join(UPLOAD_DIR, resume_file.filename)
    resume_file.save(file_path)

    resume_text = extract_text(file_path)
    resume_data = extract_structured_data(resume_text)
    result = score_match(resume_data, job_description)

    # Free, local, non-LLM classifier — separate model from the GPT calls above.
    predicted_category, category_confidence = predict_category(resume_text)

    save_candidate(
        filename=resume_file.filename,
        resume_data=resume_data,
        job_description=job_description,
        score=result["score"],
        justification=result["justification"],
        predicted_category=predicted_category,
        category_confidence=category_confidence,
    )

    return jsonify({
        "filename": resume_file.filename,
        "extracted": resume_data,
        "score": result["score"],
        "justification": result["justification"],
        "predicted_category": predicted_category,
        "category_confidence": category_confidence,
    })


@app.route("/candidates", methods=["GET"])
def list_candidates():
    min_score = int(request.args.get("min_score", 0))
    return jsonify(get_all_candidates(min_score))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
