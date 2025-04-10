from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load text from a file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Compare resume to job descriptions
def compare_resume_to_jobs(resume_text, job_texts):
    texts = [resume_text] + job_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores

# Suggest keywords to add
def suggest_keywords(resume_text, job_text):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    missing = job_words - resume_words
    return list(missing)[:10]  # Top 10 missing

# MAIN EXECUTION
if __name__ == "__main__":
    resume = load_text("resume.txt")
    job1 = load_text("job1.txt")
    job2 = load_text("job2.txt")

    scores = compare_resume_to_jobs(resume, [job1, job2])

    print(f"\n🔍 Resume Match Scores:")
    print(f"Job 1: {round(scores[0]*100, 2)}% match")
    print(f"Job 2: {round(scores[1]*100, 2)}% match")

    print("\n💡 Suggested Keywords to Add for Job 1:")
    print(suggest_keywords(resume, job1))
