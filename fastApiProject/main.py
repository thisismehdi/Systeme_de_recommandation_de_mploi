from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Text, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import declarative_base, Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Replace these values with your MySQL connection details
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/testdb"

# SQLAlchemy engine creation
engine = create_engine(DATABASE_URL)

# Create FastAPI app
app = FastAPI()

# SQLAlchemy Models
Base = declarative_base()

class JobPost(Base):
    __tablename__ = "job_posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    _keywords = Column("keywords", Text)

    @hybrid_property
    def keywords(self):
        return json.loads(self._keywords) if self._keywords else []

    @keywords.setter
    def keywords(self, value):
        self._keywords = json.dumps(value)

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    _keywords = Column("keywords", Text)

    @hybrid_property
    def keywords(self):
        return json.loads(self._keywords) if self._keywords else []

    @keywords.setter
    def keywords(self, value):
        self._keywords = json.dumps(value)


# Dependency to get the database session
def get_db():
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()

# Utility Functions
def get_data_from_db(db: Session, query):
    result = db.execute(query)
    data = [item[0] for item in result.fetchall()]  # Extracting the first element from each tuple
    print("Data from database:", data)  # Add this line to inspect the data
    return data

def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    texts = []
    for item in data:
        keywords = item.keywords
        print("Keywords:", keywords)
        texts.append(' '.join(keywords))
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer




@app.get("/recommend-job/{candidate_id}")
async def recommend_job(candidate_id: int, db: Session = Depends(get_db)):
    try:
        candidate_query = select(Candidate).where(Candidate.id == candidate_id)
        candidate = get_data_from_db(db, candidate_query)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        job_query = select(JobPost)
        job_posts = get_data_from_db(db, job_query)

        job_vectors, vectorizer = vectorize_data(job_posts)
        candidate_vector = vectorizer.transform([' '.join(candidate[0].keywords)])

        similarity_scores = cosine_similarity(candidate_vector, job_vectors).flatten()
        sorted_job_indices = similarity_scores.argsort()[::-1]

        recommended_jobs = [{"id": job_posts[idx].id, "title": job_posts[idx].title, "score": similarity_scores[idx]}
                            for idx in sorted_job_indices]
        return recommended_jobs
    except Exception as e:
        print(f"Error: {e}")  # Add this line to log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/recommend-candidate/{job_id}")
async def recommend_candidate(job_id: int, db: Session = Depends(get_db)):
    try:
        job_query = select(JobPost).where(JobPost.id == job_id)
        job = get_data_from_db(db, job_query)
        if not job:
            raise HTTPException(status_code=404, detail="Job post not found")

        candidate_query = select(Candidate)
        candidates = get_data_from_db(db, candidate_query)

        candidate_vectors, vectorizer = vectorize_data(candidates)
        job_vector = vectorizer.transform([' '.join(job[0].keywords)])

        similarity_scores = cosine_similarity(job_vector, candidate_vectors).flatten()
        sorted_candidate_indices = similarity_scores.argsort()[::-1]

        recommended_candidates = [{"id": candidates[idx].id, "name": candidates[idx].name, "score": similarity_scores[idx]}
                                  for idx in sorted_candidate_indices]
        return recommended_candidates
    except Exception as e:
        print(f"Error: {e}")  # Add this line to log the error
        raise HTTPException(status_code=500, detail="Internal Server Error")
