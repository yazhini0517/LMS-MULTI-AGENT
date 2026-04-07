import os, re, warnings, logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline  # ← needed for LLM

# -----------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port="55432",
        database="lms_db",
        user="postgres",
        password="my_password"
    )

# -----------------------------------------------
# FETCH DATA FROM POSTGRESQL
# -----------------------------------------------
print("📦 Loading data from lms_db...")
conn = get_connection()
cursor = conn.cursor()
documents = []

queries = [
    """SELECT CONCAT(
        'Organization ID: ', org_id,
        '. Organization Name: ', org_name,
        '. Industry: ', COALESCE(industry, 'N/A'),
        '. Organization Size: ', COALESCE(organization_size, 'N/A'),
        '. Address: ', COALESCE(address, 'N/A')
    ) FROM organizations;""",

    """SELECT CONCAT(
        'User ID: ', u.user_id,
        '. Username: ', u.username,
        '. Email: ', u.email,
        '. Phone: ', COALESCE(u.phone_number, 'N/A'),
        '. Role: ', COALESCE(u.role, 'N/A'),
        '. Organization ID: ', COALESCE(u.org_id::TEXT, 'N/A'),
        '. Organization: ', COALESCE(o.org_name, 'N/A'),
        '. Address: ', COALESCE(u.address, 'N/A')
    ) FROM users u
    LEFT JOIN organizations o ON u.org_id = o.org_id;""",

    """SELECT CONCAT(
        'Team ID: ', t.team_id,
        '. Team Name: ', t.team_name,
        '. Organization ID: ', COALESCE(t.org_id::TEXT, 'N/A'),
        '. Organization: ', COALESCE(o.org_name, 'N/A'),
        '. Team Lead ID: ', COALESCE(t.team_lead_id::TEXT, 'N/A'),
        '. Team Lead: ', COALESCE(u.username, 'N/A')
    ) FROM teams t
    LEFT JOIN organizations o ON t.org_id = o.org_id
    LEFT JOIN users u ON t.team_lead_id = u.user_id;""",

    """SELECT CONCAT(
        'Project ID: ', p.project_id,
        '. Project Name: ', p.project_name,
        '. Description: ', COALESCE(p.description, 'N/A'),
        '. Organization: ', COALESCE(o.org_name, 'N/A'),
        '. Team: ', COALESCE(t.team_name, 'N/A'),
        '. Project Manager: ', COALESCE(u.username, 'N/A'),
        '. Methodology: ', COALESCE(p.methodology, 'N/A'),
        '. Budget: ', COALESCE(p.budget::TEXT, 'N/A'),
        '. Start Date: ', COALESCE(p.start_date::TEXT, 'N/A'),
        '. End Date: ', COALESCE(p.end_date::TEXT, 'N/A'),
        '. Actual End Date: ', COALESCE(p.actual_end_date::TEXT, 'N/A'),
        '. Status: ', COALESCE(p.status, 'N/A'),
        '. Priority: ', COALESCE(p.priority, 'N/A')
    ) FROM projects p
    LEFT JOIN organizations o ON p.org_id = o.org_id
    LEFT JOIN teams t ON p.team_id = t.team_id
    LEFT JOIN users u ON p.project_manager_id = u.user_id;""",

    """SELECT CONCAT(
        'Milestone ID: ', pm.milestone_id,
        '. Milestone Name: ', pm.milestone_name,
        '. Project ID: ', pm.project_id,
        '. Project: ', COALESCE(p.project_name, 'N/A'),
        '. Description: ', COALESCE(pm.description, 'N/A'),
        '. Planned Date: ', COALESCE(pm.planned_date::TEXT, 'N/A'),
        '. Actual Date: ', COALESCE(pm.actual_date::TEXT, 'N/A'),
        '. Status: ', COALESCE(pm.status, 'N/A')
    ) FROM project_milestones pm
    LEFT JOIN projects p ON pm.project_id = p.project_id;""",
]

for q in queries:
    try:
        cursor.execute(q)
        rows = cursor.fetchall()
        documents.extend([row[0] for row in rows if row[0]])
    except Exception as e:
        print(f"⚠️ Skipping query: {e}")
        conn.rollback()

cursor.close()
conn.close()
print(f"✅ Loaded {len(documents)} records from lms_db")

# -----------------------------------------------
# BUILD VECTOR STORE
# -----------------------------------------------
docs = [Document(page_content=t) for t in documents]
splits = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
).split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = InMemoryVectorStore(embedding=embedding_model)
vector_store.add_documents(splits)
print(f"✅ Vector store ready with {len(splits)} chunks")

SCORE_THRESHOLD = 0.60

# -----------------------------------------------
# SEARCH FUNCTIONS
# -----------------------------------------------
def search_materials(query: str, top_k: int = 5):
    raw = vector_store.similarity_search_with_score(query, k=top_k)
    materials = []
    for doc, score in raw:
        dist = 1.0 - score
        if dist < SCORE_THRESHOLD:
            materials.append({
                "content": doc.page_content,
                "similarity": round(score, 3),
                "course_title": "LMS Project Planning Database"
            })
    return materials[:top_k]

def format_context(materials: list) -> str:
    if not materials:
        return "No relevant database records found."
    parts = ["Relevant LMS Database Records:\n"]
    for i, m in enumerate(materials, 1):
        parts.append(
            f"{i}. {m.get('content', '')[:300]}\n"
            f"   Relevance: {m.get('similarity', 0):.2f}\n"
        )
    return "\n".join(parts)

# -----------------------------------------------
# ✅ LOAD LLM  ← ADDED HERE (after RAG is ready)
# -----------------------------------------------
print("🔄 Loading LLM...")
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1,
    max_new_tokens=256,
    do_sample=False,
    temperature=1.0,
    truncation=True,
)
print("✅ LLM Ready")