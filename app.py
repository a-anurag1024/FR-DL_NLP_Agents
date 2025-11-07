import streamlit as st
import os
import json
from pathlib import Path
import re

# --- CONFIGURATION ---
TOPICS_DIR = Path("topics")
UPDATED_DIR = Path("updated_qnas")
UPDATED_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Topic Review App", layout="wide")

# --- HELPER FUNCTIONS ---

def get_topics():
    """Return a list of topic names (subdirectory names under topics/)"""
    return [d.name for d in TOPICS_DIR.iterdir() if d.is_dir()]

def load_review_markdown(topic):
    """Load markdown review file for a topic"""
    review_path = TOPICS_DIR / topic / "review.md"
    if review_path.exists():
        with open(review_path, "r", encoding="utf-8") as f:
            return f.read()
    return "_No review markdown found for this topic._"

def load_qna(topic):
    """Load qna.json for a topic (check updated_qnas first)"""
    updated_path = UPDATED_DIR / f"{topic}.json"
    original_path = TOPICS_DIR / topic / "qna.json"

    if updated_path.exists():
        path = updated_path
    elif original_path.exists():
        path = original_path
    else:
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_updated_qna(topic, qna_data):
    """Save updated qna.json into updated_qnas directory"""
    updated_path = UPDATED_DIR / f"{topic}.json"
    with open(updated_path, "w", encoding="utf-8") as f:
        json.dump(qna_data, f, indent=4, ensure_ascii=False)

def ensure_updated_fields(qna_data):
    """Add additional_notes and marked_for_review fields if missing"""
    for q in qna_data:
        if "additional_notes" not in q:
            q["additional_notes"] = ""
        if "marked_for_review" not in q:
            q["marked_for_review"] = False
    return qna_data

def init_session_state():
    """Initialize session state for tracking changes"""
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    if 'qna_data' not in st.session_state:
        st.session_state.qna_data = []

# --- UI LOGIC ---


init_session_state()

topics = get_topics()
if not topics:
    st.warning("No topics found in the 'topics/' directory.")
    st.stop()

st.sidebar.title("📚 Topics")
selected_topic = st.sidebar.radio(
    "Choose a topic:",
    topics,
    label_visibility="collapsed"
)

# Load content
review_text = load_review_markdown(selected_topic)

# Load QnA data and handle topic switching
if st.session_state.current_topic != selected_topic:
    # Topic changed, load new data
    st.session_state.qna_data = ensure_updated_fields(load_qna(selected_topic))
    st.session_state.current_topic = selected_topic

qna_data = st.session_state.qna_data

# Create tabs for Review Sheet and Q&A
tab1, tab2 = st.tabs(["📘 Review Sheet", "❓ Q&A Practice"])

with tab1:
    st.header(f"{selected_topic.capitalize()} Review Sheet")
    # Streamlit automatically renders LaTeX in markdown with $ and $$
    st.markdown(review_text)

with tab2:
    st.header("Comprehensive Q&A Review")
    
    # Filter option
    col1, col2 = st.columns([1, 4])
    with col1:
        show_filter = st.selectbox(
            "Filter Questions:",
            ["All Questions", "Marked for Review Only"],
            key="filter_select"
        )
    
    # Filter the questions based on selection
    if show_filter == "Marked for Review Only":
        filtered_qna = [q for q in qna_data if q.get("marked_for_review", False)]
        if not filtered_qna:
            st.info("No questions marked for review yet. Mark questions below to see them here!")
    else:
        filtered_qna = qna_data
    
    # Display statistics
    total_questions = len(qna_data)
    marked_count = sum(1 for q in qna_data if q.get("marked_for_review", False))
    st.caption(f"Showing {len(filtered_qna)} of {total_questions} questions ({marked_count} marked for review)")
    
    # Group questions by category
    categories = {}
    for idx, q in enumerate(qna_data):
        category = q.get('category', 'Uncategorized')
        if category not in categories:
            categories[category] = []
        categories[category].append((idx, q))
    
    # Display questions grouped by category
    for category, questions in categories.items():
        # Check if this category has any questions to display based on filter
        visible_questions = [
            (idx, q) for idx, q in questions 
            if show_filter == "All Questions" or q.get("marked_for_review", False)
        ]
        
        if not visible_questions:
            continue
            
        st.subheader(f"📂 {category}")
        
        for idx, q in visible_questions:
            with st.expander(f"Q{idx+1}: {q['question']}" + (" ⭐" if q.get("marked_for_review", False) else "")):
                if st.button(f"Show Answer for Q{idx+1}", key=f"show_{idx}"):
                    st.info(q["answer_key"])
                
                note_key = f"note_{idx}"
                review_key = f"review_{idx}"

                # Text area for additional notes
                new_note = st.text_area("📝 Add additional notes:", value=q.get("additional_notes", ""), key=note_key)
                if new_note != q.get("additional_notes", ""):
                    q["additional_notes"] = new_note
                    save_updated_qna(selected_topic, qna_data)

                # Checkbox for "mark for review"
                new_marked = st.checkbox("⭐ Mark for Review", value=q.get("marked_for_review", False), key=review_key)
                if new_marked != q.get("marked_for_review", False):
                    q["marked_for_review"] = new_marked
                    save_updated_qna(selected_topic, qna_data)

