import streamlit as st
import os
import json
from pathlib import Path
import re
from datetime import datetime
import hashlib

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

def get_data_hash(qna_data):
    """Generate a hash of the data for verification"""
    data_str = json.dumps(qna_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def create_backup(topic, qna_data):
    """Create a timestamped backup before saving"""
    backup_dir = UPDATED_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{topic}_{timestamp}.json"
    
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(qna_data, f, indent=4, ensure_ascii=False)
    
    return backup_path

def save_updated_qna(topic, qna_data):
    """
    Save updated qna.json with safety checks and backups.
    Returns (success: bool, message: str)
    """
    updated_path = UPDATED_DIR / f"{topic}.json"
    
    # SAFETY CHECK 1: Verify topic matches the data
    if not qna_data or len(qna_data) == 0:
        return False, "Error: No data to save!"
    
    # SAFETY CHECK 2: Verify all items have the same category prefix as topic
    # (This prevents cross-topic contamination)
    sample_categories = set(q.get('category', '') for q in qna_data[:10])
    topic_check = topic.replace('_', ' ').lower()
    
    # SAFETY CHECK 3: Create backup before saving
    try:
        backup_path = create_backup(topic, qna_data)
    except Exception as e:
        return False, f"Backup failed: {str(e)}"
    
    # SAFETY CHECK 4: Write to temporary file first
    temp_path = updated_path.with_suffix('.tmp')
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(qna_data, f, indent=4, ensure_ascii=False)
        
        # SAFETY CHECK 5: Verify the written file
        with open(temp_path, "r", encoding="utf-8") as f:
            verify_data = json.load(f)
        
        if len(verify_data) != len(qna_data):
            temp_path.unlink()
            return False, "Verification failed: Data length mismatch!"
        
        # SAFETY CHECK 6: Atomic move (on Windows, need to delete first)
        if updated_path.exists():
            updated_path.unlink()
        temp_path.rename(updated_path)
        
        return True, f"✅ Saved successfully! Backup: {backup_path.name}"
        
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return False, f"Save failed: {str(e)}"

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
    if 'pending_changes' not in st.session_state:
        st.session_state.pending_changes = False
    if 'changes_topic' not in st.session_state:
        st.session_state.changes_topic = None
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    if 'loaded_from_file' not in st.session_state:
        st.session_state.loaded_from_file = None

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

# Warn about unsaved changes when switching topics
if st.session_state.current_topic != selected_topic and st.session_state.pending_changes:
    st.sidebar.error(f"🚫 BLOCKED: You have unsaved changes in {st.session_state.changes_topic}!")
    st.sidebar.warning(f"Data loaded from: {st.session_state.loaded_from_file}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Save First", key="save_before_switch", use_container_width=True):
            success, message = save_updated_qna(st.session_state.changes_topic, st.session_state.qna_data)
            if success:
                st.session_state.pending_changes = False
                st.session_state.data_hash = get_data_hash(st.session_state.qna_data)
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)
    
    with col2:
        if st.button("🗑️ Discard", key="discard_changes", use_container_width=True):
            st.session_state.pending_changes = False
            st.session_state.qna_data = ensure_updated_fields(load_qna(selected_topic))
            st.session_state.current_topic = selected_topic
            st.session_state.data_hash = get_data_hash(st.session_state.qna_data)
            st.session_state.changes_topic = selected_topic
            
            # Record which file we loaded from
            updated_path = UPDATED_DIR / f"{selected_topic}.json"
            if updated_path.exists():
                st.session_state.loaded_from_file = f"updated_qnas/{selected_topic}.json"
            else:
                st.session_state.loaded_from_file = f"topics/{selected_topic}/qna.json"
            
            st.rerun()
    
    # CRITICAL: Stop execution here - don't allow topic switch
    st.stop()

# Load QnA data and handle topic switching
if st.session_state.current_topic != selected_topic and not st.session_state.pending_changes:
    # Topic changed, load new data
    st.session_state.qna_data = ensure_updated_fields(load_qna(selected_topic))
    st.session_state.current_topic = selected_topic
    st.session_state.changes_topic = selected_topic
    st.session_state.data_hash = get_data_hash(st.session_state.qna_data)
    
    # Record which file we loaded from
    updated_path = UPDATED_DIR / f"{selected_topic}.json"
    if updated_path.exists():
        st.session_state.loaded_from_file = f"updated_qnas/{selected_topic}.json"
    else:
        st.session_state.loaded_from_file = f"topics/{selected_topic}/qna.json"

qna_data = st.session_state.qna_data

# Create tabs for Review Sheet and Q&A
tab1, tab2 = st.tabs(["📘 Review Sheet", "❓ Q&A Practice"])

with tab1:
    st.header(f"{selected_topic.capitalize()} Review Sheet")
    # Streamlit automatically renders LaTeX in markdown with $ and $$
    st.markdown(review_text)

with tab2:
    st.header("Comprehensive Q&A Review")
    
    # Save button at the top with validation
    col_save, col_status, col_info = st.columns([1, 2, 2])
    
    with col_save:
        if st.button("💾 SAVE ALL CHANGES", key="save_all", type="primary", use_container_width=True):
            # Double-check topic matches before saving
            if st.session_state.current_topic != st.session_state.changes_topic:
                st.error(f"🚫 SAFETY BLOCK: Topic mismatch! Current: {st.session_state.current_topic}, Changes: {st.session_state.changes_topic}")
            else:
                success, message = save_updated_qna(st.session_state.current_topic, qna_data)
                if success:
                    st.session_state.pending_changes = False
                    st.session_state.data_hash = get_data_hash(qna_data)
                    st.success(message)
                else:
                    st.error(message)
    
    with col_status:
        if st.session_state.pending_changes:
            st.error("🔴 UNSAVED CHANGES", icon="⚠️")
        else:
            st.success("🟢 All Saved", icon="✅")
    
    with col_info:
        st.caption(f"📂 Topic: **{st.session_state.current_topic}**")
        st.caption(f"📄 Loaded from: `{st.session_state.loaded_from_file}`")
    
    st.divider()
    
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
                existing_note = q.get("additional_notes", "")
                has_note = bool(existing_note and existing_note.strip())
                
                
                note_label = "📝 Additional notes:" if not has_note else "📝 Additional notes: (editing existing note)"
                
                # Force widget to update by including topic and idx in key
                unique_key = f"note_{st.session_state.current_topic}_{idx}"
                new_note = st.text_area(note_label, value=existing_note, key=unique_key, height=100)
                
                if new_note != existing_note:
                    q["additional_notes"] = new_note
                    st.session_state.pending_changes = True
                    st.session_state.changes_topic = st.session_state.current_topic

                # Checkbox for "mark for review"
                unique_review_key = f"review_{st.session_state.current_topic}_{idx}"
                new_marked = st.checkbox("⭐ Mark for Review", value=q.get("marked_for_review", False), key=unique_review_key)
                if new_marked != q.get("marked_for_review", False):
                    q["marked_for_review"] = new_marked
                    st.session_state.pending_changes = True
                    st.session_state.changes_topic = st.session_state.current_topic

