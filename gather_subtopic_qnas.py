import os
import json
from pathlib import Path

def merge_qna_jsons(folder_path, output_path="merged_qna.json"):
    """
    Merges multiple JSON files (each containing Q&A pairs) from a folder into one.
    Each JSON filename (snake_case) becomes the category name in Title Case.
    
    Args:
        folder_path (str or Path): Folder containing the category JSON files.
        output_path (str or Path): Output path for the merged JSON file.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"The folder '{folder}' does not exist or is not a directory.")

    merged_data = []

    for json_file in folder.glob("*.json"):
        category_snake = json_file.stem  # e.g., "deep_learning_fundamentals"
        category_name = " ".join(word.capitalize() for word in category_snake.split("_"))  # → "Deep Learning Fundamentals"

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"File '{json_file}' does not contain a list of Q&A items.")

            for item in data:
                merged_data.append({
                    "question": item.get("question", "").strip(),
                    "answer_key": item.get("answer", "").strip(),
                    "category": category_name
                })

        except Exception as e:
            print(f"Skipping {json_file.name} due to error: {e}")

    # Save combined JSON
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"Merged {len(merged_data)} questions from {len(list(folder.glob('*.json')))} files into '{output_path}'.")


#merge_qna_jsons("subtopic_qnas/DL_fundamentals", "qna.json")
#merge_qna_jsons("subtopic_qnas/Computer_Vision", "topics/Computer_Vision/qna.json")
#merge_qna_jsons("subtopic_qnas/Sequence_Models", "topics/Sequence_Models/qna.json")
merge_qna_jsons("subtopic_qnas/NLP_1", "topics/NLP_1/qna.json")