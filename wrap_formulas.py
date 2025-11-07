import re

input_path = "topics/DL_fundamentals/review.md"
output_path = "topics/DL_fundamentals/review.md"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Function to recursively match balanced parentheses
def get_balanced_formula(text, start_pos):
    """Extract formula with balanced parentheses starting from start_pos"""
    count = 1
    i = start_pos + 1
    while i < len(text) and count > 0:
        if text[i] == '(' and (i == 0 or text[i-1] != '\\'):
            count += 1
        elif text[i] == ')' and (i == 0 or text[i-1] != '\\'):
            count -= 1
        i += 1
    return i if count == 0 else -1

# Replace formulas with balanced parentheses
def replace_formulas(text):
    result = []
    i = 0
    
    while i < len(text):
        # Look for opening parenthesis
        if text[i] == '(' and (i == 0 or text[i-1] != '\\'):
            # Find the matching closing parenthesis
            end_pos = get_balanced_formula(text, i)
            
            if end_pos != -1:
                formula = text[i+1:end_pos-1]
                
                # Check if this looks like a LaTeX formula
                # (contains backslashes, or math symbols like ^, _, {, }, [, ])
                if any(char in formula for char in ['\\', '^', '_', '{', '}']) or \
                   re.search(r'[a-zA-Z]+_[a-zA-Z0-9]+|[a-zA-Z]+\^', formula):
                    # Wrap the entire formula including parentheses
                    result.append(f"$$( {formula} )$$")
                    i = end_pos
                    continue
        
        result.append(text[i])
        i += 1
    
    return ''.join(result)

# Replace all formulas
new_content = replace_formulas(content)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ All formulas wrapped in $$( ... )$$")
print(f"Formula wrapping completed successfully!")
print(f"Updated file: {output_path}")