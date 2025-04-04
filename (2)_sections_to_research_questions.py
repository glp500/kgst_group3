import os
import csv
import anthropic
from pathlib import Path

# ======= CONFIGURATION - MODIFY THESE VALUES =======
# Path to the input CSV file from the previous script
INPUT_CSV = "/Volumes/Extreme SSD/Python_Projects/KG_Project/data/research_paper_breakdown_2.csv"
# Path to save the updated CSV with keywords
OUTPUT_CSV = "/Volumes/Extreme SSD/Python_Projects/KG_Project/data/research_paper_rq_2.csv"
# Anthropic API key (can also be set as environment variable)
API_KEY = "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"  # Optional: set your API key here if not using environment variable
# ===================================================

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key=API_KEY if API_KEY else os.environ.get("ANTHROPIC_API_KEY")
)

def extract_keywords(abstract, introduction, conclusion):
    """
    Extract the 10 most relevant keywords from the paper content using Claude API.
    
    Args:
        abstract (str): The paper's abstract
        introduction (str): The paper's introduction
        conclusion (str): The paper's conclusion
        
    Returns:
        str: A comma-separated list of 10 keywords
    """
    
    # System prompt for Claude
    system_prompt = '''You are an expert research assistant specialized in analyzing scientific papers and extracting research questions. Your task is to identify the primary research question(s) addressed in the paper.

Guidelines for research question extraction:
1. Focus on identifying the central question(s) that the research aims to answer.
2. Look for explicit statements of research questions, hypotheses, or objectives, typically found in the abstract, introduction, or methodology sections.
3. If the research question is not explicitly stated, synthesize it from the paper's stated goals, hypotheses, and conclusions.
4. Distinguish between the primary research question and any secondary or subordinate questions.
5. Pay attention to how the authors frame their contribution to the field.
6. Consider both theoretical questions (advancing understanding) and practical questions (solving problems).
7. Extract the question in its complete form, preserving the specific variables, populations, or phenomena being studied.
8. If multiple equally important research questions exist, include all of them (up to 3).
9. Ensure the extracted question captures the scope and specificity of the research.

Output format:
Provide the research question(s) as complete sentences with proper capitalization and punctuation. If multiple questions are identified, separate them with line breaks. Include no additional text, explanation, or numbering.

Example of correct output format:
How do multilayer perceptrons perform compared to convolutional neural networks when classifying medical images with limited training data?

Remember to focus on capturing the essence of what the researchers sought to investigate, not just what they found.'''

    # Combine the text sections with section labels
    combined_text = f"""ABSTRACT:
{abstract}

INTRODUCTION:
{introduction}

CONCLUSION:
{conclusion}"""

    # User message
    user_message = f"""Please identify the research from the following research paper sections:

{combined_text}"""

    try:
        # Make the API call to Claude
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=150,  # Limited as we only need a short list of keywords
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        # Handle different response formats based on the Anthropic SDK version
        if hasattr(message, 'content'):
            if isinstance(message.content, list):
                # Content is a list of content blocks
                text_content = ""
                for block in message.content:
                    if hasattr(block, 'type') and block.type == 'text':
                        if hasattr(block, 'text'):
                            text_content += block.text
                return text_content.strip()
            else:
                # Content is a string (older SDK versions)
                return message.content.strip()
        else:
            print("API response has unexpected format")
            return "Error extracting keywords"
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return "Error extracting keywords"

def process_csv_file():
    """Process the input CSV file, extract keywords for each paper, and create a new CSV with keywords."""
    # Check if the input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV file {INPUT_CSV} does not exist.")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paper_data = []
    
    # Read the input CSV
    print(f"Reading input CSV: {INPUT_CSV}")
    try:
        with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Validate that the required columns exist
            required_columns = ['filename', 'title', 'abstract', 'introduction', 'conclusion']
            if not all(col in reader.fieldnames for col in required_columns):
                missing_cols = [col for col in required_columns if col not in reader.fieldnames]
                print(f"Error: Input CSV is missing required columns: {', '.join(missing_cols)}")
                return False
            
            # Read all rows into memory
            paper_data = list(reader)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return False
    
    # Process each paper and extract keywords
    total_papers = len(paper_data)
    print(f"Found {total_papers} papers to process.")
    
    for i, paper in enumerate(paper_data, 1):
        print(f"Processing paper {i} of {total_papers}: {paper['filename']}")
        
        # Extract keywords using Claude
        keywords = extract_keywords(
            paper['abstract'], 
            paper['introduction'], 
            paper['conclusion']
        )
        
        # Add keywords to the paper data
        paper['research_question'] = str(keywords)
        
        print(f"  Extracted research_question: {keywords}")
    
    # Write the updated data to a new CSV
    print(f"Writing output CSV: {OUTPUT_CSV}")
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            # Create fieldnames with the new 'keywords' column
            fieldnames = list(paper_data[0].keys())
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                   quoting=csv.QUOTE_ALL,  # Quote all fields
                                   quotechar='"',  # Use double quotes
                                   escapechar='\\')  # Use backslash to escape quotes
            
            writer.writeheader()
            writer.writerows(paper_data)
        
        print(f"Successfully created {OUTPUT_CSV} with research questions added.")
        return True
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        return False

if __name__ == "__main__":
    # Check if the API key is set
    if not API_KEY and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: No API key found. Either set the ANTHROPIC_API_KEY environment variable or update the API_KEY variable in this script.")
        exit(1)
    
    # Process the CSV file
    success = process_csv_file()
    
    if success:
        print("Research question extraction completed successfully.")
    else:
        print("Research question extraction failed.")