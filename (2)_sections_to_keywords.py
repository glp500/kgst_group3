import os
import csv
import anthropic
from pathlib import Path

# ======= CONFIGURATION - MODIFY THESE VALUES =======
# Path to the input CSV file from the previous script
INPUT_CSV = ".../data/research_paper_sections.csv"
# Path to save the updated CSV with keywords
OUTPUT_CSV = ".../data/research_paper_keywords.csv"
# Anthropic API key (can also be set as environment variable)
API_KEY = "..."  # Optional: set your API key here if not using environment variable
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
    system_prompt = '''You are an expert research assistant specialized in analyzing scientific papers and extracting the most relevant keywords. Your task is to identify the 10 most significant keywords from sections of a research paper.

Guidelines for keyword extraction:

1. Focus on extracting terms that represent core concepts, methodologies, technical terminology, and main topics.
2. Prioritize specialized technical terms and domain-specific vocabulary over general words.
3. Include important named entities, frameworks, models, or theories mentioned in the text.
4. Select keywords that would be useful for categorizing or searching for this paper in a database.
5. Consider both the frequency of terms and their significance to the paper's contribution.
6. Avoid overly general words (e.g., "research", "study", "analysis") unless they have a specific meaning in this context.
7. Include multi-word terms when they represent a single concept (e.g., "machine learning", "genetic algorithm").
8. Arrange keywords in order of relevance, with most important first.
9. Provide EXACTLY 10 keywords, no more and no less.

Output format:
Provide the keywords as a comma-separated list with no additional text, explanation, or numbering. Each keyword or key phrase should be in lowercase unless it's a proper noun or acronym.

Example of correct output format:
multiagent systems, reinforcement learning, nash equilibrium, cooperation protocols, emergent behavior, policy optimization, multi-objective decision making, decentralized control, stochastic games, reward sharing

Remember to focus only on the key concepts and terminology in the provided text sections.'''

    # Combine the text sections with section labels
    combined_text = f"""ABSTRACT:
{abstract}

INTRODUCTION:
{introduction}

CONCLUSION:
{conclusion}"""

    # User message
    user_message = f"""Please identify the 10 most relevant keywords from the following research paper sections:

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
        paper['keywords'] = str(keywords)
        
        print(f"  Extracted keywords: {keywords}")
    
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
        
        print(f"Successfully created {OUTPUT_CSV} with keywords added.")
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
        print("Keyword extraction completed successfully.")
    else:
        print("Keyword extraction failed.")