import os
import anthropic
import PyPDF2
import csv
import re
import glob
from pathlib import Path

# ======= CONFIGURATION - MODIFY THESE VALUES =======
# Path to a PDF file or folder containing PDF files
INPUT_PATH = ".../pdf_files"  
# Path to output CSV file
OUTPUT_CSV = ".../output_sections.csv"  
# Anthropic API key (can also be set as environment variable)
API_KEY = "api key"  # Optional: set your API key here if not using environment variable
# ===================================================

# Initialize the Anthropic client (use API_KEY if provided, otherwise use environment variable)
client = anthropic.Anthropic(
    api_key=API_KEY if API_KEY else os.environ.get("ANTHROPIC_API_KEY")
)

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_sections_with_claude(pdf_text):
    """Send PDF content to Claude with specific instructions to extract sections."""
    
    # System prompt for Claude - Using single quotes for the outer string to avoid issues with nested triple quotes
    system_prompt = '''You are an AI assistant specialized in extracting key information from scientific research papers. Your task is to analyze the content of a PDF document that has been converted to text and extract specific sections: the title, abstract, introduction, and conclusion.

Your goal is to carefully read through the document and extract the following:

1. Title: The main title of the research paper.
2. Abstract: A brief summary of the research, usually found at the beginning of the document.
3. Introduction: Typically follows the abstract, providing background information, research questions, and the study's purpose.
4. Conclusion: Usually found near the end of the document, summarizing the main findings and implications of the research.

Follow these steps to complete the task:

1. Read through the entire document carefully.
2. Identify and extract the title of the paper.
3. Locate the start and end of each required section (Abstract, Introduction, and Conclusion).
4. Extract the text of each section, preserving its original formatting as much as possible.
5. If a section is not clearly labeled, look for content that matches the typical characteristics of that section.
6. If you cannot find a particular section, use the placeholder text: "[Section not found in the document]"

Before providing your final output, wrap your thought process in <extraction_process> tags. In this section:
- Write down specific quotes that indicate the start and end of each section.
- Note any challenges in identifying sections and how you resolve them.
- Count the number of paragraphs in each section to ensure completeness.
This will help ensure a thorough and accurate extraction of the required information. It's OK for this section to be quite long.

Format your final output EXACTLY as follows (keep the exact format with the triple quotes):

<extracted_sections>
title = """
[Extracted title here]
"""

abstract = """
[Extracted abstract text here]
"""

introduction = """
[Extracted introduction text here]
"""

conclusion = """
[Extracted conclusion text here]
"""
</extracted_sections>

Remember, your final output should only include the <extracted_sections> block with the four sections formatted exactly as shown above. Do not include any additional explanations or comments outside of this block in your final output.'''

    # Create the user message with the PDF content
    user_message = f"""Here is the content of the PDF document:

<pdf_content>
{pdf_text}
</pdf_content>"""

    try:
        # Make the API call to Claude
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        # Handle different response formats based on the Anthropic SDK version
        if hasattr(message, 'content'):
            # For newer Anthropic SDK (>=0.8.0)
            if isinstance(message.content, list):
                # Content is a list of content blocks (TextBlock objects)
                text_content = ""
                for block in message.content:
                    # Check if the block has a 'type' attribute and if it's 'text'
                    if hasattr(block, 'type') and block.type == 'text':
                        # Access the text directly as an attribute
                        if hasattr(block, 'text'):
                            text_content += block.text
                return text_content
            else:
                # Content is a string (older SDK versions)
                return message.content
        else:
            print("API response has unexpected format")
            print(f"Message attributes: {dir(message)}")
            if hasattr(message, '__dict__'):
                print(f"Message dict: {message.__dict__}")
            return None
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        print(f"Exception details: {str(e)}")
        return None

def parse_extracted_sections(response):
    """Parse the response from Claude to extract the individual sections."""
    if not response:
        print("No response received from Claude API")
        return None
    
    if not isinstance(response, str):
        print(f"Unexpected response type: {type(response)}. Expected string.")
        return None
    
    # Extract the content between the extracted_sections tags
    match = re.search(r'<extracted_sections>(.*?)</extracted_sections>', response, re.DOTALL)
    if not match:
        print("Could not find extracted_sections in the response")
        print("Response preview (first 200 chars):", response[:200])
        return None
    
    sections_text = match.group(1)
    
    try:
        # Extract each section using more robust regex patterns
        # Looking for content between triple quotes with non-greedy matching
        title_pattern = r'title\s*=\s*"""(.*?)"""'
        abstract_pattern = r'abstract\s*=\s*"""(.*?)"""'
        intro_pattern = r'introduction\s*=\s*"""(.*?)"""'
        conclusion_pattern = r'conclusion\s*=\s*"""(.*?)"""'
        
        title_match = re.search(title_pattern, sections_text, re.DOTALL)
        abstract_match = re.search(abstract_pattern, sections_text, re.DOTALL)
        intro_match = re.search(intro_pattern, sections_text, re.DOTALL)
        conclusion_match = re.search(conclusion_pattern, sections_text, re.DOTALL)
        
        # Create a dictionary with the extracted sections
        sections = {
            'title': title_match.group(1).strip() if title_match else "[Title not found]",
            'abstract': abstract_match.group(1).strip() if abstract_match else "[Abstract not found]",
            'introduction': intro_match.group(1).strip() if intro_match else "[Introduction not found]",
            'conclusion': conclusion_match.group(1).strip() if conclusion_match else "[Conclusion not found]"
        }
        
        return sections
    except Exception as e:
        print(f"Error parsing extracted sections: {e}")
        print(f"Raw text from extracted sections (first 200 chars):\n{sections_text[:200]}")
        return None

def process_pdf_folder(folder_path, output_csv):
    """Process all PDFs in a folder and save the extracted sections to a CSV file."""
    # Get all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    # Create the CSV file and write the header
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'title', 'abstract', 'introduction', 'conclusion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, 
                                quoting=csv.QUOTE_ALL,  # Quote all fields
                                quotechar='"',  # Use double quotes
                                escapechar='\\')  # Use backslash to escape quotes
        writer.writeheader()
        
        # Process each PDF file
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            
            # Extract text from the PDF
            pdf_text = extract_text_from_pdf(pdf_file)
            if not pdf_text:
                print(f"  Skipping {pdf_file} - could not extract text")
                continue
            
            # Extract sections using Claude
            claude_response = extract_sections_with_claude(pdf_text)
            
            # Parse the response to get individual sections
            sections = parse_extracted_sections(claude_response)
            if not sections:
                print(f"  Skipping {pdf_file} - could not parse sections from Claude's response")
                continue
            
            # Add the filename to the sections dictionary
            sections['filename'] = os.path.basename(pdf_file)
            
            # Write the sections to the CSV file
            writer.writerow(sections)
            
            print(f"  Successfully processed {pdf_file}")
    
    print(f"All PDF files processed. Results saved to {output_csv}")

def process_single_pdf(pdf_path, output_csv):
    """Process a single PDF file and save the extracted sections to a CSV file."""
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print(f"Could not extract text from {pdf_path}")
        return
    
    # Extract sections using Claude
    claude_response = extract_sections_with_claude(pdf_text)
    
    # Parse the response to get individual sections
    sections = parse_extracted_sections(claude_response)
    if not sections:
        print(f"Could not parse sections from Claude's response for {pdf_path}")
        return
    
    # Create the CSV file and write the header
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'title', 'abstract', 'introduction', 'conclusion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                               quoting=csv.QUOTE_ALL,  # Quote all fields
                               quotechar='"',  # Use double quotes
                               escapechar='\\')  # Use backslash to escape quotes
        
        if not file_exists:
            writer.writeheader()
        
        # Add the filename to the sections dictionary
        sections['filename'] = os.path.basename(pdf_path)
        
        # Write the sections to the CSV file
        writer.writerow(sections)
    
    print(f"Successfully processed {pdf_path}. Results saved to {output_csv}")

if __name__ == "__main__":
    # The paths are now defined at the top of the script
    
    # Check if the API key is set
    if not API_KEY and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: No API key found. Either set the ANTHROPIC_API_KEY environment variable or update the API_KEY variable in this script.")
        exit(1)
    
    # Process input (either a single file or a folder)
    input_path = Path(INPUT_PATH)
    
    print(f"Using input path: {INPUT_PATH}")
    print(f"Using output CSV: {OUTPUT_CSV}")
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        process_single_pdf(str(input_path), OUTPUT_CSV)
    elif input_path.is_dir():
        process_pdf_folder(str(input_path), OUTPUT_CSV)
    else:
        print(f"Error: Input path {INPUT_PATH} is not a valid PDF file or directory")