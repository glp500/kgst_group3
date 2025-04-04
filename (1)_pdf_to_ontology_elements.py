import os
import anthropic
import PyPDF2
import csv
import re
import glob
import json
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

def extract_hybrid_intelligence_with_claude(pdf_text):
    """Send PDF content to Claude with specific instructions to extract hybrid intelligence elements in JSON format."""
    
    # System prompt for Claude - Using single quotes for the outer string to avoid issues with nested triple quotes
    system_prompt = '''You are an AI assistant specialized in extracting information about hybrid intelligence systems from scientific research papers. Your task is to analyze the content of a PDF document and extract specific elements related to hybrid intelligence scenarios in a structured JSON format.

Your goal is to carefully read through the document and extract the following elements with their specific structure:

1. Scenario:
   * Scenario Name (a brief 1-3 word keyword)
   * Scenario description (detailed explanation)
   * Contains (reference to the Task)

2. Task:
   * Task Name (a brief 1-3 word keyword)
   * Task Description (detailed explanation)
   * hasActors (list of Actor names)
   * usedIn (list of Capability names)

3. Actor:
   * Actor Name (a brief 1-3 word keyword)
   * Actor type (must be either "Artificial Agent" or "Human")
   * Has (list of Capability names)

4. Capability:
   * Capability Name (a brief 1-3 word keyword)
   * Description (detailed explanation)
   * hasProcessingMethod (list of Processing Method names)
   * Produces (list of Metric names)

5. Processing Method:
   * Processing Method name (a brief 1-3 word keyword)
   * Processing Method type (must be one of: "Statistical", "Symbolic", "Neuro-symbolic", or "Other")

6. Metric:
   * Metric Name (a brief 1-3 word keyword)
   * Metric Type (must be either "Qualitative" or "Quantitative")

Follow these steps to complete the task:

1. Read through the entire document carefully.
2. Identify any hybrid intelligence scenarios described in the paper.
3. For each element (Scenario, Task, Actor, etc.), extract both a concise keyword name (1-3 words) and a more detailed description.
4. Format each element according to the JSON structure specified below.
5. If you cannot find information about a particular element, use a placeholder like "Unknown" for required fields.

Before providing your final output, wrap your thought process in <extraction_process> tags. In this section:
- Write down specific quotes that indicate each element
- Note any challenges in identifying elements and how you resolve them
- Explain your reasoning for categorizing elements

Format your final output EXACTLY as follows (keep the exact format with the triple quotes):

<extracted_elements>
scenario = """
{
  "name": "Brief Scenario Name",
  "description": "Detailed scenario description...",
  "contains": ["Task Name"]
}
"""

task = """
{
  "name": "Brief Task Name",
  "description": "Detailed task description...",
  "hasActors": ["Actor Name 1", "Actor Name 2"],
  "usedIn": ["Capability Name 1", "Capability Name 2"]
}
"""

actor = """
[
  {
    "name": "Actor Name 1",
    "type": "Human",
    "has": ["Capability Name 1"]
  },
  {
    "name": "Actor Name 2",
    "type": "Artificial Agent",
    "has": ["Capability Name 2"]
  }
]
"""

capability = """
[
  {
    "name": "Capability Name 1",
    "description": "Detailed capability description...",
    "hasProcessingMethod": ["Processing Method Name 1"],
    "produces": ["Metric Name 1"]
  },
  {
    "name": "Capability Name 2",
    "description": "Detailed capability description...",
    "hasProcessingMethod": ["Processing Method Name 2"],
    "produces": ["Metric Name 2"]
  }
]
"""

processing_method = """
[
  {
    "name": "Processing Method Name 1",
    "type": "Statistical"
  },
  {
    "name": "Processing Method Name 2",
    "type": "Symbolic"
  }
]
"""

metric = """
[
  {
    "name": "Metric Name 1",
    "type": "Quantitative"
  },
  {
    "name": "Metric Name 2",
    "type": "Qualitative"
  }
]
"""
</extracted_elements>

Remember, your final output should only include the <extracted_elements> block with the six sections formatted exactly as shown above. Each section must contain valid JSON. Names should be concise (1-3 words) while descriptions can be more detailed. Make sure all relationships between elements are consistent (e.g., if Actor A has Capability B, then Capability B should exist in the capability section).'''

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

def validate_element_relationships(elements):
    """Validates that relationships between elements are consistent."""
    try:
        # Parse all JSON strings into Python objects
        scenario_obj = json.loads(elements['scenario'])
        task_obj = json.loads(elements['task'])
        actors_obj = json.loads(elements['actor'])
        capabilities_obj = json.loads(elements['capability'])
        processing_methods_obj = json.loads(elements['processing_method'])
        metrics_obj = json.loads(elements['metric'])
        
        # Create sets of names for quick lookup
        actor_names = {actor['name'] for actor in actors_obj}
        capability_names = {cap['name'] for cap in capabilities_obj}
        processing_method_names = {pm['name'] for pm in processing_methods_obj}
        metric_names = {metric['name'] for metric in metrics_obj}
        
        # Validate scenario contains references to valid tasks
        if 'contains' in scenario_obj:
            for task_name in scenario_obj['contains']:
                if task_name != task_obj['name']:
                    print(f"Warning: Scenario contains task '{task_name}' but task name is '{task_obj['name']}'")
                    # Fix the reference
                    scenario_obj['contains'] = [task_obj['name']]
        
        # Validate task references valid actors and capabilities
        if 'hasActors' in task_obj:
            for actor_name in task_obj['hasActors']:
                if actor_name not in actor_names:
                    print(f"Warning: Task references actor '{actor_name}' not found in actors list")
                    # Remove invalid reference
                    task_obj['hasActors'] = [name for name in task_obj['hasActors'] if name in actor_names]
        
        if 'usedIn' in task_obj:
            for cap_name in task_obj['usedIn']:
                if cap_name not in capability_names:
                    print(f"Warning: Task references capability '{cap_name}' not found in capabilities list")
                    # Remove invalid reference
                    task_obj['usedIn'] = [name for name in task_obj['usedIn'] if name in capability_names]
        
        # Validate actor capabilities
        for actor in actors_obj:
            if 'has' in actor:
                invalid_capabilities = [cap for cap in actor['has'] if cap not in capability_names]
                if invalid_capabilities:
                    print(f"Warning: Actor '{actor['name']}' references invalid capabilities: {invalid_capabilities}")
                    # Fix references
                    actor['has'] = [cap for cap in actor['has'] if cap in capability_names]
        
        # Validate capability processing methods and metrics
        for capability in capabilities_obj:
            if 'hasProcessingMethod' in capability:
                invalid_methods = [m for m in capability['hasProcessingMethod'] if m not in processing_method_names]
                if invalid_methods:
                    print(f"Warning: Capability '{capability['name']}' references invalid processing methods: {invalid_methods}")
                    # Fix references
                    capability['hasProcessingMethod'] = [m for m in capability['hasProcessingMethod'] if m in processing_method_names]
            
            if 'produces' in capability:
                invalid_metrics = [m for m in capability['produces'] if m not in metric_names]
                if invalid_metrics:
                    print(f"Warning: Capability '{capability['name']}' references invalid metrics: {invalid_metrics}")
                    # Fix references
                    capability['produces'] = [m for m in capability['produces'] if m in metric_names]
        
        # Update the elements with validated JSON
        elements['scenario'] = json.dumps(scenario_obj)
        elements['task'] = json.dumps(task_obj)
        elements['actor'] = json.dumps(actors_obj)
        elements['capability'] = json.dumps(capabilities_obj)
        elements['processing_method'] = json.dumps(processing_methods_obj)
        elements['metric'] = json.dumps(metrics_obj)
        
        return elements
    except Exception as e:
        print(f"Error validating relationships: {e}")
        # Return original elements if validation fails
        return elements

def parse_extracted_elements(response):
    """Parse the response from Claude to extract the hybrid intelligence elements in JSON format."""
    if not response:
        print("No response received from Claude API")
        return None
    
    if not isinstance(response, str):
        print(f"Unexpected response type: {type(response)}. Expected string.")
        return None
    
    # Extract the content between the extracted_elements tags
    match = re.search(r'<extracted_elements>(.*?)</extracted_elements>', response, re.DOTALL)
    if not match:
        print("Could not find extracted_elements in the response")
        print("Response preview (first 200 chars):", response[:200])
        return None
    
    elements_text = match.group(1)
    
    try:
        # Extract each element using more robust regex patterns
        # Looking for content between triple quotes with non-greedy matching
        scenario_pattern = r'scenario\s*=\s*"""(.*?)"""'
        task_pattern = r'task\s*=\s*"""(.*?)"""'
        actor_pattern = r'actor\s*=\s*"""(.*?)"""'
        capability_pattern = r'capability\s*=\s*"""(.*?)"""'
        processing_method_pattern = r'processing_method\s*=\s*"""(.*?)"""'
        metric_pattern = r'metric\s*=\s*"""(.*?)"""'
        
        scenario_match = re.search(scenario_pattern, elements_text, re.DOTALL)
        task_match = re.search(task_pattern, elements_text, re.DOTALL)
        actor_match = re.search(actor_pattern, elements_text, re.DOTALL)
        capability_match = re.search(capability_pattern, elements_text, re.DOTALL)
        processing_method_match = re.search(processing_method_pattern, elements_text, re.DOTALL)
        metric_match = re.search(metric_pattern, elements_text, re.DOTALL)
        
        # Function to safely parse JSON or return a placeholder
        def safe_json_parse(match_obj, element_name):
            if not match_obj:
                print(f"{element_name} not found in response")
                return json.dumps({"error": f"{element_name} not found"})
            
            try:
                # Ensure the extracted text is valid JSON
                json_str = match_obj.group(1).strip()
                # Test if it's valid JSON by parsing it
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in {element_name}: {e}")
                print(f"Raw text: {match_obj.group(1)[:200]}")
                return json.dumps({"error": f"Invalid JSON in {element_name}"})
        
        # Create a dictionary with the extracted elements as JSON strings
        elements = {
            'scenario': safe_json_parse(scenario_match, "scenario"),
            'task': safe_json_parse(task_match, "task"),
            'actor': safe_json_parse(actor_match, "actor"),
            'capability': safe_json_parse(capability_match, "capability"),
            'processing_method': safe_json_parse(processing_method_match, "processing_method"),
            'metric': safe_json_parse(metric_match, "metric")
        }
        
        return elements
    except Exception as e:
        print(f"Error parsing extracted elements: {e}")
        print(f"Raw text from extracted elements (first 200 chars):\n{elements_text[:200]}")
        return None

def process_pdf_folder(folder_path, output_csv):
    """Process all PDFs in a folder and save the extracted elements to a CSV file."""
    # Get all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    # Create the CSV file and write the header
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'scenario', 'task', 'actor', 'capability', 'processing_method', 'metric']
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
            
            # Extract hybrid intelligence elements using Claude
            claude_response = extract_hybrid_intelligence_with_claude(pdf_text)
            
            # Parse the response to get individual elements
            elements = parse_extracted_elements(claude_response)
            if not elements:
                print(f"  Skipping {pdf_file} - could not parse elements from Claude's response")
                continue
                
            # Validate and fix relationships between elements
            elements = validate_element_relationships(elements)
            
            # Add the filename to the elements dictionary
            elements['filename'] = os.path.basename(pdf_file)
            
            # Write the elements to the CSV file
            writer.writerow(elements)
            
            print(f"  Successfully processed {pdf_file}")
    
    print(f"All PDF files processed. Results saved to {output_csv}")

def process_single_pdf(pdf_path, output_csv):
    """Process a single PDF file and save the extracted elements to a CSV file."""
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print(f"Could not extract text from {pdf_path}")
        return
    
    # Extract hybrid intelligence elements using Claude
    claude_response = extract_hybrid_intelligence_with_claude(pdf_text)
    
    # Parse the response to get individual elements
    elements = parse_extracted_elements(claude_response)
    if not elements:
        print(f"Could not parse elements from Claude's response for {pdf_path}")
        return
        
    # Validate and fix relationships between elements
    elements = validate_element_relationships(elements)
    
    # Create the CSV file and write the header
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'scenario', 'task', 'actor', 'capability', 'processing_method', 'metric']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                               quoting=csv.QUOTE_ALL,  # Quote all fields
                               quotechar='"',  # Use double quotes
                               escapechar='\\')  # Use backslash to escape quotes
        
        if not file_exists:
            writer.writeheader()
        
        # Add the filename to the elements dictionary
        elements['filename'] = os.path.basename(pdf_path)
        
        # Write the elements to the CSV file
        writer.writerow(elements)
    
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