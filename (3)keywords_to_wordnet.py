import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import ast
import json
import time
import os
import argparse
from tqdm import tqdm

# ====== HARD-CODED CONFIGURATION (edit these values) ======
# Input and output paths
INPUT_CSV_PATH = ".../data/research_paper.csv"  
OUTPUT_PATH = ".../data/research_paper_wordnet.csv"  # Output file path
OUTPUT_FORMAT = "csv"  # 'csv' or 'json'
# ========================================================

# Download necessary NLTK data (uncomment when first running)
# nltk.download('wordnet')
# nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_info(keyword):
    """
    Get WordNet synsets and related words for a given keyword.
    
    Args:
        keyword (str): The keyword to look up in WordNet
        
    Returns:
        dict: Dictionary containing WordNet information for the keyword
    """
    # Remove any leading/trailing whitespace
    keyword = keyword.strip()
    
    # For multi-word phrases, try both the whole phrase and individual words
    synsets = wn.synsets(keyword)
    individual_word_info = {}
    
    # Try lemmatizing the keyword if no synsets found
    if not synsets:
        # Tokenize the keyword
        tokens = nltk.word_tokenize(keyword)
        
        # Lemmatize each token and reconstruct
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_keyword = ' '.join(lemmatized_tokens)
        
        if lemmatized_keyword != keyword:
            synsets = wn.synsets(lemmatized_keyword)
    
    # If it's a multi-word phrase and no synsets were found, try individual words
    if not synsets and ' ' in keyword:
        words = keyword.split()
        for word in words:
            word_synsets = wn.synsets(word)
            if word_synsets:
                individual_word_info[word] = get_wordnet_info(word)
    
    if not synsets and not individual_word_info:
        return {
            'keyword': keyword,
            'found_in_wordnet': False,
            'synsets': [],
            'related_words': []
        }
    
    # Get related words from all synsets
    related_words = set()
    synset_info = []
    
    for synset in synsets:
        # Get lemma names (synonyms)
        lemma_names = synset.lemma_names()
        
        # Get definition and examples
        definition = synset.definition()
        examples = synset.examples()
        
        # Get hypernyms (more general terms)
        hypernyms = [h.lemma_names() for h in synset.hypernyms()]
        
        # Get hyponyms (more specific terms)
        hyponyms = [h.lemma_names() for h in synset.hyponyms()]
        
        # Add all words to the related words set
        for word in lemma_names:
            # Convert from WordNet format (remove underscores and lowercase)
            word = word.replace('_', ' ')
            if word.lower() != keyword.lower():
                related_words.add(word)
        
        for hyper_list in hypernyms:
            for word in hyper_list:
                word = word.replace('_', ' ')
                related_words.add(word)
                
        for hypo_list in hyponyms:
            for word in hypo_list:
                word = word.replace('_', ' ')
                related_words.add(word)
        
        # Add synset info
        synset_info.append({
            'name': synset.name(),
            'pos': synset.pos(),
            'definition': definition,
            'examples': examples,
            'lemmas': [lemma.replace('_', ' ') for lemma in lemma_names],
            'hypernyms': [[w.replace('_', ' ') for w in wlist] for wlist in hypernyms],
            'hyponyms': [[w.replace('_', ' ') for w in wlist] for wlist in hyponyms]
        })
    
    result = {
        'keyword': keyword,
        'found_in_wordnet': len(synsets) > 0,
        'synsets': synset_info,
        'related_words': list(related_words),
    }
    
    if individual_word_info:
        result['individual_words'] = individual_word_info
    
    return result

def parse_keywords(keywords_str):
    """
    Parse keywords from various formats.
    
    Args:
        keywords_str: The keywords in various possible formats
        
    Returns:
        list: List of keyword strings
    """
    if keywords_str is None or pd.isna(keywords_str):
        return []
    
    # If already a list, return it
    if isinstance(keywords_str, list):
        return keywords_str
    
    # Convert to string if not already
    if not isinstance(keywords_str, str):
        keywords_str = str(keywords_str)
    
    # For the specific format in your CSV where keywords are comma-separated
    # and the whole string may be quoted
    keywords_str = keywords_str.strip('"\'')
    
    # Try splitting by common delimiters
    if ';' in keywords_str:
        return [k.strip() for k in keywords_str.split(';')]
    elif ',' in keywords_str:
        return [k.strip() for k in keywords_str.split(',')]
    
    # If it looks like a single keyword with spaces
    return [keywords_str.strip()]

def process_keywords_csv(csv_path, output_path=None, output_format='csv'):
    """
    Process a CSV file containing a 'keywords' column with research paper keywords.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the output file
        output_format (str): Format of the output file ('csv' or 'json')
        
    Returns:
        pd.DataFrame: DataFrame with the original data and added WordNet information
    """
    print(f"Reading CSV file: {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    # Check if 'keywords' column exists
    if 'keywords' not in df.columns:
        # Try alternate casing
        if 'Keywords' in df.columns:
            print("Found 'Keywords' column (with capital K). Renaming to 'keywords'.")
            df.rename(columns={'Keywords': 'keywords'}, inplace=True)
        else:
            print(f"Warning: 'keywords' column not found. Available columns: {df.columns.tolist()}")
            raise ValueError("CSV file must contain a 'keywords' or 'Keywords' column")
    
    print(f"Found {len(df)} rows to process")
    print(f"First few rows of keywords column:")
    for i, kw in enumerate(df['keywords'].head(3)):
        print(f"  Row {i}: {kw}")
    
    # Create a fresh DataFrame for our results - copy all columns from original
    # Fill the wordnet_info column with empty JSON arrays to start
    result_df = df.copy()
    result_df['wordnet_info'] = '[]'  # Initialize with empty JSON arrays
    
    # Create a list to store intermediate results
    all_results = []
    
    # Track successful and failed rows
    success_count = 0
    failed_count = 0
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing keywords"):
        try:
            # Print raw keyword value for debugging
            print(f"\nRow {idx}, Raw keywords value: {row['keywords']}")
            
            # Parse the keywords 
            keywords = parse_keywords(row['keywords'])
            
            print(f"Row {idx}, Parsed {len(keywords)} keywords: {keywords}")
            
            # Default to empty array
            wordnet_info = []
            
            if not keywords:
                print(f"Warning: No keywords found in row {idx}")
            else:
                # Get WordNet info for each keyword
                for keyword in keywords:
                    info = get_wordnet_info(keyword)
                    wordnet_info.append(info)
                    print(f"Processed keyword: {keyword}")
            
            # Convert to JSON string
            wordnet_info_json = json.dumps(wordnet_info)
            
            # Force store the value back into the DataFrame by index
            # This should prevent any NaN issues
            result_df.loc[idx, 'wordnet_info'] = wordnet_info_json
            
            print(f"Added WordNet info for row {idx}, length: {len(wordnet_info_json)}")
            
            # Also store in our results list for incremental saving
            row_dict = dict(row)  # Convert to dict to avoid Series issues
            row_dict['wordnet_info'] = wordnet_info
            all_results.append(row_dict)
            
            success_count += 1
            
            # Save intermediate results after every 5 rows or when we hit the last row
            if output_path and (idx % 5 == 0 or idx == len(df) - 1):
                output_base, output_ext = os.path.splitext(output_path)
                intermediate_path = f"{output_base}_partial{output_ext}"
                
                # Create a clean DataFrame from our collected results
                temp_df = pd.DataFrame(all_results)
                
                if output_format.lower() == 'csv':
                    # Save directly to CSV
                    temp_df.to_csv(intermediate_path, index=False)
                elif output_format.lower() == 'json':
                    # For JSON, we need to properly convert the wordnet_info column
                    json_data = []
                    for _, r in temp_df.iterrows():
                        r_dict = r.to_dict()
                        json_data.append(r_dict)
                    
                    with open(intermediate_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                print(f"Saved intermediate results after row {idx} to: {intermediate_path}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            print(f"Raw keywords value: {row['keywords']}")
            import traceback
            traceback.print_exc()
            
            # Store empty results for this row - make sure this runs successfully
            try:
                result_df.loc[idx, 'wordnet_info'] = json.dumps([])
                # Also add to all_results with empty wordnet_info
                row_dict = dict(row)
                row_dict['wordnet_info'] = []
                all_results.append(row_dict)
            except Exception as e2:
                print(f"Error even storing empty results for row {idx}: {str(e2)}")
            
            failed_count += 1
    
    # Check for any NaN values in the wordnet_info column
    nan_count = result_df['wordnet_info'].isna().sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values in wordnet_info column")
        print("Fixing NaN values...")
        result_df['wordnet_info'] = result_df['wordnet_info'].fillna('[]')
    
    # Double-verify all rows have a value for wordnet_info
    for idx in result_df.index:
        if pd.isna(result_df.loc[idx, 'wordnet_info']):
            print(f"Still found NaN at row {idx}, fixing...")
            result_df.loc[idx, 'wordnet_info'] = '[]'
    
    # Final verification
    print("\nResult DataFrame verification:")
    print(f"Total rows: {len(result_df)}")
    print(f"Successful rows: {success_count}")
    print(f"Failed rows: {failed_count}")
    nan_count = result_df['wordnet_info'].isna().sum()
    print(f"Final NaN count: {nan_count}")
    
    # Check how many rows have non-empty values
    non_empty_count = result_df['wordnet_info'].apply(
        lambda x: not pd.isna(x) and x != 'null' and x != '[]'
    ).sum()
    print(f"Rows with non-empty wordnet_info: {non_empty_count}")
    
    # Save final output file if specified
    if output_path:
        if output_format.lower() == 'csv':
            # Ensure no NaN values remain
            for col in result_df.columns:
                if result_df[col].dtype == 'object':
                    result_df[col] = result_df[col].fillna('')
            
            # Save the final output
            try:
                result_df.to_csv(output_path, index=False)
                print(f"Saved final CSV output to: {output_path}")
            except Exception as e:
                print(f"Error saving final CSV: {str(e)}")
                # Try again with a different filename
                backup_path = f"{os.path.splitext(output_path)[0]}_backup.csv"
                result_df.to_csv(backup_path, index=False)
                print(f"Saved backup CSV to: {backup_path}")
                
        elif output_format.lower() == 'json':
            try:
                json_data = []
                for _, row in result_df.iterrows():
                    row_dict = row.to_dict()
                    # Convert the JSON strings back to objects
                    if 'wordnet_info' in row_dict:
                        if isinstance(row_dict['wordnet_info'], str):
                            try:
                                row_dict['wordnet_info'] = json.loads(row_dict['wordnet_info'])
                            except json.JSONDecodeError:
                                row_dict['wordnet_info'] = []
                        elif pd.isna(row_dict['wordnet_info']):
                            row_dict['wordnet_info'] = []
                    else:
                        row_dict['wordnet_info'] = []
                        
                    json_data.append(row_dict)
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print(f"Saved final JSON output to: {output_path}")
            except Exception as e:
                print(f"Error saving final JSON: {str(e)}")
                # Try again with a different filename
                backup_path = f"{os.path.splitext(output_path)[0]}_backup.json"
                with open(backup_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Saved backup JSON to: {backup_path}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Process research paper keywords and find related WordNet terms.')
    parser.add_argument('--input', '-i', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to the output file')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], help='Output format')
    
    args = parser.parse_args()
    
    # Use command line args if provided, otherwise use hard-coded values
    input_path = args.input if args.input else INPUT_CSV_PATH
    output_format = args.format if args.format else OUTPUT_FORMAT
    
    # If output path is not specified in args or hard-coded config is empty
    if args.output:
        output_path = args.output
    elif OUTPUT_PATH:
        output_path = OUTPUT_PATH
    else:
        # Derive output path from input path
        base, ext = os.path.splitext(input_path)
        if output_format == 'csv':
            output_path = f"{base}_enriched.csv"
        else:
            output_path = f"{base}_enriched.json"
    
    print(f"Using input file: {input_path}")
    print(f"Output will be saved to: {output_path} (format: {output_format})")
    
    process_keywords_csv(
        input_path,
        output_path,
        output_format=output_format
    )

if __name__ == "__main__":
    main()