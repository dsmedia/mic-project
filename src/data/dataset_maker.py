import os
import json
import duckdb
#from dotenv import load_dotenv
import logging
from tqdm import tqdm
from collections import defaultdict
import sys
from datetime import datetime
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#load_dotenv()

# --- Constants ---
DB_PATH = 'data/processed/mic_analysis.duckdb'
# Input file containing PRE-GENERATED LLM responses (one JSON object per line)
LLM_RESULTS_FILE = "data/processed/training_model_results.jsonl"
# Output file for the ShareGPT dataset
SHAREGPT_OUTPUT_FILE = "data/processed/mic_training_dataset_for_phi_4.json" # Renamed output file
# Fields to exclude from the individual JSON objects before assembling the 'gpt' response value
FIELDS_TO_EXCLUDE = ['validation_status', 'validation_issues', 'record_type']
# Maximum characters for text truncation in create_prompt
MAX_PROMPT_TEXT_CHARS = 18000 # Adjust as needed based on model context limits

# --- System Prompt Template (Stable Instructions) ---
# Contains the core definitions, rules, constraints, and output format.
SYSTEM_PROMPT_TEMPLATE = """You are an AI expert specializing in identifying Militarized Interstate Confrontation (MIC) events involving fatalities from text, based on strict definitions. Respond *only* with a valid JSON **array** containing objects as specified below.

**MIC Definition (Strict Application Required):**
A relevant MIC event occurs when the **military forces** of one **internationally recognized state** directly cause the death of one or more **military personnel** belonging to **another internationally recognized state**.

**Military Forces Classification Guidelines:**
When determining whether forces qualify as "military forces" for MIC coding purposes, apply these guidelines:

**Primary Military Forces (Always Code as Military)**
- **Regular Armed Services:** Army, Navy, Air Force, Marines, Space Force, etc. operating under official military command
- **National Guard/Reserve Forces:** When federalized/nationalized or deployed in combat operations
- **Border Guards/Coast Guards:** When under military command structure or conducting military operations (not routine patrol)
- **Military Intelligence Units:** When operating in combat roles

**Secondary Military Forces (Code as Military When Criteria Met)**
- **State-Controlled Paramilitary Groups:** Require evidence of:
  1. Direct state control through military chain of command
  2. Military-style operations beyond internal security
  3. State acknowledgment of their actions
- **Military Contractors/Mercenaries:** Only when:
  1. Formally integrated into military operations
  2. Under direct military command control
  3. Officially acknowledged as operating on behalf of state

**Exclusions (Do Not Code as Military)**
- Civilian law enforcement agencies performing domestic functions
- Intelligence agents in non-combat operations
- Rebel/insurgent groups lacking formal state recognition/control
- Private security companies operating independently
- Civilian border or customs officials
- Peacekeepers operating under UN/international command (not national)

**Decision Rules for Ambiguous Cases**
1. Official designation in source material (article specifically identifies as military)
2. Command structure (reports to military chain of command)
3. Function (performing traditional military operations)
4. Equipment (using military-grade weapons/vehicles)
5. State acknowledgment (officially recognized as state military action)

When coding, an entity must satisfy at least three of these criteria to qualify as military forces for MIC purposes.

*   **Eligible Countries:** The states involved (`countries_suffering_losses`, `countries_causing_losses`) **must** be mappable to the standard, full names provided in the list below. Country names in these arrays MUST match the spelling and format in this list *exactly*. Do not abbreviate or use alternative names.
    *   `eligible_countries`: ["United States of America", "Canada", "Bahamas", "Cuba", "Haiti", "Dominican Republic", "Jamaica", "Trinidad and Tobago", "Barbados", "Dominica", "Grenada", "St. Lucia", "St. Vincent and the Grenadines", "Antigua & Barbuda", "St. Kitts and Nevis", "Mexico", "Belize", "Guatemala", "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama", "Colombia", "Venezuela", "Guyana", "Suriname", "Ecuador", "Peru", "Brazil", "Bolivia", "Paraguay", "Chile", "Argentina", "Uruguay", "United Kingdom", "Ireland", "Netherlands", "Belgium", "Luxembourg", "France", "Monaco", "Liechtenstein", "Switzerland", "Spain", "Andorra", "Portugal", "Germany", "Poland", "Austria", "Hungary", "Czech Republic", "Slovakia", "Italy", "San Marino", "Malta", "Albania", "Montenegro", "Macedonia", "Croatia", "Yugoslavia", "Bosnia and Herzegovina", "Kosovo", "Slovenia", "Greece", "Cyprus", "Bulgaria", "Moldova", "Romania", "Russia", "Estonia", "Latvia", "Lithuania", "Ukraine", "Belarus", "Armenia", "Georgia", "Azerbaijan", "Finland", "Sweden", "Norway", "Denmark", "Iceland", "Cape Verde", "Sao Tome and Principe", "Guinea-Bissau", "Equatorial Guinea", "Gambia", "Mali", "Senegal", "Benin", "Mauritania", "Niger", "Ivory Coast", "Guinea", "Burkina Faso", "Liberia", "Sierra Leone", "Ghana", "Togo", "Cameroon", "Nigeria", "Gabon", "Central African Republic", "Chad", "Congo", "Democratic Republic of the Congo", "Uganda", "Kenya", "Tanzania", "Burundi", "Rwanda", "Somalia", "Djibouti", "Ethiopia", "Eritrea", "Angola", "Mozambique", "Zambia", "Zimbabwe", "Malawi", "South Africa", "Namibia", "Lesotho", "Botswana", "Swaziland", "Madagascar", "Comoros", "Mauritius", "Seychelles", "Morocco", "Algeria", "Tunisia", "Libya", "Sudan", "South Sudan", "Iran", "Turkey", "Iraq", "Egypt", "Syria", "Lebanon", "Jordan", "Israel", "Saudi Arabia", "Yemen", "Kuwait", "Bahrain", "Qatar", "United Arab Emirates", "Oman", "Afghanistan", "Turkmenistan", "Tajikistan", "Kyrgyzstan", "Uzbekistan", "Kazakhstan", "China", "Mongolia", "Taiwan", "North Korea", "South Korea", "Japan", "India", "Bhutan", "Pakistan", "Bangladesh", "Myanmar", "Sri Lanka", "Maldives", "Nepal", "Thailand", "Cambodia", "Laos", "Vietnam", "Malaysia", "Singapore", "Brunei", "Philippines", "Indonesia", "East Timor", "Australia", "Papua New Guinea", "New Zealand", "Vanuatu", "Solomon Islands", "Kiribati", "Tuvalu", "Fiji", "Tonga", "Nauru", "Marshall Islands", "Palau", "Federated States of Micronesia", "Samoa"]
*   **Focus ONLY on deaths of *military personnel*.** Ignore civilian deaths.
*   **Crucially, exclude incidents where fatalities are identified *only* as police, border guards, non-military security forces, or other non-military state personnel, even if they are involved in security operations.** If the text is ambiguous but strongly implies military personnel in a military context (e.g., 'soldiers', 'sailors', 'airmen' during a clash), they can be included, but note the ambiguity in the explanation.
*   **Must be *interstate*:** Directly between the official armed forces of two or more recognized countries from the `eligible_countries` list.
*   **Entities NOT considered states for this task:** Do not classify incidents as interstate if they primarily involve non-state actors, international organizations as unified actors (unless specific state forces clash), sub-national entities fighting their own government, or unidentified forces (unless explicitly linked to an eligible state).
*   **Direct Causation:** One eligible state's forces must have killed the other eligible state's forces.
*   **Exclude:** Internal conflicts (unless specific interstate clash occurs within), non-state actor conflicts, internal accidents, incidents with no military fatalities, peacekeeping deaths (unless State A peacekeeper killed by State B military), events involving non-eligible states.
*   **Distinct Events:** Pay attention to the narrative. If an article describes multiple separate clashes or incidents over time (even if close together), generate a distinct JSON object for *each* incident that meets the MIC definition.
*   **Scan Entire Text:** Thoroughly examine the *entire* article text. Do **not** ignore an event that meets the MIC definition simply because it is mentioned as historical background, context, a comparison, or is not the main focus of the article. If it's described and fits the criteria, extract it.

**Output Format:**
*   **If Relevant Event(s) Found:** Return a JSON array containing one JSON object per distinct event with fields: `article_id` (Int), `is_relevant` (Bool, true), `start_year` (Int/-9), `start_month` (Int/-9), `start_day` (Int/-9), `end_year` (Int/-9), `end_month` (Int/-9), `end_day` (Int/-9), `fatalities_min` (Int), `fatalities_max` (Int), `countries_suffering_losses` (Array[Str]), `countries_causing_losses` (Array[Str]), `explanation` (Str).
*   **If NO Relevant Events Found:** Return a JSON array containing exactly one JSON object with fields: `article_id` (Int), `is_relevant` (Bool, false), `start_year`: null, `start_month`: null, `start_day`: null, `end_year`: null, `end_month`: null, `end_day`: null, `fatalities_min`: null, `fatalities_max`: null, `countries_suffering_losses`: [], `countries_causing_losses`: [], `explanation` (Str).

Respond ONLY with a single, valid JSON array. Each object must include all the fields specified above in the correct order.
"""

# --- Helper function to safely get nested keys ---
def safe_get(data, keys, default=None):
    """Safely get a nested key from a dictionary."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

# --- Create User Prompt Content function ---
def create_user_prompt_content(article_data):
    """
    Creates the user prompt content, containing the specific article data
    and a reference to the system prompt instructions.
    """
    article_id = article_data.get('id', 'N/A')
    raw_pub_date = article_data.get('publication_date', 'N/A')
    if raw_pub_date and raw_pub_date != 'N/A':
        try:
            # Parse the date string (adjust the format as needed based on your actual data)
            # Common formats: YYYY-MM-DD, MM/DD/YYYY, etc.
            date_obj = datetime.strptime(raw_pub_date, '%b %d, %Y')
            # date_obj = datetime.strptime(raw_pub_date, '%Y-%m-%d')
            # Format to "Tuesday, January 15, 2017"
            pub_date = date_obj.strftime('%A, %B %d, %Y')
        except ValueError:
            # If parsing fails, keep the original format
            logger.warning(f"Failed to parse publication date '{raw_pub_date}' for article ID {article_id}")
            pub_date = raw_pub_date
    else:
        pub_date = 'N/A'
    # pub_date = article_data.get('publication_date', 'N/A')
    full_text = article_data.get('full_text', '') or ''
    location = article_data.get('location', 'N/A')
    subject = article_data.get('subject', 'N/A')
    people = article_data.get('people', 'N/A')

    # Apply truncation to the article text
    if len(full_text) > MAX_PROMPT_TEXT_CHARS:
        original_length = len(full_text)
        full_text = full_text[:MAX_PROMPT_TEXT_CHARS] + " [TEXT TRUNCATED]"
        logger.debug(f"Truncating full_text for article ID {article_id} in prompt generation (Original length: {original_length})")

    location_str = f"Location Context: {location}" if location and location != 'N/A' else "Location Mentioned: Not Available"
    subject_str = f"Subject Context: {subject}" if subject and subject != 'N/A' else "Subject Keywords: Not Available"
    people_str = f"People Context: {people}" if people and people != 'N/A' else "People Mentioned: Not Available"

    # User prompt content: brief instruction + article details + article text
    user_content = f"""Analyze the following news article using the MIC definitions and formatting rules provided in the system prompt.

**Input Article Context:**
*   Article ID: {article_id}
*   Publication Date: {pub_date}
*   {location_str}
*   {subject_str}
*   {people_str}

**Full Article Text:**
--- START TEXT ---
{full_text}
--- END TEXT ---
"""
    return user_content

# --- Main Function ---
def generate_sharegpt_dataset():
    """
    Generates the ShareGPT dataset from pre-existing LLM responses and DB articles,
    using the recommended system/user prompt structure.
    """
    logger.info(f"Starting ShareGPT dataset generation with system prompt structure.")
    logger.info(f"Reading PRE-GENERATED LLM results from: {LLM_RESULTS_FILE}")

    # Step 1 & 2: Read JSONL and group by article_id
    llm_responses_by_id = defaultdict(list)
    processed_ids = set()
    line_count = 0
    read_errors = 0

    try:
        with open(LLM_RESULTS_FILE, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    article_id_val = safe_get(data, ['article_id'])
                    if article_id_val is not None:
                        try:
                            article_id_int = int(article_id_val)
                            # Store the raw response object
                            llm_responses_by_id[article_id_int].append(data)
                            processed_ids.add(article_id_int)
                        except (ValueError, TypeError):
                            logger.warning(f"Line {line_count}: Invalid non-integer article_id '{article_id_val}'. Skipping.")
                            read_errors += 1
                    else:
                        logger.warning(f"Line {line_count}: Missing 'article_id'. Skipping line.")
                        read_errors += 1
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_count}: Failed to decode JSON. Skipping line.")
                    read_errors += 1
                except Exception as e:
                    logger.warning(f"Line {line_count}: Unexpected error: {e}. Skipping.")
                    read_errors += 1
    except FileNotFoundError:
        logger.critical(f"FATAL ERROR: LLM results file not found: {LLM_RESULTS_FILE}")
        sys.exit(f"Error: Input file {LLM_RESULTS_FILE} not found.")
    except IOError as e:
        logger.critical(f"FATAL ERROR: Cannot read LLM results file: {e}")
        sys.exit("Error: Cannot read input file.")

    if not processed_ids:
        logger.critical("FATAL ERROR: No valid article IDs found in the LLM results file.")
        sys.exit("Error: No data to process.")

    logger.info(f"Read {line_count} lines. Found {len(processed_ids)} unique article IDs with {sum(len(v) for v in llm_responses_by_id.values())} total responses.")

    # Step 3 & 4: Fetch article data from DuckDB
    article_data_map = {}
    con = None
    try:
        logger.info(f"Connecting to DuckDB database: {DB_PATH}")
        con = duckdb.connect(database=DB_PATH, read_only=True)

        ids_tuple = tuple(processed_ids)
        # Handle single ID case for SQL IN clause compatibility if needed by DB version
        if len(ids_tuple) == 1:
             sql_query = f"SELECT id, publication_date, full_text, location, subject, people FROM raw.articles WHERE id = {ids_tuple[0]};"
             params = None # No parameters needed for direct value injection
        else:
            sql_query = f"SELECT id, publication_date, full_text, location, subject, people FROM raw.articles WHERE id IN ?;"
            params = (ids_tuple,) # Use parameter binding for multiple IDs

        logger.info(f"Executing query to fetch data for {len(ids_tuple)} articles...")
        # Use parameter binding if params is not None
        cursor = con.execute(sql_query, parameters=params) if params else con.execute(sql_query)
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} articles from the database.")

        for row in rows:
            article_dict = dict(zip(column_names, row))
            article_id = article_dict.get('id')
            if article_id is not None:
                article_data_map[article_id] = article_dict
    except duckdb.Error as db_err:
        logger.error(f"DuckDB Error: {db_err}", exc_info=True)
        sys.exit("Error: Database operation failed.")
    finally:
        if con:
            try:
                con.close()
                logger.info("Database connection closed.")
            except Exception as close_e:
                logger.error(f"Error closing DuckDB connection: {close_e}")

    # Step 5: Generate ShareGPT Data
    sharegpt_data = []
    skipped_count = 0
    error_count = 0

    logger.info("Generating ShareGPT conversation entries...")
    sorted_ids = sorted(list(processed_ids))

    for article_id in tqdm(sorted_ids, desc="Generating ShareGPT Entries"):
        # Check if we have both article data and LLM responses
        if article_id not in article_data_map:
            logger.warning(f"Article ID {article_id} found in LLM results but not in DB. Skipping.")
            skipped_count += 1
            continue

        if article_id not in llm_responses_by_id or not llm_responses_by_id[article_id]:
            logger.warning(f"Article ID {article_id} found in DB but no valid LLM responses. Skipping.")
            skipped_count += 1
            continue

        article_data = article_data_map[article_id]
        # Use the first LLM response found for this article_id as the target 'assistant' response
        # You might want more sophisticated logic if multiple responses exist per article
        llm_result = llm_responses_by_id[article_id][0]

        try:
            # Generate user prompt content
            user_content = create_user_prompt_content(article_data)

            # Clean the single LLM response
            cleaned_response_obj = llm_result.copy()
            for field in FIELDS_TO_EXCLUDE:
                cleaned_response_obj.pop(field, None)

            # Format the cleaned response object as a JSON array string (as required by the prompt)
            # The prompt asks for an array, even if it's a single 'false' object
            gpt_response_content = json.dumps([cleaned_response_obj], ensure_ascii=False)

            # Create ShareGPT format entry with system, user, and assistant roles
            conversation_entry = {
                "conversations": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_TEMPLATE # Use the constant system prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": gpt_response_content
                    }
                ]
            }

            sharegpt_data.append(conversation_entry)

        except Exception as e:
            logger.error(f"Error processing article ID {article_id}: {e}", exc_info=True) # Added exc_info for traceback
            error_count += 1

    logger.info(f"Successfully generated {len(sharegpt_data)} ShareGPT entries.")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} articles due to missing data or responses.")
    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing.")

    # Step 6: Write Output and Print Sample
    if sharegpt_data:
        logger.info(f"Writing ShareGPT dataset to: {SHAREGPT_OUTPUT_FILE}")
        try:
            with open(SHAREGPT_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                # Write as JSON Lines (one conversation object per line)
                # This is often preferred for large datasets and streaming processing
                # If you need a single large JSON array, use json.dump(sharegpt_data, f_out, ...)
                for entry in sharegpt_data:
                    json.dump(entry, f_out, ensure_ascii=False)
                    f_out.write('\n')
            logger.info("Successfully wrote ShareGPT dataset (JSON Lines format).")
            
            # Save and print a sample
            sample_file = SHAREGPT_OUTPUT_FILE.replace('.json', '_sample.jsonl')
            sample_size = min(3, len(sharegpt_data))  # Take up to 3 samples
            logger.info(f"Saving {sample_size} sample entries to {sample_file}")
            
            try:
                with open(sample_file, 'w', encoding='utf-8') as f_sample:
                    for i in range(sample_size):
                        json.dump(sharegpt_data[i], f_sample, ensure_ascii=False)
                        f_sample.write('\n')
                logger.info(f"Sample saved to {sample_file}")
                
                # Print the first sample to console
                if sample_size > 0:
                    logger.info("Sample data (first entry):")
                    sample_entry = sharegpt_data[0]
                    # Extract key information for display
                    system_prompt = sample_entry['conversations'][0]['content']
                    user_prompt = sample_entry['conversations'][1]['content']
                    assistant_response = sample_entry['conversations'][2]['content']
                    
                    # Truncate for display purposes
                    max_display = 500
                    print(f"System prompt (first {min(max_display, len(system_prompt))} chars): {system_prompt[:max_display]}...")
                    print(f"User prompt (first {min(max_display, len(user_prompt))} chars): {user_prompt[:max_display]}...")
                    print(f"Assistant response: {assistant_response}")
            except IOError as e:
                logger.error(f"Failed to write sample file: {e}")
        except IOError as e:
            logger.error(f"Failed to write output file: {e}")
    else:
        logger.warning("No ShareGPT data was generated. Output file will not be created.")

    logger.info("ShareGPT dataset generation complete.")

if __name__ == "__main__":
    generate_sharegpt_dataset()