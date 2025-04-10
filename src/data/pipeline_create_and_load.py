#!/usr/bin/env python3
"""
Combined script to:
1. Create the DuckDB database structure for MIC analysis.
2. Load articles in parallel from ProQuest text files into raw.articles.
3. Load articles in parallel from NYT-style text files into raw.parsed_articles.

Prompts the user if the target database file already exists. Logs to console and file.
"""
import os
import sys
import re
import logging
import logging.handlers
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import datetime # For NYT date conversion
import chardet # For encoding detection: pip install chardet
import duckdb
import yaml # For category filtering config: pip install pyyaml
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm # Progress bar: pip install tqdm
import uuid

# --- Calculate Project Root ---
try:
    # Assumes script is in PROJECT_ROOT/src/data
    project_root = Path(__file__).resolve().parents[2]
    # Add debug logging to verify the path
    print(f"DEBUG: Project root calculated as: {project_root}")
    print(f"DEBUG: Expected data path: {project_root / 'data' / 'raw'}")
    print(f"DEBUG: This path exists: {(project_root / 'data' / 'raw').exists()}")
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    project_root = Path(os.getcwd()).resolve()
    logging.basicConfig(level=logging.WARNING)
    logging.warning(f"__file__ not defined. Assuming project root is current working directory: {project_root}")

# Add project root's 'config' directory to Python path for config import
try:
    from .config import get_config, add_config_args
except ImportError:
    # Try absolute import as fallback
    try:
        sys.path.insert(0, str(project_root / "src" / "data"))
        from config import get_config, add_config_args
    except ImportError as e:
        # Use basic print before logging is fully configured
        print(f"ERROR: Could not import configuration module from {project_root}/src/data/config.py", file=sys.stderr)
        print(f"Ensure the script is placed correctly (e.g., in src/data) and src/data/config.py exists.", file=sys.stderr)
        print(f"Import error details: {e}", file=sys.stderr)
        sys.exit(1)

# --- Set up Logging (Console and File) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_level = logging.INFO  # Set the desired level (e.g., INFO, DEBUG)

# --- Constants ---
# ProQuest constants moved to config, but keep fallback/general ones if needed
PROQUEST_SEPARATOR = "\nDocument "  # Or whatever the appropriate separator pattern is
ENCODING_FALLBACKS = ['utf-8', 'windows-1252', 'iso-8859-1']

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)  # Set the minimum level for the root logger

# Clear existing handlers (important if script is re-run in same session or if basicConfig was used)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Console Handler (StreamHandler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File Handler
log_file_path = project_root / "pipeline_run.log"  # Define log file path
try:
    # Use RotatingFileHandler for larger logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    # Use print here as logger might not be fully ready if file handler fails
    print(f"INFO: Logging to console and file: {log_file_path}")
except Exception as e:
    print(f"ERROR: Failed to set up file logging to {log_file_path}: {e}", file=sys.stderr)
    # Continue with console logging only

# Get the logger for this specific script
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# =============================================================================
# == DATABASE CREATION FUNCTIONS
# =============================================================================

def create_category_filtering_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Creates and populates category filtering tables in staging schema."""
    logger.info("Creating/Populating category filtering tables in staging schema")
    config_path = project_root / "config" / "category_filtering.yaml"
    filter_config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                filter_config = yaml.safe_load(file) or {}
            logger.info(f"Loaded category filtering config from {config_path}")
        else:
            logger.warning(f"Category filtering config file not found: {config_path}. Tables will be created empty.")
    except Exception as e:
        logger.error(f"Error loading category filtering config: {e}")

    excluded_categories_list = filter_config.get("excluded_categories", []) or []
    relevant_subjects_list = filter_config.get("relevant_subjects", []) or []
    excludable_subjects_list = filter_config.get("excludable_subjects", []) or []
    domestic_locations_list = filter_config.get("domestic_locations", []) or []
    filtered_articles_sql = filter_config.get("filtered_articles", "")

    conn.execute("CREATE TABLE IF NOT EXISTS staging.excluded_categories (category_name VARCHAR UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS staging.relevant_subjects (subject_name VARCHAR UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS staging.excludable_subjects (subject_name VARCHAR UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS staging.domestic_locations (location_name VARCHAR UNIQUE NOT NULL)")

    # Clear existing data before repopulating
    conn.execute("DELETE FROM staging.excluded_categories")
    conn.execute("DELETE FROM staging.relevant_subjects")
    conn.execute("DELETE FROM staging.excludable_subjects")
    conn.execute("DELETE FROM staging.domestic_locations")

    # Use executemany for potentially better performance
    if excluded_categories_list:
        categories_data = [(c,) for c in set(excluded_categories_list) if c is not None]
        if categories_data:
            conn.executemany("INSERT INTO staging.excluded_categories (category_name) VALUES (?)", categories_data)
            logger.info(f"Inserted {len(categories_data)} excluded categories.")

    if relevant_subjects_list:
        subjects_data = [(s,) for s in set(relevant_subjects_list) if s is not None]
        if subjects_data:
            conn.executemany("INSERT INTO staging.relevant_subjects (subject_name) VALUES (?)", subjects_data)
            logger.info(f"Inserted {len(subjects_data)} relevant subjects.")

    if excludable_subjects_list:
        subjects_data = [(s,) for s in set(excludable_subjects_list) if s is not None]
        if subjects_data:
            conn.executemany("INSERT INTO staging.excludable_subjects (subject_name) VALUES (?)", subjects_data)
            logger.info(f"Inserted {len(subjects_data)} excludable subjects.")

    if domestic_locations_list:
        locations_data = [(l,) for l in set(domestic_locations_list) if l is not None]
        if locations_data:
            conn.executemany("INSERT INTO staging.domestic_locations (location_name) VALUES (?)", locations_data)
            logger.info(f"Inserted {len(locations_data)} domestic locations.")

    # Create or replace the filtered_articles view
    if filtered_articles_sql:
        try:
            # Drop the view if it exists
            conn.execute("DROP VIEW IF EXISTS staging.filtered_articles")
            # Create view using the SQL from the config
            create_view_sql = f"CREATE VIEW staging.filtered_articles AS {filtered_articles_sql}"
            conn.execute(create_view_sql)
            # Verify view was created by checking system catalog
            view_exists = conn.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.views
                    WHERE table_schema = 'staging' AND table_name = 'filtered_articles'
                )
            """).fetchone()[0]
            if view_exists:
                logger.info("Created and verified view 'staging.filtered_articles' from config SQL.")
            else:
                logger.error("View 'staging.filtered_articles' was not found in system catalog after creation attempt.")
                logger.debug(f"Failed SQL: {filtered_articles_sql[:200]}...")
        except Exception as view_e:
            logger.error(f"Error creating filtered_articles view: {view_e}")
            logger.debug(f"Failed SQL: {filtered_articles_sql[:200]}...")
    else:
        # Create a default view if no SQL provided in config
        try:
            conn.execute("DROP VIEW IF EXISTS staging.filtered_articles")
            default_view_sql = """
                CREATE VIEW staging.filtered_articles AS
                SELECT a.*
                FROM raw.articles a
                WHERE NOT EXISTS (
                    SELECT 1 FROM staging.excluded_categories ec
                    WHERE ec.category_name = bracket_category(a.title)
                )
                AND (
                    NOT EXISTS (SELECT 1 FROM staging.relevant_subjects) 
                    OR EXISTS (
                        SELECT 1 FROM staging.relevant_subjects rs
                        WHERE a.subject ILIKE '%' || rs.subject_name || '%'
                    )
                )
            """
            conn.execute(default_view_sql)
            logger.info("Created default 'staging.filtered_articles' view (no SQL found in config).")
        except Exception as def_view_e:
            logger.error(f"Error creating default filtered_articles view: {def_view_e}")

    # Create macro (consider moving complex SQL to separate files if it grows)
    conn.execute("""
        CREATE OR REPLACE MACRO bracket_category(title_col) AS
          CASE
              WHEN title_col LIKE '%Paid Notice:%' THEN lower(regexp_replace(trim(split_part(title_col, ':', 2)), '[^a-zA-Z0-9]', '', 'g'))
              WHEN title_col LIKE '%[%]%' THEN
                  CASE
                      WHEN array_length(regexp_extract_all(title_col, '\\[(.*?)\\]')) > 1 THEN
                          CASE
                              WHEN regexp_matches(regexp_extract_all(title_col, '\\[(.*?)\\]')[array_length(regexp_extract_all(title_col, '\\[(.*?)\\]'))], '^\\s*\\d+\\s*$')
                              THEN lower(regexp_replace(trim(regexp_extract_all(title_col, '\\[(.*?)\\]')[array_length(regexp_extract_all(title_col, '\\[(.*?)\\]')) - 1]), '[^a-zA-Z0-9]', '', 'g'))
                              ELSE lower(regexp_replace(trim(regexp_extract_all(title_col, '\\[(.*?)\\]')[array_length(regexp_extract_all(title_col, '\\[(.*?)\\]'))]), '[^a-zA-Z0-9]', '', 'g'))
                          END
                      ELSE
                          CASE
                              WHEN regexp_matches(regexp_extract_all(title_col, '\\[(.*?)\\]')[1], '^\\s*\\d+\\s*$') THEN NULL
                              ELSE lower(regexp_replace(trim(regexp_extract_all(title_col, '\\[(.*?)\\]')[1]), '[^a-zA-Z0-9]', '', 'g'))
                          END
                  END
              ELSE NULL
          END
    """)
    logger.info("Created/Replaced macro 'bracket_category'.")
    logger.info("Finished setting up category filtering tables and objects.")

def create_database_structure(db_path: str, config: Dict[str, Any]) -> bool:
    """Creates and initializes the DuckDB database structure."""
    logger.info(f"Ensuring DuckDB database structure exists at {db_path}")
    conn = None
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            logger.debug(f"Ensured directory exists: {db_dir}")

        conn = duckdb.connect(db_path)
        logger.info("Database connection established.")

        # Create Schemas
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        conn.execute("CREATE SCHEMA IF NOT EXISTS staging;")
        conn.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
        logger.info("Schemas created (if not exist): raw, staging, analytics.")

        # --- Create Tables ---
        # Raw table for ProQuest data
        proquest_table = config['loading']['proquest']['target_table']
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {proquest_table} (
                id UBIGINT PRIMARY KEY,
                title TEXT,
                clean_title TEXT,
                section TEXT,
                section_code TEXT,
                author TEXT,
                abstract TEXT,
                full_text TEXT,
                raw_text_length INTEGER,
                publication_title TEXT,
                publication_date TEXT,       -- Store raw string from ProQuest
                publication_date_raw TEXT, -- Keep original if needed
                publication_year TEXT,       -- Extracted year
                page TEXT,
                url TEXT,
                location TEXT,
                subject TEXT,
                people TEXT,
                keywords TEXT,
                document_id VARCHAR,
                source TEXT,                 -- Often same as publication_title
                file_path TEXT,              -- Source file path
                place_of_publication TEXT,
                country_of_publication TEXT,
                document_type TEXT,
                publisher TEXT,
                last_updated TEXT,
                copyright TEXT,
                issn TEXT,
                source_type TEXT,
                language TEXT,
                database TEXT,
                document_content TEXT      -- Potentially redundant with full_text? Check usage.
            )
        """)
        logger.info(f"Table {proquest_table} created (if not exist).")

        # Raw table for NYT-style parsed data
        nyt_table = config['loading']['nyt']['target_table']
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {nyt_table} (
                id UBIGINT PRIMARY KEY,      -- Auto-incrementing ID like raw.articles
                source_filepath TEXT,        -- Path to the source text file
                format_type VARCHAR,         -- Identifier for the format (e.g., 'NYT_2011_2014')
                format_note TEXT,            -- Any notes about parsing quirks
                headline TEXT,
                body TEXT,                   -- Renamed from full_text for clarity vs ProQuest
                publication_date DATE,       -- Use DATE type for proper querying
                nyt_internal_id VARCHAR,     -- The 'Key' field from the source file (UUID)
                nyt_country_codes VARCHAR,   -- The 'Countries' field
                nyt_source_info TEXT,        -- Placeholder for other potential metadata
                nyt_svm_score FLOAT,         -- Placeholder
                -- Add Factiva fields as placeholders if needed later, keep NULL for now
                factiva_key TEXT,
                factiva_word_count INTEGER,
                factiva_source_name TEXT,
                factiva_language TEXT,
                factiva_document_type TEXT,
                factiva_region TEXT,
                factiva_industry TEXT,
                factiva_subject TEXT,
                factiva_company_codes TEXT,
                factiva_other_metadata MAP(VARCHAR, VARCHAR) -- Flexible map for extra fields
            )
        """)
        logger.info(f"Table {nyt_table} created (if not exist).")

        # Staging table for COW state list (example)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staging.states (
                stateabb VARCHAR,
                ccode INTEGER,
                statenme VARCHAR,
                styear INTEGER,
                stmonth INTEGER,
                stday INTEGER,
                endyear INTEGER,
                endmonth INTEGER,
                endday INTEGER,
                version INTEGER
            )
        """)
        logger.info("Table staging.states created (if not exist).")

        # Staging table for bad NYT keys
        bad_keys_table = config['loading']['nyt']['bad_keys_table']
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {bad_keys_table} (
                key VARCHAR,                 -- The problematic key value
                filepath VARCHAR,            -- Source file where it was found
                reason VARCHAR,              -- e.g., 'Invalid UUID', 'Empty Key'
                first_detected_timestamp TIMESTAMP DEFAULT current_timestamp,
                UNIQUE (key, filepath, reason) -- Prevent exact duplicates
            );
        """)
        logger.info(f"Table {bad_keys_table} created (if not exist).")

        # Analytics table (example)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.mic_events (
                event_id VARCHAR PRIMARY KEY,
                event_date DATE,
                country1 VARCHAR,
                country2 VARCHAR,
                fatalities_min INTEGER,
                fatalities_max INTEGER,
                fatalities_exact INTEGER,
                confidence FLOAT,
                source_articles UBIGINT[], -- Array of IDs from raw tables
                notes TEXT
            )
        """)
        logger.info("Table analytics.mic_events created (if not exist).")

        # Create category filtering tables and macro
        create_category_filtering_tables(conn)

        # --- Create Staging Tables ---
        # Locations table for normalized location data - now after all tables are created
        locations_table = 'staging.locations'
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {locations_table} (
                location_name VARCHAR PRIMARY KEY NOT NULL
            );
        """)
        logger.info(f"Table {locations_table} created (if not exist).")
        # Don't populate locations here - will be done after loading

        # Verify all expected objects after creation
        verify_database_objects(conn)

        # Commit all schema changes before verification
        conn.commit()
        logger.info("Database structure committed.")

        # Verification step (handle errors gracefully)
        logger.info("Verifying database objects creation:")
        try:
            # Verify tables
            tables = conn.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema IN ('raw', 'staging', 'analytics')
                ORDER BY 1, 2
            """).fetchall()
            logger.info(f"Tables found: {tables}")
        except Exception as e_tables:
            logger.warning(f"Could not verify tables: {e_tables}") # Log warning instead of error

        # Verify macro existence by trying to use it in a simple query
        try:
            # Try to run a simple query using the bracket_category macro
            test_result = conn.execute("SELECT bracket_category('Test [Category]') AS category").fetchone()
            if test_result:
                logger.info("Successfully verified 'bracket_category' macro works")
            else:
                logger.warning("Macro verification query returned no results")
        except Exception as e_macro:
            logger.warning(f"Could not verify macros via test query: {e_macro}")
            logger.info("This might be normal if the macro has dependencies or requires specific input formats")

        logger.info("Successfully created and initialized DuckDB database structure.")
        return True # Return True as creation succeeded, even if verification had warnings
    except duckdb.Error as e:
        # This block handles errors during the actual CREATE/INSERT operations *before* commit
        logger.error(f"DuckDB Error during database structure creation: {e}")
        traceback.print_exc()
        if conn:
            try:
                # Attempt rollback if an error occurred *before* the commit
                # Check if transaction is active (might not be if error was early)
                if conn.in_transaction:
                     conn.rollback()
                     logger.info("Transaction rolled back due to error during creation.")
                else:
                     logger.debug("No active transaction to rollback (error likely before transaction start or after commit).")
            except AttributeError: # Handle if conn doesn't have in_transaction (older versions?)
                 logger.warning("Could not determine transaction state for rollback.")
            except Exception as rb_e:
                logger.error(f"Rollback attempt failed: {rb_e}")
        return False # Return False on actual creation errors
    except Exception as e:
        # Handles other unexpected errors during setup
        logger.error(f"Unexpected Error creating DuckDB database structure: {e}")
        traceback.print_exc()
        # Rollback might be relevant here too if error happened after connect but before commit
        if conn:
             try:
                 if conn.in_transaction:
                     conn.rollback()
                     logger.info("Transaction rolled back due to unexpected error during creation.")
                 else:
                     logger.debug("No active transaction to rollback.")
             except AttributeError:
                 logger.warning("Could not determine transaction state for rollback.")
             except Exception as rb_e:
                 logger.error(f"Rollback attempt failed: {rb_e}")
        return False
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

def verify_database_objects(conn: duckdb.DuckDBPyConnection) -> None:
    """Verifies that all expected database objects exist after creation."""
    logger.info("Verifying all expected database objects...")
    try:
        # Check for views specifically
        views = conn.execute("""
            SELECT table_schema, table_name
            FROM information_schema.views
            WHERE table_schema IN ('raw', 'staging', 'analytics')
            ORDER BY 1, 2
        """).fetchall()
        logger.info(f"Views found: {views}")
        if not views:
            logger.warning("No views found in database. Check if filtered_articles view was created.")
        
        # Test filtered_articles view exists by trying to query it
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM staging.filtered_articles LIMIT 1").fetchone()
            logger.info(f"View staging.filtered_articles exists and can be queried (row count calculation: {row_count[0]})")
        except Exception as view_test_e:
            logger.error(f"View staging.filtered_articles does not exist or cannot be queried: {view_test_e}")
    except Exception as e:
        logger.warning(f"Could not fully verify database objects: {e}")

# =============================================================================
# == ARTICLE PARSING FUNCTIONS
# =============================================================================

# --- ProQuest Parsing ---
def extract_metadata_from_proquest(text: str) -> Dict[str, Any]:
    """Extracts metadata from a single ProQuest article chunk."""
    # Keep this function largely as it was in the original pipeline_create_and_load.py
    # Ensure it returns a dictionary where keys match the columns of raw.articles
    # (excluding the 'id' column, which will be generated during insertion).
    text = text.strip()
    if not text:
        return {} # Return empty dict for empty chunks

    article = {}

    # Raw text length (excluding whitespace)
    article['raw_text_length'] = len(''.join(text.split()))

    # --- Title and Section Extraction (Refactored) ---
    title_match = re.search(r'^Title:\s*(.*?)(?=\n(?:Author:|Publication title:|Publicationtitle:|Abstract:|Full text:|$))', text, re.DOTALL | re.MULTILINE)
    raw_title_line = None
    extracted_section_from_title = None
    clean_title_candidate = None
    if title_match:
        raw_title_line = title_match.group(1).strip()
        article['title'] = raw_title_line # Store the original full title line first
        # Try extracting section from brackets at the end of the raw title line
        # Regex: (capture_title_part)(optional_colon_and_space)[capture_section_part]optional_trailing_space
        section_in_title_match = re.search(r'(.*?)(?::\s*)?\[(.*?)\]\s*$', raw_title_line)
        if section_in_title_match:
            title_part = section_in_title_match.group(1).strip()
            extracted_section_from_title = section_in_title_match.group(2).strip()
            # Clean the title part: remove potential trailing colon and junk characters
            if title_part.endswith(':'):
                title_part = title_part[:-1].strip()
            clean_title_candidate = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]+\s*$', '', title_part).strip()
            logger.debug(f"Extracted section '{extracted_section_from_title}' from title brackets. Clean title candidate: '{clean_title_candidate}'")
        else:
            # No section found in title brackets. The raw title is the basis for the clean title.
            clean_title_candidate = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]+\s*$', '', raw_title_line).strip()
            logger.debug("No section found in title brackets. Clean title candidate: '{clean_title_candidate}'")
    else:
        # Fallback: Use the first non-empty line as title if no "Title:" field found
        lines = text.splitlines()
        first_line = lines[0].strip() if lines else ""
        if first_line:
            article['title'] = first_line # Store fallback title
            # Attempt bracket extraction even on fallback title line
            section_in_fallback_match = re.search(r'(.*?)(?::\s*)?\[(.*?)\]\s*$', first_line)
            if section_in_fallback_match:
                 title_part = section_in_fallback_match.group(1).strip()
                 extracted_section_from_title = section_in_fallback_match.group(2).strip()
                 if title_part.endswith(':'):
                     title_part = title_part[:-1].strip()
                 clean_title_candidate = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]+\s*$', '', title_part).strip() # Clean fallback title
                 logger.debug(f"Extracted section '{extracted_section_from_title}' from fallback title brackets.")
            else:
                 clean_title_candidate = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]+\s*$', '', first_line).strip() # Clean fallback title
                 logger.debug("Using first line as fallback title, no section in brackets found there.")
        else:
            logger.warning("Could not find ProQuest title for chunk starting with: '%s...'", text[:50].replace('\n', ' '))
            # Assign None explicitly if no title found at all
            article['title'] = None
            clean_title_candidate = None

    # Assign the determined clean title
    article['clean_title'] = clean_title_candidate

    # --- Section Code Extraction (Separate Field) ---
    # Always try to find a separate "Section:" field, store in section_code for reference
    # This field might contain different info or be redundant, but capture it.
    article['section_code'] = safe_search(r'^Section\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    if article['section_code']:
        logger.debug(f"Found separate section code field: '{article['section_code']}'")

    # --- Final Section Assignment ---
    # Prioritize section extracted from title brackets.
    # If not found there, use the separate section_code if it exists.
    if extracted_section_from_title:
        article['section'] = extracted_section_from_title
    elif article['section_code']:
        article['section'] = article['section_code']
        logger.debug(f"Using separate section_code '{article['section_code']}' as primary section.")
    else:
        article['section'] = None # Explicitly set to None if not found anywhere
        logger.debug("No section found in title brackets or as separate section code field.")

    # --- Continue extracting other fields ---
    article['author'] = safe_search(r'^Author\s*:(.*?)(?=\n(?:Publication title:|Publicationtitle:|Abstract:|Full text:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['abstract'] = safe_search(r'^Abstract\s*:(.*?)(?=\n(?:Links\s*:|Full text\s*:|Subject\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['full_text'] = safe_search(r'^Full text\s*:(.*?)(?=\n(?:Subject\s*:|Location\s*:|People\s*:|Company\s*/\s*Org\s*:|Identifier\s*/\s*keyword\s*:|ProQuest document ID\s*:|Document URL\s*:|Copyright\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)

    # Fallback for full_text if standard header not found
    if not article.get('full_text'):
        last_header_field_end = 0
        # Find the end of the latest known header field before potential body text
        for field_pattern in [
            r'^Title:.*?\n', r'^Author\s*:.*?\n', r'^Publication title\s*:.*?\n',
            r'^Publication date\s*:.*?\n', r'^Abstract\s*:.*?\n'
            ]:
            match = re.search(field_pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if match:
                last_header_field_end = max(last_header_field_end, match.end())
        potential_body = text[last_header_field_end:].strip()
        # Find the start of the earliest known footer metadata field
        trailing_meta_start = re.search(r'\n(?:Subject\s*:|Location\s*:|People\s*:|Company\s*/\s*Org\s*:|Identifier\s*/\s*keyword\s*:|ProQuest document ID\s*:|Document URL\s*:|Copyright\s*:)', potential_body, re.IGNORECASE | re.MULTILINE)
        if trailing_meta_start:
            article['full_text'] = potential_body[:trailing_meta_start.start()].strip()
            logger.debug("Using fallback logic for full_text extraction.")
        elif potential_body:
            # If no trailing metadata found, assume rest is body (might include some metadata)
            article['full_text'] = potential_body
            logger.debug("Using fallback logic for full_text (no trailing meta detected).")
        # else: full_text remains None or empty

    # Clean "Enlarge this image." from abstract and full_text
    for key in ['abstract', 'full_text']:
        if article.get(key):
            original_len = len(article[key])
            article[key] = article[key].replace("Enlarge this image.", "").strip()
            if len(article[key]) < original_len:
                logger.debug(f"Removed 'Enlarge this image.' from {key}.")

    # Continue extracting other fields
    article['url'] = safe_search(r'^(?:Document URL|URL)\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['publication_title'] = safe_search(r'^(?:Publication title|Publicationtitle)\s*:(.*?)(?=\n(?:Pages\s*:|Publication year\s*:|Publication date\s*:|Section\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['page'] = safe_search(r'^Pages\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['publication_date_raw'] = safe_search(r'^Publication date\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['publication_date'] = article['publication_date_raw'] # Keep raw string

    # Extract publication year
    pub_year_match = safe_search(r'^Publication year\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    if pub_year_match:
        article['publication_year'] = pub_year_match
    elif article.get('publication_date_raw'):
        year_extract = re.search(r'\b(\d{4})\b', article['publication_date_raw']) # Look for 4 digits as a word
        if year_extract:
            article['publication_year'] = year_extract.group(1)
            logger.debug(f"Extracted year {article['publication_year']} from date string '{article['publication_date_raw']}'")

    article['location'] = safe_search(r'^Location\s*:(.*?)(?=\n(?:Subject\s*:|People\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['subject'] = safe_search(r'^Subject\s*:(.*?)(?=\n(?:Location\s*:|People\s*:|Company\s*/\s*Org\s*:|Identifier\s*/\s*keyword\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['people'] = safe_search(r'^People\s*:(.*?)(?=\n(?:Company\s*/\s*Org\s*:|Identifier\s*/\s*keyword\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['keywords'] = safe_search(r'^Identifier\s*/\s*keyword\s*:(.*?)(?=\n(?:ProQuest document ID\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['document_id'] = safe_search(r'^ProQuest document ID\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)

    article['place_of_publication'] = safe_search(r'^Place of publication\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['country_of_publication'] = safe_search(r'^Country of publication\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['document_type'] = safe_search(r'^Document type\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['publisher'] = safe_search(r'^Publisher\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['last_updated'] = safe_search(r'^Last updated\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['copyright'] = safe_search(r'^Copyright\s*:(.*?)(?=\n(?:ProQuest document ID\s*:|$))', text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    article['issn'] = safe_search(r'^ISSN\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['source_type'] = safe_search(r'^Source type\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['language'] = safe_search(r'^Language(?: of publication)?\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)
    article['database'] = safe_search(r'^Database\s*:(.*?)(?=\n|$)', text, re.MULTILINE | re.IGNORECASE)

    # Use publication_title as source if available
    if article.get('publication_title'):
        article['source'] = article['publication_title']

    # Basic validation: Check if essential fields like text or abstract are present
    if not article.get('full_text') and not article.get('abstract'):
        logger.warning(f"Article chunk starting with '{text[:100].replace('\n',' ')}...' seems to lack extractable full_text or abstract. Fields found: {list(article.keys())}")
        # Decide whether to return partial data or skip
        # return {} # Option: skip if no text content
        return article

    return article

def safe_search(pattern: str, text: str, flags=0) -> Optional[str]:
    """Helper for safe extraction"""
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None

# --- NYT-Style Parsing ---
def is_valid_uuid(key_str: Optional[str]) -> bool:
    """Checks if a string is a valid UUID (any version)."""
    if not key_str: # Handle None or empty strings
        return False
    try:
        uuid.UUID(key_str) # Check if it can be parsed as a UUID
        return True
    except (ValueError, TypeError): # Catch invalid format or non-string input
        return False

def parse_nyt_article(article_block: str, filepath: str, config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[str, str, str]], List[str]]:
    """
    Parses a single NYT-style article block.

    Args:
        article_block: The text content of a single article.
        filepath: The path to the source file.
        config: The application configuration dictionary.

    Returns:
        A tuple containing:
        - dict or None: Parsed data matching raw.parsed_articles schema (None if parsing fails critically, e.g., no key).
        - tuple or None: Bad key info (key_value, filepath, reason) if a key is found but invalid/empty.
        - list: A list of warning messages encountered during parsing.
    """
    nyt_config = config['loading']['nyt']
    text_start_marker = nyt_config['text_start_marker']
    text_end_marker = nyt_config['text_end_marker']

    # Target schema: id, source_filepath, format_type, format_note, headline, body,
    #                publication_date, nyt_internal_id, nyt_country_codes, ...
    data = {
        'source_filepath': filepath,
        'format_type': 'NYT_2011_2014', # Example type, could be dynamic
        'format_note': None,
        'headline': None,
        'body': None,
        'publication_date': None, # Store as YYYY-MM-DD string or None
        'nyt_internal_id': None,
        'nyt_country_codes': None,
        'nyt_source_info': None,
        'nyt_svm_score': None,
        'factiva_key': None,
        'factiva_word_count': None,
        'factiva_source_name': None,
        'factiva_language': None,
        'factiva_document_type': None,
        'factiva_region': None,
        'factiva_industry': None,
        'factiva_subject': None,
        'factiva_company_codes': None,
        'factiva_other_metadata': None
    }
    bad_key_info = None # Tuple: (key_value, filepath, reason)
    warnings = []

    lines = article_block.strip().split('\n')
    text_lines = []
    in_text_block = False
    found_start_marker = False
    found_end_marker = False
    key_value_found = None # Store the raw key value found

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # --- Metadata Extraction ---
        try:
            if stripped_line.startswith('Key:'):
                key_value_found = line.split(':', 1)[1].strip()
                data['nyt_internal_id'] = key_value_found # Store the found key value
                if not key_value_found:
                    warnings.append(f"Empty Key value found: '{line}'")
                    bad_key_info = (key_value_found, filepath, 'Empty Key')
                elif not is_valid_uuid(key_value_found):
                    warnings.append(f"Malformed Key (UUID validation failed): '{key_value_found}'")
                    bad_key_info = (key_value_found, filepath, 'Invalid UUID')
                # If valid, bad_key_info remains None

            elif stripped_line.startswith('Headline:'):
                data['headline'] = line.split(':', 1)[1].strip()

            elif stripped_line.startswith('Date:'):
                date_str = line.split(':', 1)[1].strip()
                if re.match(r'^\d{8}$', date_str):
                    # Convert YYYYMMDD to YYYY-MM-DD for DATE type
                    data['publication_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                elif not date_str:
                    warnings.append("Empty Date field found. Storing as NULL.")
                    data['publication_date'] = None # Explicitly None for DB
                else:
                    warnings.append(f"Non-standard date format: '{date_str}'. Storing as NULL.")
                    data['publication_date'] = None # Store NULL if format is wrong

            elif stripped_line.startswith('Countries:'):
                data['nyt_country_codes'] = line.split(':', 1)[1].strip()

            # --- Full Text Extraction ---
            elif stripped_line.startswith(text_start_marker):
                in_text_block = True
                found_start_marker = True
                continue # Skip the marker line
            elif stripped_line.startswith(text_end_marker):
                in_text_block = False
                found_end_marker = True
                # Continue processing lines after text block
            if in_text_block:
                text_lines.append(line) # Append original line

        except IndexError:
            # Handle lines like "Key:" with nothing after
            field_name = stripped_line.split(':', 1)[0]
            warnings.append(f"Could not parse value for field '{field_name}'. Line: '{line}'")
            if field_name == 'Key':
                 # If the key line itself is malformed, treat it as a bad key scenario
                 key_value_found = "" # Treat as empty
                 data['nyt_internal_id'] = key_value_found
                 bad_key_info = (key_value_found, filepath, 'Malformed Key Line')

    # --- Post-processing Text ---
    if text_lines:
        data['body'] = '\n'.join(text_lines).strip()
    else:
        # Add warnings based on marker presence/absence
        key_for_warn = data.get('nyt_internal_id', 'N/A')
        if found_start_marker and found_end_marker:
             data['body'] = "" # Explicitly empty string
             warnings.append(f"Text block markers found but no text content between them for key '{key_for_warn}'")
        elif found_start_marker and not found_end_marker:
             warnings.append(f"Text block START marker found but NO END marker for key '{key_for_warn}'")
             # Capture partial text if start marker found but no end marker
             data['body'] = '\n'.join(text_lines).strip()
             if data['body']:
                 warnings.append(f"Partial text captured for key '{key_for_warn}' due to missing end marker.")
        elif not found_start_marker and found_end_marker:
             warnings.append(f"Text block END marker found but NO START marker for key '{key_for_warn}'")
        # else: # Neither marker found - don't warn here, maybe no text was expected

    # --- Final Checks ---
    if data['nyt_internal_id'] is None: # Check if 'Key:' line was completely missing
        if not any("Could not parse value for field 'Key'" in w for w in warnings): # Avoid duplicate warnings
             warnings.append("No 'Key:' line found in this article block.")
        # Cannot proceed without a key, return None for data
        return None, None, warnings

    # If a key was found, return the data (even if key is bad), bad_key_info tuple, and warnings
    return data, bad_key_info, warnings

# =============================================================================
# == PARALLEL LOADING FUNCTIONS
# =============================================================================

def process_file_worker(args_tuple: Tuple[str, str, str, Dict[str, Any], List[str], List[str]]) -> Optional[Tuple[str, str, Optional[List[Tuple[str, str, str]]]]]:
    """
    Worker function: Parses one file (ProQuest or NYT), writes results to temp Parquet.

    Args:
        args_tuple: Contains:
            - file_path (str): Path to the input text file.
            - file_type (str): 'proquest' or 'nyt'.
            - temp_dir (str): Path to the temporary directory for Parquet files.
            - config (Dict): The application configuration.
            - proquest_cols (List[str]): List of columns for ProQuest schema.
            - nyt_cols (List[str]): List of columns for NYT schema.

    Returns:
        Tuple containing:
        - temp_filepath (str): Path to the created Parquet file. Returns None if the file cannot be read or no valid articles are parsed.
        - data_type (str): 'proquest' or 'nyt'.
        - bad_keys (Optional[List[Tuple[str, str, str]]]): List of bad key info tuples for NYT files, None otherwise.
    """
    file_path, file_type, temp_dir, config, proquest_cols, nyt_cols = args_tuple
    worker_pid = os.getpid()
    basename = os.path.basename(file_path)
    logger.info(f"[Worker {worker_pid}] Processing {file_type} file: {basename}")

    try:
        # --- 1. Read File Content ---
        detected_encoding = None
        try:
            # Try detecting encoding quickly
            with open(file_path, 'rb') as fb:
                raw_head = fb.read(8192) # Read first 8KB
                detected = chardet.detect(raw_head)
                # Be stricter with confidence for auto-detection
                if detected['encoding'] and detected['confidence'] > 0.9:
                    detected_encoding = detected['encoding']
                logger.debug(f"[Worker {worker_pid}] Detected encoding {detected_encoding} for {basename}")
        except Exception as detect_e:
            logger.warning(f"[Worker {worker_pid}] Encoding detection failed for {basename}: {detect_e}")

        content = None
        # Always try utf-8 first, then detected encoding, then fallbacks
        encodings_to_try = ['utf-8']
        if detected_encoding and detected_encoding.lower() != 'utf-8':
            encodings_to_try.append(detected_encoding)
        for enc in ENCODING_FALLBACKS:
            if enc.lower() not in [e.lower() for e in encodings_to_try]:
                encodings_to_try.append(enc)

        MAX_ENCODING_ERRORS_TO_LOG = 100
        # Counter for encoding errors (limit to 100)
        encoding_error_count = 0
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='strict') as f:
                    content = f.read()
                logger.info(f"[Worker {worker_pid}] File {basename} processed using encoding: {enc} (confidence: {detected.get('confidence', 'N/A') if detected_encoding == enc else 'fallback'})")
                break # Stop trying once read successfully
            except UnicodeDecodeError as ude:
                # Log detailed info about the first 100 encoding errors
                if encoding_error_count < MAX_ENCODING_ERRORS_TO_LOG:
                    error_pos = getattr(ude, 'start', 'unknown')
                    error_object = getattr(ude, 'object', b'unknown')
                    error_reason = str(ude)
                    if isinstance(error_object, bytes) and error_pos != 'unknown':
                        try:
                            # Show up to 20 bytes around the error position
                            context_start = max(0, error_pos - 10)
                            context_end = min(len(error_object), error_pos + 10)
                            context_bytes = error_object[context_start:context_end]
                            hex_bytes = ' '.join(f'{b:02x}' for b in context_bytes)
                            logger.warning(f"[Worker {worker_pid}] Encoding error #{encoding_error_count+1} in {basename} with {enc}: at position {error_pos}, bytes: {hex_bytes}")
                        except Exception as hex_e:
                            logger.warning(f"[Worker {worker_pid}] Encoding error #{encoding_error_count+1} in {basename} with {enc}: {error_reason}")
                    else:
                        logger.warning(f"[Worker {worker_pid}] Encoding error #{encoding_error_count+1} in {basename} with {enc}: {error_reason}")
                    encoding_error_count += 1
                logger.debug(f"[Worker {worker_pid}] Failed to read {basename} with encoding {enc}")
                continue # Try next encoding
            except Exception as read_e:
                logger.error(f"[Worker {worker_pid}] Error reading {basename} with encoding {enc}: {read_e}")
                continue # Try next encoding

        # If strict mode failed for all encodings, try again with 'replace' for utf-8
        if content is None:
            try:
                logger.info(f"[Worker {worker_pid}] Retrying {basename} with utf-8 encoding and error replacement")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                logger.info(f"[Worker {worker_pid}] Successfully read {basename} with utf-8 (with character replacement)")
            except Exception as fallback_e:
                logger.error(f"[Worker {worker_pid}] Failed to read {basename} even with error replacement: {fallback_e}")

        if content is None:
            logger.error(f"[Worker {worker_pid}] Failed to read {basename} with any attempted encoding ({encodings_to_try}). Skipping.")
            return None

        if encoding_error_count > 0:
            logger.warning(f"[Worker {worker_pid}] File {basename} had {encoding_error_count} encoding errors before successful read")

        # --- 2. Split into Articles/Chunks ---
        parsed_articles = []
        all_bad_keys_in_file = [] # Specific to NYT parsing
        warnings_in_file = []

        if file_type == 'proquest':
            pq_separator = config['loading']['proquest'].get('separator', PROQUEST_SEPARATOR) # Get from config or use default
            chunks = content.split(pq_separator)
            # ProQuest often has header/footer text outside the separators
            # Heuristic: Skip first 2 and last 1 chunk if separator is present multiple times
            article_chunks = chunks[2:-1] if len(chunks) > 3 else [c for c in chunks if c.strip()]
            if not article_chunks:
                 logger.warning(f"[Worker {worker_pid}] No article chunks found using separator in ProQuest file {basename}.")
                 return None # Skip file if standard chunking fails
                 # Fallback: treat whole file as one chunk? Depends on format variability.
                 # article_chunks = [content]
            parser_func = extract_metadata_from_proquest
            target_cols = proquest_cols
            for i, article_text in enumerate(article_chunks):
                article_text = article_text.strip()
                if not article_text: continue
                try:
                    metadata = parser_func(article_text)
                    if metadata: # Ensure parser didn't return empty dict
                        # Add file_path if it's a target column
                        if 'file_path' in target_cols:
                            metadata['file_path'] = file_path
                        # Filter metadata to only include target columns
                        filtered_metadata = {col: metadata.get(col) for col in target_cols}
                        parsed_articles.append(filtered_metadata)
                    # else: Parser decided to skip this chunk (e.g., no title/text)
                except Exception as parse_e:
                    logger.warning(f"[Worker {worker_pid}] Error parsing ProQuest chunk {i+1} in {basename}: {parse_e}", exc_info=False) # Log exception without full traceback for brevity
                    # Optionally log traceback in debug mode: logger.debug("Traceback:", exc_info=True)
                    continue # Skip chunk on error

        elif file_type == 'nyt':
            nyt_separator = config['loading']['nyt']['article_separator']
            article_blocks = content.split(nyt_separator)
            parser_func = parse_nyt_article
            target_cols = nyt_cols
            for i, article_block in enumerate(article_blocks):
                block_content = article_block.strip()
                if not block_content: continue
                try:
                    parsed_data, bad_key_info, warnings = parser_func(block_content, file_path, config)
                    if warnings:
                        warnings_in_file.extend([f"[Block ~{i+1}] {w}" for w in warnings])
                    if parsed_data: # Parsing returned data (key was present)
                        # Filter data to only include target columns
                        filtered_data = {col: parsed_data.get(col) for col in target_cols}
                        parsed_articles.append(filtered_data)
                        if bad_key_info: # If key was present but invalid/empty
                            all_bad_keys_in_file.append(bad_key_info)
                    # else: Parsing failed critically (e.g., no key), warnings logged by parser
                except Exception as parse_e:
                    logger.warning(f"[Worker {worker_pid}] Error parsing NYT block {i+1} in {basename}: {parse_e}", exc_info=False)
                    continue # Skip block on error

        else:
            logger.error(f"[Worker {worker_pid}] Unknown file type '{file_type}' for {basename}. Skipping.")
            return None

        # Log accumulated warnings for the file (if any)
        if warnings_in_file:
             # Log first few warnings clearly, summarize if many
             max_warnings_to_log = 5
             for idx, warn_msg in enumerate(warnings_in_file):
                 if idx < max_warnings_to_log:
                     logger.warning(f"[Worker {worker_pid}] Parser Warning in {basename}: {warn_msg}")
                 elif idx == max_warnings_to_log:
                     logger.warning(f"[Worker {worker_pid}] ... ({len(warnings_in_file) - max_warnings_to_log} more warnings suppressed for {basename})")
                     break

        if not parsed_articles:
            logger.warning(f"[Worker {worker_pid}] No articles successfully parsed from {basename}. Skipping file.")
            return None

        # --- 3. Prepare Data for Arrow/Parquet ---
        try:
            data_for_arrow = {}
            # Ensure all target columns exist as keys in the dictionary
            for col in target_cols:
                col_data = [article.get(col) for article in parsed_articles]

                # Specific type conversions needed before creating table
                # Example for ProQuest raw_text_length (already handled in original)
                if file_type == 'proquest' and col == 'raw_text_length':
                     col_data = [int(x) if x is not None else None for x in col_data]
                # Example for NYT publication_date (needs to be date type for Parquet)
                elif file_type == 'nyt' and col == 'publication_date':
                     # Convert 'YYYY-MM-DD' strings to date objects
                     converted_dates = []
                     for date_str in col_data:
                         if isinstance(date_str, str):
                             try:
                                 converted_dates.append(datetime.datetime.strptime(date_str, '%Y-%m-%d').date())
                             except ValueError:
                                 converted_dates.append(None) # Handle conversion errors
                         else:
                             converted_dates.append(None) # Handle None or unexpected types
                     col_data = converted_dates

                data_for_arrow[col] = col_data

            # Define schema explicitly for robustness, especially for NYT dates
            if file_type == 'nyt':
                 schema_fields = []
                 for col in target_cols:
                     pa_type = pa.string() # Default to string
                     if col == 'publication_date':
                         pa_type = pa.date32() # Use Arrow date type
                     elif col == 'nyt_svm_score':
                         pa_type = pa.float32() # Example float type
                     elif col == 'factiva_word_count':
                         pa_type = pa.int32() # Example integer type
                     # Add other type mappings as needed
                     schema_fields.append(pa.field(col, pa_type))
                 schema = pa.schema(schema_fields)
                 arrow_table = pa.Table.from_pydict(data_for_arrow, schema=schema)
            else: # ProQuest - infer schema or define explicitly if needed
                 arrow_table = pa.Table.from_pydict(data_for_arrow)

        except Exception as arrow_e:
            logger.error(f"[Worker {worker_pid}] Error creating Arrow table for {basename} ({file_type}): {arrow_e}", exc_info=True)
            return None

        # --- 4. Write Temporary Parquet File ---
        temp_filename = f"{file_type}_{uuid.uuid4()}.parquet"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        try:
            pq.write_table(arrow_table, temp_filepath, compression='snappy') # Use compression
            logger.info(f"[Worker {worker_pid}] Wrote {len(parsed_articles)} {file_type} articles from {basename} to {temp_filename}")
            # Return path, type, and any bad keys found (only relevant for NYT)
            return temp_filepath, file_type, all_bad_keys_in_file if file_type == 'nyt' else None
        except Exception as write_e:
            logger.error(f"[Worker {worker_pid}] Error writing Parquet file {temp_filename}: {write_e}", exc_info=True)
            # Attempt to clean up the potentially corrupted temp file
            try:
                if os.path.exists(temp_filepath): os.unlink(temp_filepath)
            except Exception: pass
            return None

    except Exception as worker_e:
        logger.error(f"[Worker {worker_pid}] Unhandled error processing {basename} ({file_type}): {worker_e}", exc_info=True)
        return None

def load_articles_parallel(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    """
    Loads articles in parallel using multiprocessing. Handles both ProQuest and NYT types.
    """
    # Add these debug statements at the start of load_articles_parallel function
    db_path = config['database']['path']
    overall_success = True
    conn = None
    temp_dir_path = None

    try:
        # --- 1. Identify Files to Process ---
        files_to_process = []  # List of tuples: (filepath, file_type)
        raw_data_dir = Path(config['data']['raw_dir'])
        logger.info(f"Using raw_data_dir: {raw_data_dir}")
        logger.info(f"Database path is: {db_path}")

        # Process ProQuest files if enabled
        if config['loading']['proquest']['enabled']:
            pq_config = config['loading']['proquest']
            pq_source_dir = raw_data_dir / pq_config['source_subdir'] if pq_config['source_subdir'] else raw_data_dir
            pq_prefix = pq_config['filename_prefix']
            pq_recursive = pq_config['recursive']
            pq_excluded_rel = pq_config.get('excluded_subdirs', []) or []
            pq_excluded_abs = {(pq_source_dir / p).resolve() for p in pq_excluded_rel}
            logger.info(f"Scanning for ProQuest files in: {pq_source_dir} (Prefix: '{pq_prefix}', Recursive: {pq_recursive})")
            if not pq_source_dir.is_dir():
                logger.warning(f"ProQuest source directory not found: {pq_source_dir}. Skipping ProQuest loading.")
            else:
                file_iterator = pq_source_dir.rglob('*.txt') if pq_recursive else pq_source_dir.glob('*.txt')
                count = 0
                skipped_excluded = 0
                skipped_prefix = 0
                for file_path in file_iterator:
                    if not file_path.is_file(): continue

                    # Check exclusion based on absolute path
                    is_excluded = False
                    for excluded_path in pq_excluded_abs:
                        try:  # Use resolve() to handle symlinks etc. consistently
                            if file_path.resolve().is_relative_to(excluded_path):
                                is_excluded = True
                                break
                        except ValueError:  # Not relative
                            pass
                        except Exception as path_e:  # Filesystem errors, permissions etc.
                            logger.warning(f"Path comparison error for {file_path} against {excluded_path}: {path_e}")
                    if is_excluded:
                        skipped_excluded += 1
                        continue

                    if file_path.name.startswith(pq_prefix):
                        files_to_process.append((str(file_path), 'proquest'))
                        count += 1
                    else:
                        # Avoid warning for common non-data files
                        if file_path.name.lower() not in ['readme.txt', 'readme.md', '.ds_store']:
                            skipped_prefix += 1
                            logger.debug(f"Skipping file (doesn't match ProQuest prefix '{pq_prefix}'): {file_path.name}")
                if skipped_prefix > 0: logger.info(f"Skipped {skipped_prefix} files in ProQuest dir (prefix mismatch).")
                logger.info(f"Found {count} ProQuest files matching criteria.")
                if skipped_excluded > 0: logger.info(f"Skipped {skipped_excluded} ProQuest files due to exclusion rules.")

        # Process NYT files if enabled
        if config['loading']['nyt']['enabled']:
            nyt_config = config['loading']['nyt']
            nyt_source_dir = raw_data_dir / nyt_config['source_subdir'] if nyt_config['source_subdir'] else raw_data_dir
            nyt_prefix = nyt_config['filename_prefix']
            nyt_recursive = nyt_config['recursive']
            logger.info(f"Scanning for NYT files in: {nyt_source_dir} (Prefix: '{nyt_prefix}', Recursive: {nyt_recursive})")
            if not nyt_source_dir.is_dir():
                logger.warning(f"NYT source directory not found: {nyt_source_dir}. Skipping NYT loading.")
            else:
                file_iterator = nyt_source_dir.rglob('*.txt') if nyt_recursive else nyt_source_dir.glob('*.txt')
                count = 0
                skipped_prefix = 0
                for file_path in file_iterator:
                    if not file_path.is_file(): continue
                    if file_path.name.startswith(nyt_prefix):
                        files_to_process.append((str(file_path), 'nyt'))
                        count += 1
                    else:
                        # Avoid warning for common non-data files
                        if file_path.name.lower() not in ['readme.txt', 'readme.md', '.ds_store']:
                            skipped_prefix += 1
                            logger.debug(f"Skipping file (doesn't match NYT prefix '{nyt_prefix}'): {file_path.name}")
                if skipped_prefix > 0: logger.info(f"Skipped {skipped_prefix} files in NYT dir (prefix mismatch).")
                logger.info(f"Found {count} NYT files matching criteria.")

        if not files_to_process:
            logger.warning("No files identified for processing based on current configuration. Exiting loading step.")
            return True  # Nothing to do is considered success

        # --- 2. Prepare for Parallel Processing ---
        temp_dir_name = f"mic_load_temp_{uuid.uuid4()}"
        temp_dir_path = project_root / "temp" / temp_dir_name  # Place temp in project root/temp
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_dir_path}")

        # Get target table column names (excluding 'id')
        try:
            conn_schema = duckdb.connect(db_path, read_only=True)
            proquest_cols_all = [col[0] for col in conn_schema.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{config['loading']['proquest']['target_table'].split('.')[-1]}' AND table_schema = '{config['loading']['proquest']['target_table'].split('.')[0]}' ORDER BY ordinal_position;").fetchall()]
            nyt_cols_all = [col[0] for col in conn_schema.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{config['loading']['nyt']['target_table'].split('.')[-1]}' AND table_schema = '{config['loading']['nyt']['target_table'].split('.')[0]}' ORDER BY ordinal_position;").fetchall()]
            conn_schema.close()
            if not proquest_cols_all: raise ValueError(f"Failed to retrieve columns for ProQuest table {config['loading']['proquest']['target_table']}")
            if not nyt_cols_all: raise ValueError(f"Failed to retrieve columns for NYT table {config['loading']['nyt']['target_table']}")

            # Exclude 'id' column as it's generated during insert
            proquest_cols_for_worker = [col for col in proquest_cols_all if col.lower() != 'id']
            nyt_cols_for_worker = [col for col in nyt_cols_all if col.lower() != 'id']
            logger.debug(f"ProQuest columns for worker: {proquest_cols_for_worker}")
            logger.debug(f"NYT columns for worker: {nyt_cols_for_worker}")

        except Exception as e:
            logger.error(f"CRITICAL: Could not get target table schemas from {db_path}. Error: {e}", exc_info=True)
            overall_success = False
            # Cleanup temp dir before exiting
            if temp_dir_path and temp_dir_path.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir_path)
                    logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
                except Exception as cleanup_e:
                    logger.warning(f"Could not fully clean up {temp_dir_path}: {cleanup_e}")
            return False

        # Prepare tasks for the pool
        tasks = [(fp, ftype, str(temp_dir_path), config, proquest_cols_for_worker, nyt_cols_for_worker)
                 for fp, ftype in files_to_process]

        intermediate_results = []  # List of tuples: (temp_filepath, data_type, bad_keys_list)
        num_workers = config['system']['parallel_workers']
        logger.info(f"Starting multiprocessing pool with {num_workers} workers.")
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(process_file_worker, tasks)
            with tqdm(total=len(tasks), desc="Parsing files in parallel", unit="file") as pbar:
                for result in results_iterator:
                    if result:  # Worker returned successfully
                        intermediate_results.append(result)
                    # Else: Worker failed, error logged by worker itself
                    pbar.update(1)

        logger.info(f"Parallel processing complete. {len(intermediate_results)} intermediate result sets generated.")

        if not intermediate_results:
            logger.warning("No intermediate files created. No data to load.")
            # Cleanup temp dir
            if temp_dir_path and temp_dir_path.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir_path)
                    logger.info(f"Cleaned up temporary directory: {temp_dir_path}")
                except Exception as cleanup_e:
                    logger.warning(f"Could not fully clean up {temp_dir_path}: {cleanup_e}")
            return True

        # --- 4. Bulk Load into DuckDB ---
        logger.info("Starting bulk load into DuckDB...")
        conn = duckdb.connect(db_path)
        # Separate results by data type
        proquest_files = [res[0] for res in intermediate_results if res[1] == 'proquest']
        nyt_files = [res[0] for res in intermediate_results if res[1] == 'nyt']
        all_bad_keys = []
        for res in intermediate_results:
            if res[1] == 'nyt' and res[2]:  # If NYT result and bad keys list is not empty
                all_bad_keys.extend(res[2])

        # --- Process ProQuest Files (if any) ---
        if proquest_files:
            logger.info(f"Processing {len(proquest_files)} ProQuest files...")
            conn.begin()  # Start ProQuest transaction
            try:
                pq_target_table = config['loading']['proquest']['target_table']
                pq_target_cols_str = ", ".join([f'"{col}"' for col in proquest_cols_for_worker])
                # Create a temporary table to consolidate all ProQuest data
                temp_pq_table = f"temp_proquest_load_{uuid.uuid4().hex}"
                logger.info(f"Creating temporary table '{temp_pq_table}' to consolidate ProQuest data")
                conn.execute(f"CREATE TEMP TABLE {temp_pq_table} AS SELECT * FROM read_parquet({proquest_files}, union_by_name=True)")
                temp_pq_count = conn.execute(f"SELECT COUNT(*) FROM {temp_pq_table}").fetchone()[0]
                logger.info(f"Consolidated {temp_pq_count} ProQuest records into temporary table")
                # Get max ID from target table
                max_id_current = conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {pq_target_table}").fetchone()[0]
                logger.info(f"Current max ID in {pq_target_table}: {max_id_current}")
                # Insert from temp table with sequential IDs
                conn.execute(f"""
                    INSERT INTO {pq_target_table} (id, {pq_target_cols_str})
                    SELECT {max_id_current} + row_number() OVER () AS id,
                        {pq_target_cols_str}
                    FROM {temp_pq_table}
                """)
                # Verify insertion
                final_count = conn.execute(f"SELECT COUNT(*) FROM {pq_target_table}").fetchone()[0]
                inserted_count = final_count - (final_count - temp_pq_count)
                logger.info(f"Successfully inserted {inserted_count} ProQuest articles into {pq_target_table}")
                # Cleanup temp table
                conn.execute(f"DROP TABLE IF EXISTS {temp_pq_table}")
                conn.commit()
                logger.info(f"ProQuest data load committed successfully.")
            except Exception as pq_e:
                logger.error(f"Error during ProQuest data load: {pq_e}", exc_info=True)
                conn.rollback()
                overall_success = False
        else:
            logger.info("No ProQuest data to load.")

        # --- Process NYT Files (if any) ---
        if nyt_files:
            logger.info(f"Processing {len(nyt_files)} NYT files...")
            conn.begin()  # Start NYT transaction
            try:
                nyt_target_table = config['loading']['nyt']['target_table']
                nyt_target_cols_str = ", ".join([f'"{col}"' for col in nyt_cols_for_worker])
                # Create a temporary table to consolidate all NYT data
                temp_nyt_table = f"temp_nyt_load_{uuid.uuid4().hex}"
                logger.info(f"Creating temporary table '{temp_nyt_table}' to consolidate NYT data")
                conn.execute(f"CREATE TEMP TABLE {temp_nyt_table} AS SELECT * FROM read_parquet({nyt_files}, union_by_name=True)")
                temp_nyt_count = conn.execute(f"SELECT COUNT(*) FROM {temp_nyt_table}").fetchone()[0]
                logger.info(f"Consolidated {temp_nyt_count} NYT records into temporary table")
                # Get max ID from target table
                max_id_current = conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {nyt_target_table}").fetchone()[0]
                logger.info(f"Current max ID in {nyt_target_table}: {max_id_current}")
                # Insert from temp table with sequential IDs
                conn.execute(f"""
                    INSERT INTO {nyt_target_table} (id, {nyt_target_cols_str})
                    SELECT {max_id_current} + row_number() OVER () AS id,
                        {nyt_target_cols_str}
                    FROM {temp_nyt_table}
                """)
                # Verify insertion
                final_count = conn.execute(f"SELECT COUNT(*) FROM {nyt_target_table}").fetchone()[0]
                inserted_count = final_count - (max_id_current > 0)  # Adjust if table was initially non-empty
                logger.info(f"Successfully inserted {inserted_count} NYT articles into {nyt_target_table}")
                # Cleanup temp table
                conn.execute(f"DROP TABLE IF EXISTS {temp_nyt_table}")
                conn.commit()
                logger.info(f"NYT data load committed successfully.")
            except Exception as nyt_e:
                logger.error(f"Error during NYT data load: {nyt_e}", exc_info=True)
                conn.rollback()
                overall_success = False
        else:
            logger.info("No NYT data to load.")

        # --- Process Bad Keys (in separate transaction) ---
        if all_bad_keys:
            logger.info(f"Processing {len(all_bad_keys)} bad keys...")
            conn.begin()  # Start bad keys transaction
            try:
                bad_keys_table = config['loading']['nyt']['bad_keys_table']
                logger.info(f"Storing bad key references into {bad_keys_table}...")
                # Prepare data: list of tuples (key, filepath, reason)
                bad_keys_tuples = [(bk[0], bk[1], bk[2]) for bk in all_bad_keys if len(bk) == 3]
                # Use INSERT OR IGNORE to avoid errors on duplicates
                conn.executemany(f"INSERT OR IGNORE INTO {bad_keys_table} (key, filepath, reason) VALUES (?, ?, ?)", bad_keys_tuples)
                conn.commit()
                logger.info(f"Successfully stored {len(bad_keys_tuples)} bad key references")
            except Exception as bk_e:
                logger.error(f"Error during bad keys processing: {bk_e}", exc_info=True)
                conn.rollback()
                # Don't set overall_success to False, as bad keys are secondary
        else:
            logger.info("No bad keys to process.")

        return overall_success

    except Exception as e:
        logger.error(f"Critical error during parallel article loading setup or execution: {e}", exc_info=True)
        return False

    finally:
        # --- 5. Cleanup ---
        if conn:
            try:
                conn.close()
                logger.info("Database connection closed after loading.")
            except Exception:
                pass
        if temp_dir_path and temp_dir_path.exists():
            try:
                logger.info(f"Cleaning up temporary directory: {temp_dir_path}")
                # Use shutil.rmtree for recursive deletion
                import shutil
                shutil.rmtree(temp_dir_path)
                logger.info("Temporary directory cleaned up successfully.")
            except Exception as cleanup_e:
                logger.warning(f"Could not fully clean up temporary directory {temp_dir_path}: {cleanup_e}")

def populate_locations_table(conn, config):
    """Populates staging.locations table with distinct locations from raw.articles"""
    locations_table = 'staging.locations'
    logger.info(f"Populating {locations_table} table with locations from raw.articles...")
    
    # Check if source table has data before attempting to populate locations
    has_articles = conn.execute("SELECT EXISTS(SELECT 1 FROM raw.articles LIMIT 1)").fetchone()[0]
    if has_articles:
        logger.info("Found data in raw.articles, populating staging.locations...")
        conn.execute(f"TRUNCATE TABLE {locations_table};")
        conn.execute(f"""
            INSERT INTO {locations_table} (location_name)
            SELECT DISTINCT trim(value) as location_name
            FROM raw.articles a,
                unnest(string_split(a.location, ';')) as t(value)
            WHERE trim(value) != ''  -- Skip empty values
            ORDER BY location_name;
        """)
        logger.info(f"Table {locations_table} populated with {conn.execute(f'SELECT COUNT(*) FROM {locations_table}').fetchone()[0]} unique locations.")
        return True
    else:
        logger.info("No data in raw.articles yet, staging.locations will remain empty.")
        return False

# =============================================================================
# == MAIN EXECUTION LOGIC
# =============================================================================

def parse_args():
    """Parse command line arguments for combined script."""
    parser = argparse.ArgumentParser(
        description='Create/Update DuckDB structure and load ProQuest/NYT articles in parallel.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    # Add arguments from config module first
    parser = add_config_args(parser)

    # Add script-specific arguments
    parser.add_argument('--force', action='store_true',
                        help='Automatically delete existing database file without prompting.')
    parser.add_argument('--skip-db-creation', action='store_true',
                        help='Skip the database structure creation/update step.')
    parser.add_argument('--skip-loading', action='store_true',
                        help='Skip the article loading step.')
    # --single-file argument removed as parallel loading handles multiple files efficiently.
    # Add it back if specifically needed for debugging one file.

    return parser.parse_args()

def main(args=None):
    """Main function to create database structure and load articles."""
    script_start_time = datetime.datetime.now()
    logger.info("========================================================")
    logger.info("Starting Combined Database Creation and Loading Pipeline")
    logger.info(f"Start Time: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("========================================================")

    if args is None:
        args = parse_args()
    config = get_config(args) # Load config considering args

    db_path = config['database']['path']
    db_file = Path(db_path)
    wal_file = Path(f"{db_path}.wal")

    # --- Database Deletion Check ---
    if db_file.exists() and not args.skip_db_creation:
        if args.force:
            logger.warning(f"Database file '{db_path}' exists. --force specified, deleting automatically.")
            try:
                if db_file.exists(): db_file.unlink()
                if wal_file.exists(): wal_file.unlink() # Delete WAL file too
                logger.info(f"Deleted existing database file(s): {db_path} (.wal)")
            except OSError as e:
                logger.error(f"Error deleting existing database file(s): {e}. Please check permissions.")
                sys.exit(1)
        else:
            prompt = input(f"Database file '{db_path}' already exists. "
                           f"Delete it and recreate structure? (yes/No): ")
            if prompt.lower() in ['y', 'yes']:
                logger.info("User chose to delete existing database.")
                try:
                    if db_file.exists(): db_file.unlink()
                    if wal_file.exists(): wal_file.unlink()
                    logger.info(f"Deleted existing database file(s): {db_path} (.wal)")
                except OSError as e:
                    logger.error(f"Error deleting existing database file(s): {e}. Please check permissions.")
                    sys.exit(1)
            else:
                logger.info("Proceeding with existing database file. Structure might not be updated if --skip-db-creation is not used.")
                # If user says no, we *don't* skip creation by default, allowing updates.
                # If they want to keep the DB *and* skip updates, they need --skip-db-creation.

    # --- Database Creation Step ---
    if not args.skip_db_creation:
        logger.info("--- Running Database Creation/Update Step ---")
        creation_success = create_database_structure(db_path, config)
        if not creation_success:
            logger.error("Database structure creation/update failed. Aborting.")
            sys.exit(1)
        logger.info("--- Database Creation/Update Step Complete ---")
    else:
        logger.info("--- Skipping Database Creation/Update Step (as requested) ---")
        if not db_file.exists():
             logger.error(f"Database file '{db_path}' does not exist, but creation was skipped. Cannot proceed.")
             sys.exit(1)

    # --- Article Loading Step ---
    proquest_enabled = config['loading']['proquest']['enabled']
    nyt_enabled = config['loading']['nyt']['enabled']
    if not args.skip_loading and (proquest_enabled or nyt_enabled):
        logger.info("--- Running Parallel Article Loading Step ---")
        load_success = load_articles_parallel(config, args)
        if not load_success:
            logger.error("Parallel article loading step finished with errors.")
            # Decide if this should be a fatal error (sys.exit(1))
            # For now, just log error and continue to show summary.
        else:
             logger.info("--- Parallel Article Loading Step Complete ---")
    elif args.skip_loading:
        logger.info("--- Skipping Article Loading Step (as requested) ---")
    else:
         logger.info("--- Skipping Article Loading Step (ProQuest and NYT loading disabled in config/args) ---")

    # --- Populate staging.locations table if articles were loaded ---
    if not args.skip_loading and (proquest_enabled or nyt_enabled) and load_success:
        logger.info("--- Populating Locations Table After Loading ---")
        try:
            conn = duckdb.connect(db_path)
            populate_success = populate_locations_table(conn, config)
            if populate_success:
                logger.info("Successfully populated locations table from loaded articles.")
            else:
                logger.warning("No articles found for populating locations table.")
            conn.close()
        except Exception as e:
            logger.error(f"Error populating locations table: {e}", exc_info=True)
    
    # --- Final Summary ---
    script_end_time = datetime.datetime.now()
    duration = script_end_time - script_start_time
    logger.info("========================================================")
    logger.info("Pipeline Finished")
    logger.info(f"End Time: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Duration: {duration}")
    logger.info("========================================================")

if __name__ == "__main__":
    # freeze_support() might be needed on Windows/macOS if using certain start methods
    # or packaging the script. Uncomment if you encounter related issues.
    # from multiprocessing import freeze_support
    # freeze_support()

    main()