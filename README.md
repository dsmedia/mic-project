# MIC Project

A pipeline for downloading, processing, and analyzing news articles to extract Militarized Interstate Confrontation (MIC) events.

## Overview

This project provides a comprehensive pipeline to:

1. Download news articles from a zip file corpus
2. Create a DuckDB database structure with appropriate tables and views
3. Load articles into the database with filtering capabilities
4. Generate responses/analysis of articles using Gemini 2.5 Pro to identify MIC events
5. Create a training dataset from the analysis results

## Requirements

The project requires Python 3.8+ and the following packages:

```
pip install -r requirements.txt
```

## Setup

1. Clone this repository
2. Install the requirements
3. Create a `.env` file with your Gemini API key (for response generation):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Complete Pipeline

To run the entire pipeline at once, use the main script:

```bash
python run_pipeline.py
```

This will execute all steps in sequence, skipping any steps where output files already exist.

Command-line options:
- `--skip-download`: Skip the article download step
- `--skip-database`: Skip the database creation/loading step
- `--skip-responses`: Skip the response generation step
- `--skip-dataset`: Skip the training dataset creation step
- `--force-all`: Force all steps to run, even if output files exist

Example:
```bash
# Run only the dataset creation step
python run_pipeline.py --skip-download --skip-database --skip-responses

# Force all steps to run
python run_pipeline.py --force-all
```

### Running Individual Steps

Alternatively, you can run each step individually in sequence:

#### 1. Download Articles

```bash
python src/data/download_articles.py
```

This script downloads the corpus zip file if it doesn't already exist locally and extracts it to the `data/raw` directory.

#### 2. Create Database and Load Articles

```bash
python src/data/pipeline_create_and_load.py
```

This script creates the DuckDB database structure and loads articles from the extracted files. It includes:

- Creating schemas: raw, staging, and analytics
- Creating tables for articles, parsed articles, states, locations, etc.
- Creating category filtering tables and views
- Loading articles in parallel from ProQuest and NYT-style text files

##### SQL Filtering

The script supports article filtering via SQL through the `staging.filtered_articles` view. This view applies multiple filter criteria to focus on articles that are relevant to MIC events:

1. **Category Filtering**: Excludes articles from certain categories (e.g., Style, Sports, Obituaries)
2. **Subject Filtering**: Focuses on articles with relevant subjects (e.g., War, Military, Weapons)
3. **Location Filtering**: Ensures non-domestic locations are present (international focus)

The actual SQL query (defined in `config/category_filtering.yaml`) is:

```sql
SELECT id,
  strftime(TRY_STRPTIME(publication_date, '%b %d, %Y')::DATE,'%a., %b. %-d, %Y') as publication_date,
  COALESCE(location, 'N/A') as location,
  COALESCE(subject, 'N/A') as subject,
  COALESCE(people, 'N/A') as people,
  coalesce(full_text, '[TEXT MISSING]') as full_text
FROM raw.articles a
WHERE
  -- Subject filtering
  (
    (replace(section, ' ', '') IS NULL OR replace(section, ' ', '') LIKE '%Fore%')
    -- Exclude articles with any excludable subjects
    AND NOT EXISTS (
      SELECT 1
      FROM staging.excludable_subjects e
      JOIN (
        SELECT trim(value) as subject
        FROM unnest(string_split(a.subject, ';')) AS t(value)
      ) s ON e.subject_name = s.subject
    )
    -- Include only articles with at least one relevant subject
    AND (
      a.subject IS NULL
      OR EXISTS (
        SELECT 1
        FROM staging.relevant_subjects r
        JOIN (
          SELECT trim(value) as subject
          FROM unnest(string_split(a.subject, ';')) AS t(value)
        ) s ON r.subject_name = s.subject
      )
    )
  )
  -- Location filtering - only include if location is NULL or contains at least one non-domestic location
  AND (
    a.location IS NULL
    OR EXISTS (
      SELECT 1
      FROM UNNEST(regexp_split_to_array(a.location, ';')) AS ss(value)
      LEFT JOIN staging.domestic_locations dl
        ON TRIM(ss.value) = dl.location_name
      WHERE dl.location_name IS NULL
    )
  )
```

### Filtering Configuration

You can customize the filtering by editing the lists in `config/category_filtering.yaml`:

1. **excluded_categories**: Categories to exclude (e.g., "Fashion", "Sports")
2. **excludable_subjects**: Subjects to specifically exclude (e.g., "Air fares", "Theater") 
3. **relevant_subjects**: Subjects to specifically include (e.g., "War", "Military personnel")
4. **domestic_locations**: US locations to exclude from international analysis

These filtering tables help narrow the focus to articles that are most likely to contain relevant MIC events, saving processing time during response generation and improving accuracy of the final dataset.

#### 3. Generate Responses

```bash
python src/data/response_generator.py
```

This script analyzes articles using Gemini 2.5 Pro to identify MIC events. If the output file already exists, the script will skip processing to avoid duplicate work.

#### 4. Create Training Dataset

```bash
python src/data/dataset_maker.py
```

This script creates a training dataset from the LLM responses, generating:

- A primary training dataset file in ShareGPT format
- A sample file of the dataset for inspection

## Configuration

The project uses YAML configuration files in the `config` directory:

- `default.yaml`: Default configuration for database paths and loading settings
- `category_filtering.yaml`: Define categories and subjects for filtering articles

You can also override settings via command-line arguments.

## Project Structure

```
mic-project/
├── config/                 # Configuration files
├── data/                   # Data directories
│   ├── processed/          # Processed data files
│   └── raw/                # Raw downloaded articles
├── src/                    # Source code
│   └── data/               # Data processing scripts
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── run_pipeline.py         # Main pipeline script
```
