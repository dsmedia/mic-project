#!/usr/bin/env python3
"""
Main script to run the entire MIC project pipeline.
This script sequentially executes all the processing steps:
1. Download articles
2. Create/load database
3. Generate article responses
4. Create training dataset
"""

import os
import sys
import logging
import argparse
import importlib.util
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mic_pipeline")

def import_script(script_path):
    """Dynamically import a Python script."""
    logger.debug(f"Importing script: {script_path}")
    try:
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to import script {script_path}: {e}")
        raise

def run_script(script_path, function_name="main"):
    """Run a specific function from a script."""
    logger.info(f"Running script: {script_path}")
    try:
        module = import_script(script_path)
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            func()
            logger.info(f"Completed running: {script_path}")
            return True
        else:
            logger.error(f"Function '{function_name}' not found in {script_path}")
            return False
    except Exception as e:
        logger.error(f"Error running {script_path}: {e}", exc_info=True)
        return False

def parse_args():
    """Parse command line arguments."""
    # This ensures that we parse arguments used by pipeline_create_and_load.py
    # but also maintain our own arguments specific to run_pipeline.py
    
    parser = argparse.ArgumentParser(
        description='Run the MIC project pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    pipeline_group = parser.add_argument_group('Pipeline Steps')
    pipeline_group.add_argument('--skip-download', action='store_true',
                       help='Skip the article download step')
    
    pipeline_group.add_argument('--skip-database', action='store_true',
                       help='Skip the database creation and loading step')
    
    pipeline_group.add_argument('--skip-responses', action='store_true',
                       help='Skip the response generation step')
    
    pipeline_group.add_argument('--skip-dataset', action='store_true',
                       help='Skip the training dataset creation step')
    
    pipeline_group.add_argument('--force-all', action='store_true',
                       help='Force all steps to run even if output files exist')
    
    # Import config arguments from src/data/config.py
    try:
        from src.data.config import add_config_args
        parser = add_config_args(parser)
    except ImportError as e:
        logger.warning(f"Could not import config.add_config_args: {e}")
        logger.warning("Configuration arguments will not be available.")
    
    args = parser.parse_args()
    return args

def run_pipeline_create_and_load(project_root, args):
    """Run the database creation and loading step directly with arguments."""
    try:
        from src.data.pipeline_create_and_load import main as db_main
        
        # Create a new namespace with only the relevant arguments for pipeline_create_and_load
        db_args = argparse.Namespace()
        
        # Copy arguments used by pipeline_create_and_load
        if args.force_all:
            db_args.force = True
        else:
            db_args.force = getattr(args, 'force', False)
            
        # Copy any config-related arguments
        for arg_name in ['config', 'db_path', 'data_dir', 'load_proquest', 'load_nyt', 
                        'load_all', 'proquest_dir', 'proquest_recursive', 'nyt_dir', 
                        'nyt_recursive']:
            if hasattr(args, arg_name):
                setattr(db_args, arg_name, getattr(args, arg_name))
        
        # Set the skip arguments correctly
        db_args.skip_db_creation = args.skip_database
        db_args.skip_loading = getattr(args, 'skip_loading', False)
        
        # Run the function directly with our args
        return db_main(db_args)
    except (ImportError, TypeError) as e:
        logger.error(f"Error using direct function call: {e}")
        logger.error("Falling back to script execution mode")
        return False

def main():
    """Main function to run the entire pipeline."""
    logger.info("Starting MIC pipeline")
    
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    scripts_dir = project_root / "src" / "data"
    
    # Paths to the pipeline scripts
    download_script = scripts_dir / "download_articles.py"
    database_script = scripts_dir / "pipeline_create_and_load.py"
    response_script = scripts_dir / "response_generator.py"
    dataset_script = scripts_dir / "dataset_maker.py"
    
    # Check if scripts exist
    for script in [download_script, database_script, response_script, dataset_script]:
        if not script.exists():
            logger.error(f"Script not found: {script}")
            sys.exit(1)
    
    # Set environment variables and check requirements
    os.environ["PYTHONPATH"] = str(project_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    # Required output paths to check for skipping steps
    db_path = project_root / "data" / "processed" / "mic_analysis.duckdb"
    response_file = project_root / "data" / "processed" / "mic_event_analysis_results_batch_gemini25_newdate.jsonl"
    dataset_file = project_root / "data" / "processed" / "training_model_results.json"
    
    # 1. Download articles
    if not args.skip_download:
        logger.info("=== Step 1: Download Articles ===")
        run_script(download_script)
    else:
        logger.info("Skipping article download (--skip-download)")
    
    # 2. Create/load database
    if not args.skip_database and (args.force_all or not db_path.exists()):
        logger.info("=== Step 2: Create Database and Load Articles ===")
        # Try direct function call first, fall back to script execution
        success = run_pipeline_create_and_load(project_root, args)
        if not success:
            # Fall back to the old script execution method
            run_script(database_script)
    else:
        logger.info(f"Skipping database creation/loading (--skip-database or file exists: {db_path.exists()})")
    
    # 3. Generate responses
    if not args.skip_responses and (args.force_all or not response_file.exists()):
        logger.info("=== Step 3: Generate Article Responses ===")
        # Check if API key is set
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY not found in environment. Response generation may fail.")
        run_script(response_script, "process_articles_in_batches")
    else:
        logger.info(f"Skipping response generation (--skip-responses or file exists: {response_file.exists()})")
    
    # 4. Create training dataset
    if not args.skip_dataset and (args.force_all or not dataset_file.exists()):
        logger.info("=== Step 4: Create Training Dataset ===")
        run_script(dataset_script, "generate_sharegpt_dataset")
    else:
        logger.info(f"Skipping dataset creation (--skip-dataset or file exists: {dataset_file.exists()})")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()