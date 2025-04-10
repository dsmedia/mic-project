import os
import json
import duckdb
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
from tqdm import tqdm
import logging
import sys
import math
from datetime import datetime

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# --- Constants ---
DB_PATH = 'data/processed/mic_analysis.duckdb'
OUTPUT_FILENAME = "data/processed/training_model_results.jsonl" # Updated filename
ERROR_FILENAME = "data/processed/training_model_errors.jsonl"   # Updated filename
PROMPT_SAVE_DIR = "data/interim/prompts"
GEMINI_MODEL_NAME = "models/gemini-2.5-pro-exp-03-25" # Consider using the latest stable model if available
DELAY_BETWEEN_BATCH_CALLS = 10 # Seconds
NUM_ARTICLES_PER_BATCH = 100 # Adjust as needed based on API limits and reliability

# --- Define your list of article IDs to process here ---
ARTICLE_IDS = [ # The list of IDs from your query
59395,40966,53256,47117,45071,43024,36886,40983,43033,49183,40994,45090,36,55334,61481,49194,57391,38960,57394,59444,61492,43065,49210,38972,61500,45118,61501,41029,55365,61509,61518,53327,53328,59473,51282,61522,45142,49239,6233,51289,57436,43112,53353,32874,49258,47214,47219,59508,61557,51318,53367,4216,59511,61564,43134,55422,43139,47236,53381,47249,59541,53398,2200,51353,43162,39070,55455,47265,59554,61605,51366,55464,43177,45227,43188,47285,47286,57525,59572,37049,55482,47296,51393,61632,61638,51401,37070,51415,47325,45278,51423,47328,61663,61665,49383,59624,61671,59633,47346,55540,57591,49401,2302,37119,53508,39173,51465,41236,53530,41243,2334,59681,39203,41252,47396,37158,41254,41258,2349,37166,57645,43314,61749,45371,61756,49469,49470,55616,61762,41288,61769,61770,51532,49491,51539,53593,41310,57696,57698,61794,59749,59752,53609,8554,45420,39278,45422,41329,61809,37235,49524,51581,41342,43391,49535,61821,61827,61830,51591,49552,55696,6553,45466,59803,41372,45468,2466,53669,39338,61866,37294,37295,51630,41395,39350,37304,61882,39356,47550,43465,49609,45519,45520,61905,45522,47571,2516,43476,37349,37350,47590,51692,35310,45550,45554,47606,55798,39416,39417,49657,37374,29185,53761,57859,37381,47621,57862,49672,57863,43532,526,55822,41490,6676,53780,49686,53787,41500,47645,59935,49698,57890,39460,55846,55848,8751,55857,45619,55860,41529,47678,41537,55875,43591,57929,53834,10827,49744,35410,41557,59995,41564,6764,37485,37489,41588,53876,53879,47737,39546,41597,43646,37512,47753,651,45719,41624,47773,47774,51876,47781,41640,58033,49843,41652,43702,695,49847,51897,47803,37567,47810,58052,49861,58056,43722,45775,41687,37593,2782,37598,47840,56033,4839,45799,49899,58091,39661,47856,58098,60147,39670,49910,60151,54010,37628,56062,51970,43779,43781,49931,51980,54027,39696,2833,37650,39699,41750,41753,47898,8993,43810,47906,43814,6951,56103,43820,47916,58157,58161,2866,2877,58175,15169,45889,56131,56132,58181,49990,58183,840,54090,50008,52058,60251,58204,45917,50016,39784,58231,41856,31617,41861,48005,58245,58249,48013,50061,54158,50074,37795,37797,48038,41895,50093,7086,48047,58285,37810,39858,39859,48051,7094,54196,54199,46014,48070,3015,43976,52171,54223,50128,46043,3042,41954,37861,46053,44008,48104,44010,44011,37869,58356,46069,50171,46080,48130,46083,52228,52230,1032,50184,58378,41998,54286,58386,58387,46103,39965,37918,44063,44064,52253,37925,58405,39977,54320,39986,48178,54322,48182,58423,60474,48191,56383,44097,60479,60480,60483,52293,37958,44102,54343,48201,56393,60488,60491,60498,52307,37977,40026,56416,40033,50280,60536,46201,54394,40064,46208,50309,56457,58506,44171,58507,60554,50320,46225,54422,40088,46232,40090,44192,40103,42152,52405,38070,46262,56501,46267,54459,60608,25796,54468,46280,52424,44234,50379,42188,54472,44241,52436,46294,42199,50390,50392,52440,54490,56534,56539,58583,60631,40163,46308,54499,38126,58608,50419,46329,3323,48383,42242,46341,48389,46346,56591,54544,52498,42259,42263,9500,58654,42271,56609,40226,40228,48421,48424,38190,42287,52531,38197,36153,36154,38204,50492,36158,52540,50496,36161,58686,36167,48455,38217,40265,48457,54608,44369,36186,58718,7519,50527,36201,40297,36203,36204,42346,50537,36209,56695,54648,36217,58744,54652,38270,36223,60798,42369,1410,46471,50568,50569,46475,40333,58766,40335,52627,48533,40343,48537,36253,46493,48544,38312,40363,60846,38319,54706,52659,42420,40373,48564,38327,38331,46523,52667,50626,50639,54739,54747,56795,1501,1505,36324,42474,36333,28143,48624,7667,50675,46582,44535,60924,48639,42497,42511,56848,52758,52759,36383,56868,42540,40494,58927,54836,36410,42555,56892,50749,44607,56897,52807,44618,44619,61004,56912,50772,36439,44631,46680,46682,46685,58975,52837,48745,26219,40557,48752,46707,52852,54904,52857,46714,48762,42621,42625,48771,40585,44681,46735,36498,52883,56987,46748,54942,46751,61087,38561,59044,44718,52911,59054,44721,48818,36531,54970,46781,48829,42689,42691,48836,50883,48842,42707,44759,50904,1754,12001,38626,48865,50914,52962,38632,46824,1780,55033,55037,50942,57085,36608,3843,55047,42762,48906,59146,55055,38674,48914,50965,40730,48922,50970,44830,55072,44834,50978,61223,40744,57131,44849,50994,1847,38711,44857,38714,55095,59191,57150,57152,53065,53068,5965,51025,59218,53081,36698,5979,55130,61274,14175,28515,48996,38757,42854,51045,57191,53097,57194,34669,46959,46963,46964,53109,12151,49015,59255,46974,49023,57214,46982,40840,53129,57225,44940,44943,42896,42899,44948,53141,49050,47003,53148,40865,51107,47015,49065,44973,53168,49075,47032,47033,44986,55237,1994,51148,36814,45013,59352,59353,59357,61405,47073,53219,53221,55269,49127,57319,59372,61420,45039,49141,40951,43000,49149,27562,140530,202547,295287,54119,235157,195144,167226,323277,223937,176392,181135,205569,159845,91524,165303,94888,176649,63439,119572,228179,227411,222656,68448,259772,13965,319716,53581,185965,207485,31274,141170,62695,215302,189971,312600,23269,279452,42711,79956,86512,110813,203941,84240,306466,95297,148012,231471,109452,48384,148654,243693,254761,245157,214652,242720,155845,321332,211570,108165,74884,100359,56323,330335,309737,274415,331774,114083,112723,93689,128398,123246,314905,163882,252289,245172,160538,121735,62803,78972,176361,7583,279446,158290,319955,221302,229901,55562,33475,176006,164826,120405,241931,110419,208711,210077,201855,98786,58969,30449,290341,115864,18817,164237,110291,226093,187862,127622,324788,226930,23986,321442,33837,156705,254466,326347,133696,77192,245057,22591,211665,321533,316877,265126,89875,210136,7006,189932,298060,255929,230674,112976,169249,213375,110087,253553,325514,222030,29438,134845,24833,85263,197639,60263,131044,313322,165756,4212,325526,105982,277890,258645,212150,150811,181306,206635,165860,97752,11517,81757,257198,86350,160356,176594,267021,205897,158366,240718,290690,228214,163336,68447,265634,186238,188414,248162,140198,175293,490,304799,179527,208657,218736,211228,162536,5302,178082,248458,228863,320428,149429,329974,260084,136931,313610,329380,283020,189571,146524,51925,45963,35171,8406,160043,218216,163278,223309,78136,126632,274917,11599,187088,32313,26809,158937,116352,76793,191315,217024,299354,216717,91982,181227,21736,183249,16071,31472,162894,23435,239569,82244,177295,136123,176877,309627,275712,178280,158155,10611,117743,200734,239424,288688,316766,238933,82106,24784,186558,45141,66031,99764,126151,202006,150475,121630,206518,164878,222869,176089,208193,222006,323838,306929,88286,60795,206999,150452,222998,135204,199307,189262,302408,71861,327712,114447,141085,70066,237560,152608,245270,126212,259010,313385,125736,208973,324163,21776,231792,174598,133876,261464,170369,55023,169840,91699,177415,151915,177394,226143,326053,159252,159955,269635,241408,233939,230093,188838,249789,149093,285517,284329,64842,281451,38959,13496,178487,239311,277063,142945,254655,10337,235895,118982,131176,242815,226316,301334,174031,90523,21064,329672,87853,165744,178423,288937,154832,315670,230807,256317,228613,291000,155477,241713,186108,87230,95458,22232,309140,251479,196194,193374,170118,242552,177993,255133,13211,166756,158524,239589,171354,230344,239197,161366,162431,19089,201721,45261,183825,230127,324500,236198,93519,310827,173857,331474,242734,252555,53387,141760,282326,128441,285772,127796,203776,196438,174064,237067,2008,129633,172213,247265,36170,214787,230201,73915,50126,298365,5079,176887,330969,218495,204330,245547,58615,230752,337825,260370,96619,123379,255960,148036,233237,229596,38159,103621,163376,232777,316217,75006,2691,275926,142170,289595,14530,162876,79327,90500,99714,134332,56374,139727,211943,90513,180718,254500,20301,227762,56067,314371,115155,58100,99107,185080,4579,233033,151754,274678,133556,110116,180194,253128,45934,216948,31629,320584,157463,224856,243729,181947,160277,259642,215565,26713,167819,111978,292516,168401,243316,250724,122625,276477,242466,130872,156636,306474,250642,300440,242467,229785,204938,233889,63604,161323,165819,323956,90021,255826,274562,15615,161473
]

# --- Ensure Prompt Save Directory Exists ---
try:
    os.makedirs(PROMPT_SAVE_DIR, exist_ok=True)
    logger.info(f"Prompt save directory set to: {PROMPT_SAVE_DIR}")
except OSError as e:
    logger.error(f"Could not create prompt save directory {PROMPT_SAVE_DIR}: {e}", exc_info=True)
    sys.exit(f"Error creating prompt directory: {e}")

# --- Gemini Client Initialization ---
def initialize_gemini_client():
    """Initializes and returns the Gemini client using genai.Client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables or .env file.")
        raise ValueError("GEMINI_API_KEY not found.")
    try:
        # Use genai.configure() for API key and then create client if needed,
        # or directly use the client if that's the preferred pattern.
        # genai.configure(api_key=api_key)
        # client = genai.GenerativeModel(GEMINI_MODEL_NAME) # Example if using GenerativeModel
        client = genai.Client(api_key=api_key) # Using genai.Client as per original script
        logger.info(f"Successfully initialized genai.Client.")
        return client
    except Exception as e:
        logger.error(f"Error initializing genai.Client: {e}", exc_info=True)
        raise

# --- JSON Schema Definitions (UPDATED FOR NEW DATE STRUCTURE & REMOVED involved) ---
single_mic_event_schema = types.Schema(
    type=types.Type.OBJECT,
    description="Details of a single Militarized Interstate Confrontation (MIC) event involving fatalities between recognized states, OR an explanation if no relevant events were found in the article. Properties MUST appear in the order specified by propertyOrdering.",
    properties={
        'article_id': types.Schema(type=types.Type.INTEGER, description="The unique identifier of the source article from which this event/explanation was extracted."),
        'is_relevant': types.Schema(type=types.Type.BOOLEAN, description="Must be True for a relevant MIC event object. Must be False for an explanation object."),
        'start_year': types.Schema(type=types.Type.INTEGER, nullable=True, description="The year the event started (YYYY). Use integer -9 if unknown. Must be null for an explanation object."),
        'start_month': types.Schema(type=types.Type.INTEGER, nullable=True, description="The month the event started (1-12). Use integer -9 if unknown or if year is unknown. Must be null for an explanation object."),
        'start_day': types.Schema(type=types.Type.INTEGER, nullable=True, description="The day the event started (1-31). Use integer -9 if unknown or if month/year is unknown. Must be null for an explanation object."),
        'end_year': types.Schema(type=types.Type.INTEGER, nullable=True, description="The year the event ended (YYYY). Use integer -9 if unknown. Must be null for an explanation object."),
        'end_month': types.Schema(type=types.Type.INTEGER, nullable=True, description="The month the event ended (1-12). Use integer -9 if unknown or if year is unknown. Must be null for an explanation object."),
        'end_day': types.Schema(type=types.Type.INTEGER, nullable=True, description="The day the event ended (1-31). Use integer -9 if unknown or if month/year is unknown. Must be null for an explanation object."),
        'fatalities_min': types.Schema(type=types.Type.INTEGER, nullable=True, description="Minimum estimated number of *military personnel* killed in this specific event. Use integer 0 if unknown/vague but fatalities confirmed. Must be null for an explanation object."),
        'fatalities_max': types.Schema(type=types.Type.INTEGER, nullable=True, description="Maximum estimated number of *military personnel* killed in this specific event. Use same as min if precise, integer 0 if unknown/vague but fatalities confirmed. Must be >= fatalities_min. Must be null for an explanation object."),
        # 'countries_involved' REMOVED
        'countries_suffering_losses': types.Schema(type=types.Type.ARRAY, description="List of standard, full names of the states whose military personnel were killed in this specific confrontation. Use names exactly from the eligible list. Must be an empty array [] for an explanation object.", items=types.Schema(type=types.Type.STRING)),
        'countries_causing_losses': types.Schema(type=types.Type.ARRAY, description="List of standard, full names of the states whose military forces killed personnel from other states in this specific confrontation. Use names exactly from the eligible list. Must be an empty array [] for an explanation object.", items=types.Schema(type=types.Type.STRING)),
        'explanation': types.Schema(type=types.Type.STRING, description="Brief explanation justifying the relevance and data extraction for *this specific event* (citing text snippets if possible), OR explaining why no relevant events were found for the article. Note ambiguities/assumptions (e.g., date inference, fatality estimation).")
    },
    # Update required fields
    required=[
        "article_id", "is_relevant",
        "start_year", "start_month", "start_day",
        "end_year", "end_month", "end_day",
        "fatalities_min", "fatalities_max",
        # "countries_involved", # REMOVED
        "countries_suffering_losses", "countries_causing_losses",
        "explanation"
    ],
    # Update property ordering
    property_ordering=[
        "article_id", "is_relevant",
        "start_year", "start_month", "start_day",
        "end_year", "end_month", "end_day",
        "fatalities_min", "fatalities_max",
        # "countries_involved", # REMOVED
        "countries_suffering_losses", "countries_causing_losses",
        "explanation"
    ]
)
# multi_event_schema and batch_analysis_schema definitions remain unchanged as they refer to single_mic_event_schema
multi_event_schema = types.Schema(
    type=types.Type.ARRAY,
    description="An array containing all distinct MIC events found in the article (each conforming to single_mic_event_schema). If no relevant events are found, return an array containing a single object with null/empty fields except for 'explanation' and 'article_id', detailing why no events were identified.",
    items=single_mic_event_schema
)
batch_analysis_schema = types.Schema(
    type=types.Type.ARRAY,
    description=(
        "Top-level array containing one result element per input article, in sequential order. "
        "Each element within this array corresponds to a single article's analysis and MUST conform "
        "to the multi_event_schema (i.e., each element is itself an array containing either distinct "
        "MIC event objects or a single explanation object for that specific article)."
    ),
    items=multi_event_schema
)

# --- Safety Settings ---
safety_settings = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

# --- Generation Configuration ---
generation_config_obj = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=batch_analysis_schema,
    safety_settings=safety_settings,
    temperature=0.2 # Low temperature for more deterministic extraction
)

# --- Helper function to write errors ---
def append_error_log(error_info):
    """Appends a single error entry to the error log file."""
    try:
        def default_serializer(obj):
            if isinstance(obj, (datetime, types.SafetyRating, types.Content)):
                 return str(obj)
            # Add specific handling for BlockReason if needed
            if isinstance(obj, types.BlockReason):
                 return str(obj) # Or obj.name if you prefer the enum name
            try:
                # Attempt standard JSON encoding first
                return json.JSONEncoder().default(obj)
            except TypeError:
                 # Fallback for non-serializable objects
                 return f"<Object of type {type(obj).__name__} not serializable>"

        with open(ERROR_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_info, default=default_serializer, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Error writing to error log file {ERROR_FILENAME}: {e}")
    except Exception as te:
         logger.error(f"Error serializing error info to JSON: {te}. Info snippet: {str(error_info)[:500]}")

# --- Normalize article data ---
def normalize_article_data(article_data):
    """Normalize article data to ensure consistent structure."""
    normalized_data = article_data.copy()

    # Ensure full_text is present and valid
    if normalized_data.get('full_text') is None and normalized_data.get('document_content') is not None:
        normalized_data['full_text'] = normalized_data['document_content']

    # Ensure ID is an integer
    if 'id' in normalized_data and not isinstance(normalized_data['id'], int):
        try:
            normalized_data['id'] = int(normalized_data['id'])
        except (ValueError, TypeError):
            logger.warning(f"Could not convert article ID {normalized_data.get('id')} to int. Keeping original.")
            pass # Keep as is if conversion fails

    return normalized_data

# --- Create Batch Prompt function (UPDATED WITH MILITARY FORCES CLASSIFICATION GUIDELINES) ---
def create_batch_prompt(batch_article_data, batch_index):
    """
    Creates the detailed prompt Content object for the Gemini model for a batch of articles
    and saves the prompt text to a file. Incorporates military forces classification guidelines,
    new date structure, and enhanced guidance.
    """
    if not batch_article_data:
        return None

    # --- Fixed Instructions (Enhanced Version with Military Forces Classification Guidelines) ---
    prompt_header = f"""Analyze the following **batch of {len(batch_article_data)} news articles** to identify **all distinct** Militarized Interstate Confrontation (MIC) events involving fatalities for **each article independently**, according to the strict definition below.

CRITICALLY IMPORTANT: Your response MUST be structured as follows:
- Return a single, valid JSON **array** with EXACTLY {len(batch_article_data)} elements (no more, no less).
- Each element MUST be an array (even if it contains only one object).
- The position of each element MUST match the position of the corresponding article (first element -> first article, etc.).
- DO NOT create additional elements or objects outside this structure.

Respond **ONLY** with a single, valid JSON **array**. This array **must** contain **exactly {len(batch_article_data)} elements** for the {len(batch_article_data)} articles provided, in the **same order** they appear (Index 0 result corresponds to Index 0 article, Index 1 result to Index 1 article, etc.). Each element within this main array **must** itself be a JSON array. This inner array will contain **one or more** JSON objects if relevant MIC events are found (one object per distinct event), OR it will contain **exactly one** JSON object explaining why no relevant events were found for that specific article.

**Crucially: Every JSON object you generate (whether describing a relevant event OR explaining why no event was found) MUST include the `article_id` field, populated with the ID of the specific article that object pertains to. Furthermore, the properties within each generated JSON object MUST appear in the exact order specified by the schema's `propertyOrdering`:** (`article_id`, `is_relevant`, `start_year`, `start_month`, `start_day`, `end_year`, `end_month`, `end_day`, `fatalities_min`, `fatalities_max`, `countries_suffering_losses`, `countries_causing_losses`, `explanation`).

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
*   **Distinct Events:** Pay close attention to the narrative. If an article describes multiple separate clashes or incidents over time (even if close together), generate a distinct JSON object for *each* incident that meets the MIC definition.
*   **Scan Entire Text:** Thoroughly examine the *entire* article text. Do **not** ignore an event that meets the MIC definition simply because it is mentioned as historical background, context, a comparison, or is not the main focus of the article. If it's described and fits the criteria, extract it.

--- START OF ARTICLE BATCH ({len(batch_article_data)} Articles) ---
"""
    # --- Append Article Data ---
    articles_section = ""
    max_chars_per_article = 18000 # Apply truncation per article if needed
    for index, article_data in enumerate(batch_article_data):
        # Normalize the article data first
        article_data = normalize_article_data(article_data)

        article_id = article_data.get('id', 'N/A')
        pub_date_raw = article_data.get('publication_date', 'N/A')
        pub_date = 'N/A'
        # Try to parse and format the date
        if pub_date_raw and pub_date_raw != 'N/A':
            try:
                date_obj = datetime.strptime(pub_date_raw, '%b %d, %Y')
                day_str = date_obj.strftime('%#d') if os.name == 'nt' else date_obj.strftime('%-d')
                pub_date = date_obj.strftime(f'%A, %B {day_str}, %Y')
                logger.debug(f"Successfully parsed date '{pub_date_raw}' with format '%b %d, %Y'.")
            except ValueError:
                try:
                    date_obj = datetime.strptime(pub_date_raw, '%Y-%m-%d')
                    day_str = date_obj.strftime('%#d') if os.name == 'nt' else date_obj.strftime('%-d')
                    pub_date = date_obj.strftime(f'%A, %B {day_str}, %Y')
                    logger.debug(f"Successfully parsed date '{pub_date_raw}' with format '%Y-%m-%d'.")
                except ValueError as e:
                    pub_date = pub_date_raw
                    logger.warning(f"Could not parse publication date: '{pub_date_raw}'. Keeping original. Error: {str(e)}")
            except Exception as date_e:
                 pub_date = pub_date_raw
                 logger.warning(f"Error formatting date '{pub_date_raw}'. Keeping original. Error: {str(date_e)}")

        full_text = article_data.get('full_text', '') or ''
        location = article_data.get('location', 'N/A')
        subject = article_data.get('subject', 'N/A')
        people = article_data.get('people', 'N/A')

        if len(full_text) > max_chars_per_article:
            full_text = full_text[:max_chars_per_article] + " [TEXT TRUNCATED]"

        location_str = f"Location Context: {location}" if location and location != 'N/A' else "Location Mentioned: Not Available"
        subject_str = f"Subject Context: {subject}" if subject and subject != 'N/A' else "Subject Keywords: Not Available"
        people_str = f"People Context: {people}" if people and people != 'N/A' else "People Mentioned: Not Available"

        articles_section += f"""
--- ARTICLE START (Index: {index}, ID: {article_id}) ---
Input Article Context:
*   Article ID: {article_id}
*   Publication Date: {pub_date}
*   {location_str}
*   {subject_str}
*   {people_str}

Full Article Text:
--- START TEXT {article_id} ---
{full_text}
--- END TEXT {article_id} ---
--- ARTICLE END (Index: {index}, ID: {article_id}) ---
"""

    # --- Final Instructions (UPDATED FOR NEW DATE STRUCTURE, REMOVED involved, & BACKGROUND EVENT GUIDANCE) ---
    prompt_footer = f"""
--- END OF ARTICLE BATCH ---

**Your Task:**
Based *strictly* on the definition, the eligible country list, and the text of **each article independently**:

1.  **Analyze Each Article:** For each article provided in the batch:
    a.  Identify **all distinct** incidents within *that article's text* that meet the **full MIC definition**. Remember that a single article might describe multiple separate events, including those mentioned in passing or as background.
    b.  Determine if one or more such relevant MIC events were found.

2.  **If Relevant Event(s) Found:**
    *   For **each** distinct relevant MIC event identified, create **one** JSON object adhering **strictly** to the required property order: (`article_id`, `is_relevant`, `start_year`, `start_month`, `start_day`, `end_year`, `end_month`, `end_day`, `fatalities_min`, `fatalities_max`, `countries_suffering_losses`, `countries_causing_losses`, `explanation`).
    *   Populate fields as follows:
        *   `article_id`: (Integer) The source article ID.
        *   `is_relevant`: (Boolean) **Must** be `true`.
        *   **Date Fields (Integers):**
            *   `start_year`, `start_month`, `start_day`: The start date components (YYYY, 1-12, 1-31).
            *   `end_year`, `end_month`, `end_day`: The end date components (YYYY, 1-12, 1-31).
            *   For single-day events, set `end_year/month/day` identical to `start_year/month/day`.
            *   Use the integer value `-9` for any component (day, month, or year) that is unknown or cannot be determined from the text. If the month is unknown, set both month and day to `-9`. If the year is unknown, set year, month, and day to `-9`.
            *   Infer relative dates (e.g., 'yesterday', 'last June', 'on Sunday') based on the article's `Publication Date`. If inference is impossible, use `-9`.
        *   `fatalities_min`: (Integer) Minimum military killed. Use `0` ONLY when fatalities are confirmed but the number is completely unknown/vague (e.g., 'soldiers died').
        *   `fatalities_max`: (Integer) Maximum military killed. Use `0` ONLY when fatalities are confirmed but the number is completely unknown/vague. Must be `>= fatalities_min`. If a range or number is mentioned (even 'several' or 'dozens'), estimate min/max.
        *   `countries_suffering_losses`: (Array of Strings) Eligible states whose personnel were killed (use exact names from list).
        *   `countries_causing_losses`: (Array of Strings) Eligible states whose forces caused fatalities (use exact names from list).
        *   `explanation`: (String) Justification citing text. Crucially, note any assumptions made (e.g., inferring pilot death, estimating fatalities from vague terms, date inference). If the article presents conflicting accounts of the *same* event, create separate event objects for each distinct account and note this in the explanation.

    *   Combine all such event objects (there might be **multiple** if several distinct events are described in the article) for this article into a single JSON array.

3.  **If NO Relevant Events Found:**
    *   Create a JSON array containing **exactly one** "explanation object" adhering **strictly** to the required property order and values:
        ```json
        [
          {{
            "article_id": [Correct Article ID Here],
            "is_relevant": false,
            "start_year": null,
            "start_month": null,
            "start_day": null,
            "end_year": null,
            "end_month": null,
            "end_day": null,
            "fatalities_min": null,
            "fatalities_max": null,
            "countries_suffering_losses": [],
            "countries_causing_losses": [],
            "explanation": "No incidents described in the text for this specific article met the strict MIC definition... [Optional brief reason, e.g., involved non-state actors, no fatalities mentioned, internal conflict only]"
          }}
        ]
        ```
    *   **Important:** Adhere strictly to `false`, `null` for all date and fatality fields, and `[]` for `countries_suffering_losses` and `countries_causing_losses` fields, and maintain the property order.

4.  **Assemble Final Output:** Combine the resulting JSON arrays (one per article, where each inner array contains either one-or-more event objects OR one explanation object) into a single **outer** JSON array, maintaining the original article order.

**Key Requirements Summary (Checklist):**
*   Output **must** be a single, valid JSON array with EXACTLY {len(batch_article_data)} elements (one per article).
*   Outer array **must** have one element per input article, in the original order.
*   Each element of the outer array **must** be an inner JSON array.
*   The inner array for an article contains **one or more** event objects if relevant events are found, or **exactly one** explanation object if none are found.
*   Every object inside the inner arrays **must** include the correct `article_id` (Integer).
*   Properties within each object **must** follow the specified `propertyOrdering` (`article_id`, `is_relevant`, `start_year`, `start_month`, `start_day`, `end_year`, `end_month`, `end_day`, `fatalities_min`, `fatalities_max`, `countries_suffering_losses`, `countries_causing_losses`, `explanation`).
*   Relevant event objects **must** have `is_relevant: true`. Date components use integer `-9` for unknown. Fatalities use integer `0` for unknown number.
*   Explanation objects **must** have `is_relevant: false`, `null` for all date and fatality fields, and empty arrays `[]` for `countries_suffering_losses` and `countries_causing_losses`.
*   Adhere strictly to the MIC definition and eligible countries list (use exact names).
*   Ensure each *separate* clash or incident gets its own JSON object if relevant.
*   Scan the *entire* text for relevant events, including background mentions.
"""

    full_prompt_text = prompt_header + articles_section + prompt_footer

    # --- Save the prompt text to a file ---
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            first_id = batch_article_data[0].get('id', 'unknown')
        except IndexError:
            first_id = 'unknown'
        try:
            last_id = batch_article_data[-1].get('id', 'unknown') if batch_article_data else 'unknown'
        except IndexError:
            last_id = 'unknown'

        prompt_filename = os.path.join(PROMPT_SAVE_DIR, f"prompt_batch_{batch_index}_ids_{first_id}-{last_id}_{timestamp}.txt")
        with open(prompt_filename, "w", encoding="utf-8") as f_prompt:
            f_prompt.write(full_prompt_text)
        logger.info(f"Saved prompt for batch {batch_index} to {prompt_filename}")
    except Exception as e_save:
        logger.error(f"Failed to save prompt for batch {batch_index}: {e_save}", exc_info=True)

    # --- Create Content object for API ---
    content_for_api = types.Content(
        parts=[types.Part(text=full_prompt_text)],
        role="user"
    )
    return content_for_api

# --- Validate Batch Response ---
def validate_batch_response(parsed_response, expected_length, batch_article_data):
    """Validates the top-level batch structure and performs necessary corrections."""
    if not isinstance(parsed_response, list):
        logger.error(f"Expected JSON array response, but received type: {type(parsed_response)}.")
        return None, False

    # If we received more results than expected, trim to expected length
    if len(parsed_response) > expected_length:
        logger.warning(f"Received {len(parsed_response)} results but expected {expected_length}. Trimming extra results.")
        parsed_response = parsed_response[:expected_length]

    # If we received fewer results than expected, pad with explanation objects
    elif len(parsed_response) < expected_length:
        logger.warning(f"Received {len(parsed_response)} results but expected {expected_length}. Padding with explanations.")
        for i in range(len(parsed_response), expected_length):
            # Ensure batch_article_data has enough elements before accessing
            if i < len(batch_article_data):
                 article_id = batch_article_data[i].get('id', -1) # Use -1 or another indicator if ID missing
            else:
                 article_id = -1 # Indicate missing source article data
                 logger.error(f"Padding error: Index {i} out of bounds for batch_article_data (length {len(batch_article_data)})")

            # Create the explanation object with the updated structure (no countries_involved)
            parsed_response.append([{
                "article_id": article_id,
                "is_relevant": False,
                "start_year": None, "start_month": None, "start_day": None,
                "end_year": None, "end_month": None, "end_day": None,
                "fatalities_min": None, "fatalities_max": None,
                # "countries_involved": [], # REMOVED
                "countries_suffering_losses": [],
                "countries_causing_losses": [],
                "explanation": f"No analysis was provided for article ID {article_id} in the API response (padded)."
            }])

    return parsed_response, True
# --- Validation and Normalization Function (UPDATED FOR NEW DATE STRUCTURE) ---
# --- Validation and Normalization Function (UPDATED FOR NEW DATE STRUCTURE & REMOVED involved) ---
def validate_and_normalize_result(analysis_result, expected_article_id, batch_index, result_index):
    """
    Validates a single event/explanation object from the API response (with new date structure,
    removed 'involved') and normalizes explanation objects. Returns the corrected object
    and a boolean indicating if validation passed without unrecoverable errors.
    """
    validation_passed = True # Assume pass initially, set to False on unrecoverable errors
    validation_issues = []

    # Ensure analysis_result is a mutable dictionary
    if isinstance(analysis_result, types.Part):
         logger.warning(f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}: Received raw Part, attempting to parse text.")
         try:
             analysis_result_dict = json.loads(analysis_result.text)
             corrected_result = analysis_result_dict.copy()
         except (json.JSONDecodeError, AttributeError) as e:
             logger.error(f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}: Could not parse text from Part: {e}")
             return {"error": "Received unparsable Part object", "article_id": expected_article_id}, False
    elif isinstance(analysis_result, dict):
        corrected_result = analysis_result.copy()
    else:
        logger.error(f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}: Received unexpected result type: {type(analysis_result)}")
        return {"error": f"Unexpected result type {type(analysis_result)}", "article_id": expected_article_id}, False

    log_prefix = f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}:"

    # --- Check Article ID ---
    actual_article_id = corrected_result.get('article_id')
    if actual_article_id is None:
        logger.warning(f"{log_prefix} Missing 'article_id' field. Fixing.")
        validation_issues.append("Missing 'article_id'")
        corrected_result['article_id'] = expected_article_id
    elif not isinstance(actual_article_id, int):
        logger.warning(f"{log_prefix} Invalid type for 'article_id': {type(actual_article_id)}. Attempting fix.")
        validation_issues.append(f"Invalid type for 'article_id' ({type(actual_article_id)})")
        try:
            corrected_result['article_id'] = int(actual_article_id)
            logger.info(f"{log_prefix} Corrected 'article_id' type to int.")
        except (ValueError, TypeError):
             logger.error(f"{log_prefix} Could not convert 'article_id' {actual_article_id} to int. Using expected ID.")
             corrected_result['article_id'] = expected_article_id
             validation_passed = False # Fail validation if conversion fails
    elif actual_article_id != expected_article_id:
        logger.warning(f"{log_prefix} Mismatched 'article_id'. Expected {expected_article_id}, got {actual_article_id}.")
        validation_issues.append(f"Mismatched 'article_id' (expected {expected_article_id}, got {actual_article_id})")
        # Don't necessarily fail validation for mismatch, but log it.

    # --- Check is_relevant ---
    is_relevant = corrected_result.get('is_relevant')
    if is_relevant is None:
        logger.error(f"{log_prefix} Missing 'is_relevant' field. Cannot proceed.")
        validation_issues.append("Missing 'is_relevant'")
        corrected_result['record_type'] = 'unknown_relevance'
        validation_passed = False
    elif not isinstance(is_relevant, bool):
        logger.warning(f"{log_prefix} Invalid type for 'is_relevant': {type(is_relevant)}. Attempting fix.")
        validation_issues.append(f"Invalid type for 'is_relevant' ({type(is_relevant)})")
        if isinstance(is_relevant, str):
            if is_relevant.lower() == 'true':
                corrected_result['is_relevant'] = True
                is_relevant = True
                logger.info(f"{log_prefix} Corrected 'is_relevant' from string 'true' to boolean.")
            elif is_relevant.lower() == 'false':
                 corrected_result['is_relevant'] = False
                 is_relevant = False
                 logger.info(f"{log_prefix} Corrected 'is_relevant' from string 'false' to boolean.")
            else:
                 logger.error(f"{log_prefix} Unrecognized string value for 'is_relevant': {is_relevant}")
                 corrected_result['record_type'] = 'unknown_relevance'
                 validation_passed = False
        else:
             logger.error(f"{log_prefix} Non-bool/non-string type for 'is_relevant': {type(is_relevant)}")
             corrected_result['record_type'] = 'unknown_relevance'
             validation_passed = False

    # Proceed only if relevance could be determined
    if validation_passed:
        # --- Explanation Object Normalization ---
        if is_relevant is False:
            corrected_result['record_type'] = 'explanation'
            # Fields to set to null for explanations
            fields_to_null = [
                'start_year', 'start_month', 'start_day',
                'end_year', 'end_month', 'end_day',
                'fatalities_min', 'fatalities_max'
            ]
            # Update this list
            fields_to_empty_array = ['countries_suffering_losses', 'countries_causing_losses']

            for field in fields_to_null:
                if field not in corrected_result or corrected_result[field] is not None:
                    if field in corrected_result: # Log only if it existed and wasn't null
                         logger.warning(f"{log_prefix} Normalizing explanation: Setting '{field}' to null (was: {corrected_result[field]}, type: {type(corrected_result[field])}).")
                         validation_issues.append(f"Normalized '{field}' to null for explanation")
                    corrected_result[field] = None # Ensure field exists and is null

            for field in fields_to_empty_array:
                current_val = corrected_result.get(field)
                if field not in corrected_result or not isinstance(current_val, list) or current_val:
                    if field in corrected_result: # Log only if it existed and wasn't an empty list
                         logger.warning(f"{log_prefix} Normalizing explanation: Setting '{field}' to [] (was: {current_val}, type: {type(current_val)}).")
                         validation_issues.append(f"Normalized '{field}' to [] for explanation")
                    corrected_result[field] = [] # Ensure field exists and is []

        # --- Event Object Validation ---
        elif is_relevant is True:
            corrected_result['record_type'] = 'event'

            # Date validation (New Structure)
            date_fields = ['start_year', 'start_month', 'start_day', 'end_year', 'end_month', 'end_day']
            for field in date_fields:
                val = corrected_result.get(field)
                if val is None:
                    logger.warning(f"{log_prefix} Date field '{field}' is null for relevant event. Setting -9.")
                    validation_issues.append(f"Set date field '{field}' to -9 (was null)")
                    corrected_result[field] = -9
                elif not isinstance(val, int):
                    logger.warning(f"{log_prefix} Invalid type for date field '{field}': {type(val)}. Attempting fix to -9.")
                    validation_issues.append(f"Invalid type for '{field}' ({type(val)}), set to -9")
                    try:
                        # Attempt conversion only if it looks like a number string
                        if isinstance(val, str) and val.strip().lstrip('-').isdigit():
                            corrected_result[field] = int(val)
                            logger.info(f"{log_prefix} Corrected date field '{field}' type to int from string.")
                        else:
                            raise ValueError("Not a simple integer string")
                    except (ValueError, TypeError):
                        corrected_result[field] = -9 # Default to -9 if conversion fails or type is wrong
                # Basic range checks (can be expanded)
                elif 'month' in field and not (1 <= val <= 12 or val == -9):
                    logger.warning(f"{log_prefix} Invalid value for month field '{field}': {val}. Setting -9.")
                    validation_issues.append(f"Invalid value for '{field}' ({val}), set to -9")
                    corrected_result[field] = -9
                elif 'day' in field and not (1 <= val <= 31 or val == -9): # Basic check, not calendar aware
                    logger.warning(f"{log_prefix} Invalid value for day field '{field}': {val}. Setting -9.")
                    validation_issues.append(f"Invalid value for '{field}' ({val}), set to -9")
                    corrected_result[field] = -9
                elif 'year' in field and not (1900 <= val <= datetime.now().year + 5 or val == -9): # Example year range check
                     logger.warning(f"{log_prefix} Suspicious value for year field '{field}': {val}. Keeping value but logging.")
                     validation_issues.append(f"Suspicious value for '{field}' ({val})")
                     # Keep the value but flag it

            # Optional: Check if end date is before start date (only if all components are known)
            s_yr, s_mo, s_dy = corrected_result.get('start_year'), corrected_result.get('start_month'), corrected_result.get('start_day')
            e_yr, e_mo, e_dy = corrected_result.get('end_year'), corrected_result.get('end_month'), corrected_result.get('end_day')
            # Check only if ALL components are valid integers and not -9
            if all(isinstance(d, int) and d != -9 for d in [s_yr, s_mo, s_dy, e_yr, e_mo, e_dy]):
                try:
                    # Validate day based on month/year before creating datetime
                    datetime(s_yr, s_mo, s_dy) # Will raise ValueError if invalid (e.g., Feb 30)
                    datetime(e_yr, e_mo, e_dy) # Will raise ValueError if invalid
                    start_dt_val = s_yr * 10000 + s_mo * 100 + s_dy
                    end_dt_val = e_yr * 10000 + e_mo * 100 + e_dy
                    if end_dt_val < start_dt_val:
                        logger.warning(f"{log_prefix} End date {e_yr}-{e_mo}-{e_dy} is before start date {s_yr}-{s_mo}-{s_dy}. Check explanation.")
                        validation_issues.append("End date is before start date")
                except ValueError: # Handle invalid date combinations like Feb 30
                    logger.warning(f"{log_prefix} Invalid date combination detected (e.g., Feb 30).")
                    validation_issues.append("Invalid date combination")


            # Fatalities validation (Updated for clarity on 0)
            f_min = corrected_result.get('fatalities_min')
            f_max = corrected_result.get('fatalities_max')
            if f_min is None:
                logger.warning(f"{log_prefix} 'fatalities_min' is null for relevant event. Setting 0 (unknown).")
                validation_issues.append("Set 'fatalities_min' to 0 (was null)")
                corrected_result['fatalities_min'] = 0
                f_min = 0
            elif not isinstance(f_min, int) or f_min < 0:
                logger.warning(f"{log_prefix} Invalid 'fatalities_min': {f_min}. Setting 0 (unknown).")
                validation_issues.append(f"Invalid 'fatalities_min' ({f_min}), set to 0")
                corrected_result['fatalities_min'] = 0
                f_min = 0

            if f_max is None:
                logger.warning(f"{log_prefix} 'fatalities_max' is null for relevant event. Setting to fatalities_min ({f_min}).")
                validation_issues.append(f"Set 'fatalities_max' to {f_min} (was null)")
                corrected_result['fatalities_max'] = f_min
                f_max = f_min
            elif not isinstance(f_max, int) or f_max < 0:
                logger.warning(f"{log_prefix} Invalid 'fatalities_max': {f_max}. Setting to fatalities_min ({f_min}).")
                validation_issues.append(f"Invalid 'fatalities_max' ({f_max}), set to fatalities_min")
                corrected_result['fatalities_max'] = f_min
                f_max = f_min

            # Ensure max >= min after potential corrections
            if isinstance(f_min, int) and isinstance(f_max, int) and f_min > f_max:
                logger.warning(f"{log_prefix} Corrected 'fatalities_min' ({f_min}) > 'fatalities_max' ({f_max}). Setting max=min.")
                validation_issues.append(f"Corrected 'fatalities_max' (was < min)")
                corrected_result['fatalities_max'] = f_min

            # Countries validation (Updated: only check suffering/causing)
            for field in ['countries_suffering_losses', 'countries_causing_losses']: # Removed 'countries_involved'
                val = corrected_result.get(field)
                if not isinstance(val, list):
                     logger.warning(f"{log_prefix} Invalid type for '{field}': {type(val)}. Setting [].")
                     validation_issues.append(f"Invalid type for '{field}' ({type(val)}), set to []")
                     corrected_result[field] = []
                     # Don't fail validation just for wrong type if we fix it
                elif not all(isinstance(c, str) for c in val):
                     logger.warning(f"{log_prefix} Non-string element found in '{field}': {val}. Attempting correction.")
                     try:
                         corrected_list = [str(c) for c in val]
                         corrected_result[field] = corrected_list
                         validation_issues.append(f"Corrected non-string elements in '{field}'")
                         logger.info(f"{log_prefix} Corrected non-string elements in '{field}' to strings.")
                     except Exception:
                         logger.error(f"{log_prefix} Could not correct non-string elements in '{field}'. Setting [].")
                         corrected_result[field] = []
                         validation_issues.append(f"Could not correct non-string elements in '{field}', set to []")
                         validation_passed = False # Fail if correction fails

        # --- Explanation Text Validation (Required for both types) ---
        explanation_val = corrected_result.get('explanation')
        if not explanation_val or not isinstance(explanation_val, str) or not explanation_val.strip():
            logger.warning(f"{log_prefix} Missing, invalid, or empty 'explanation'. Fixing.")
            validation_issues.append("Missing/invalid/empty 'explanation'")
            corrected_result['explanation'] = "[Explanation missing or invalid]"

    # --- Determine Final Status ---
    if not validation_passed:
        final_validation_status = "failed"
    elif validation_issues:
        final_validation_status = "passed_with_warnings"
    else:
        final_validation_status = "passed"

    corrected_result['validation_status'] = final_validation_status
    corrected_result['validation_issues'] = validation_issues

    return corrected_result, validation_passed

# --- Extract Text From API Response ---
def extract_json_from_response(raw_text):
    """Extracts JSON content from the API response text with robust handling."""
    if not raw_text:
        logger.warning("Received empty raw text for JSON extraction.")
        return None

    cleaned_text = raw_text.strip()
    json_extracted = False

    # Handle potential markdown code blocks
    md_start_json = cleaned_text.find("```json")
    md_start_plain = cleaned_text.find("```")
    md_end = cleaned_text.rfind("```")

    if md_start_json != -1 and md_end > md_start_json:
        # Found ```json ... ```
        cleaned_text = cleaned_text[md_start_json + 7 : md_end].strip()
        json_extracted = True
        logger.debug(f"Extracted content from ```json block.")
    elif md_start_plain != -1 and md_end > md_start_plain:
        # Found ``` ... ```
        potential_json = cleaned_text[md_start_plain + 3 : md_end].strip()
        # Check if the content inside looks like JSON
        if potential_json.startswith('[') or potential_json.startswith('{'):
            cleaned_text = potential_json
            json_extracted = True
            logger.debug(f"Extracted content from ``` block.")
        else:
            logger.warning(f"Found ``` block, but content didn't start with [ or {{. Proceeding with raw text search.")
            # Keep cleaned_text as is for further processing below

    # If no markdown block was successfully processed, look for the first JSON structure
    if not json_extracted:
        logger.debug(f"No markdown block found/processed. Searching for first [ or {{.")
        brace_index = cleaned_text.find('{')
        bracket_index = cleaned_text.find('[')

        # Find the first occurrence of either bracket or brace
        json_start_index = -1
        if bracket_index != -1 and (brace_index == -1 or bracket_index < brace_index):
            json_start_index = bracket_index
        elif brace_index != -1: # No need to check bracket_index == -1 here
            json_start_index = brace_index

        if json_start_index != -1:
            # Check if there's non-whitespace before the start
            preceding_text = cleaned_text[:json_start_index].strip()
            if preceding_text:
                logger.warning(f"Found potential JSON start at index {json_start_index}. Discarding preceding text: '{preceding_text[:100]}...'")
            cleaned_text = cleaned_text[json_start_index:]
            # Attempt to find the corresponding closing bracket/brace might be too complex/unreliable here.
            # Rely on json.loads to handle truncation/errors later.
        else:
            logger.error(f"Could not find start of JSON ('[' or '{{') in the response.")
            return None # Cannot proceed if no JSON start is found

    return cleaned_text

# --- Main Processing Function ---
def process_articles_in_batches():
    """Fetches articles in batches, processes them with Gemini, validates, normalizes,
       appends results/explanations, and adds delay."""
    if not ARTICLE_IDS:
        logger.error("ARTICLE_IDS list is empty. Cannot proceed.")
        return

    logger.info("Starting batch article processing...")
    logger.info(f"Total Article IDs to process: {len(ARTICLE_IDS)}")
    logger.info(f"Batch size: {NUM_ARTICLES_PER_BATCH}")
    logger.info(f"Results/Explanations will be appended to: {OUTPUT_FILENAME}")
    logger.info(f"Errors will be appended to: {ERROR_FILENAME}")
    logger.info(f"Prompts will be saved to: {PROMPT_SAVE_DIR}")
    logger.info(f"Delay between API calls: {DELAY_BETWEEN_BATCH_CALLS} seconds")

    total_articles_processed = 0
    total_batches = math.ceil(len(ARTICLE_IDS) / NUM_ARTICLES_PER_BATCH)
    processed_count_in_run = 0
    error_count = 0
    relevant_event_count = 0
    explanation_record_count = 0
    articles_actually_fetched = 0
    batch_index = 0

    con = None
    try:
        # Initialize Gemini client inside the try block
        gemini_client = initialize_gemini_client()
        
        logger.info(f"Connecting to DuckDB database: {DB_PATH}")
        con = duckdb.connect(database=DB_PATH, read_only=True)
        logger.info("Database connection established.")

        with tqdm(total=len(ARTICLE_IDS), desc="Analyzing Articles") as pbar:
            for i in range(0, len(ARTICLE_IDS), NUM_ARTICLES_PER_BATCH):
                batch_start_time = time.time()
                batch_index = (i // NUM_ARTICLES_PER_BATCH) + 1
                article_ids_in_batch = ARTICLE_IDS[i:i + NUM_ARTICLES_PER_BATCH]
                logger.info(f"--- Processing Batch {batch_index}/{total_batches} (IDs: {article_ids_in_batch[:3]}...{article_ids_in_batch[-1:] if len(article_ids_in_batch)>3 else ''}) ---")

                if not article_ids_in_batch:
                    logger.warning(f"Batch {batch_index}: Skipping empty batch.")
                    continue

                # --- Fetch Data for Batch ---
                ids_tuple = tuple(article_ids_in_batch)
                # Ensure correct SQL syntax for single vs multiple IDs
                if len(ids_tuple) == 1:
                    # Need comma for single-element tuple syntax in SQL IN clause
                    sql_query = f"SELECT id, publication_date, full_text, location, subject, people FROM raw.articles WHERE id IN ({ids_tuple[0]});"
                elif len(ids_tuple) > 1:
                    sql_query = f"SELECT id, publication_date, full_text, location, subject, people FROM raw.articles WHERE id IN {ids_tuple};"
                else: # Should not happen if article_ids_in_batch is not empty, but handle defensively
                    logger.warning(f"Batch {batch_index}: No IDs in tuple for SQL query.")
                    continue


                batch_article_data = []
                processed_ids_in_this_batch = []
                try:
                    logger.debug(f"Batch {batch_index}: Executing query: {sql_query}")
                    cursor = con.execute(sql_query)
                    column_names = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    articles_found_in_batch = len(rows)
                    articles_actually_fetched += articles_found_in_batch
                    logger.info(f"Batch {batch_index}: Fetched {articles_found_in_batch} articles from DB for this batch.")

                    fetched_data_map = {row[column_names.index('id')]: dict(zip(column_names, row)) for row in rows}

                    # Pre-filter articles before processing
                    for article_id in article_ids_in_batch:
                        if article_id in fetched_data_map:
                            article_data = fetched_data_map[article_id]
                            # Check for empty or placeholder text
                            article_text = article_data.get('full_text')
                            if not article_text or str(article_text).strip().lower() in ['', 'not available.', 'n/a']:
                                logger.warning(f"Batch {batch_index}, Article ID {article_id}: Skipping due to empty or placeholder full_text.")
                                error_detail = {"batch_index": batch_index, "article_id": article_id, "error": "Empty or placeholder full_text"}
                                append_error_log(error_detail)
                                error_count += 1
                                pbar.update(1) # Update progress for skipped article
                            else:
                                # Normalize the article data
                                article_data = normalize_article_data(article_data)
                                batch_article_data.append(article_data)
                                processed_ids_in_this_batch.append(article_id)
                        else:
                            logger.warning(f"Batch {batch_index}, Article ID {article_id}: Not found in database.")
                            error_detail = {"batch_index": batch_index, "article_id": article_id, "error": "Article ID not found in DB"}
                            append_error_log(error_detail)
                            error_count += 1
                            pbar.update(1) # Update progress for missing article

                except duckdb.Error as db_err_batch:
                    logger.error(f"Batch {batch_index}: DuckDB Error fetching data: {db_err_batch}", exc_info=True)
                    error_detail = {"batch_index": batch_index, "article_ids": article_ids_in_batch, "error": "DuckDB query failed", "details": str(db_err_batch)}
                    append_error_log(error_detail)
                    # Increment error count and progress for all articles intended for this batch
                    skipped_count = len(article_ids_in_batch)
                    error_count += skipped_count
                    pbar.update(skipped_count)
                    time.sleep(1) # Small delay after DB error
                    continue # Skip to next batch

                if not batch_article_data:
                    logger.warning(f"Batch {batch_index}: No processable articles found after filtering/DB check.")
                    # Apply delay only if there are more batches to process
                    if i + NUM_ARTICLES_PER_BATCH < len(ARTICLE_IDS):
                         time.sleep(max(0, DELAY_BETWEEN_BATCH_CALLS - (time.time() - batch_start_time)))
                    continue

                # --- Create and Save Prompt ---
                batch_prompt_content = create_batch_prompt(batch_article_data, batch_index)
                if not batch_prompt_content:
                    logger.error(f"Batch {batch_index}: Failed to generate prompt content.")
                    skipped_count = len(batch_article_data)
                    error_count += skipped_count
                    pbar.update(skipped_count) # Update progress for articles we couldn't generate prompt for
                    continue

                # --- API Call ---
                parsed_batch_response = None
                api_error_occurred = False
                response = None # Initialize response variable

                try:
                    logger.info(f"Batch {batch_index}: Sending request to Gemini for {len(batch_article_data)} articles...")
                    # Use the client object obtained from initialize_gemini_client
                    # Assuming genai.Client has a generate_content method similar to GenerativeModel
                    # Adjust this call based on the actual genai.Client API if different
                    response = gemini_client.models.generate_content(
                        model=GEMINI_MODEL_NAME, # Model might be specified differently for Client
                        contents=[batch_prompt_content],
                        config=generation_config_obj, # Pass the config object directly
                        # safety_settings=safety_settings # Safety settings might be part of config or separate
                    )
                    logger.info(f"Batch {batch_index}: Received response from Gemini.")

                    # --- Initial Response Validation ---
                    result_text = None
                    block_reason = None
                    finish_reason = None
                    safety_feedback = None

                    # Access feedback and candidates (adjust based on actual response structure)
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                         block_reason = response.prompt_feedback.block_reason
                         safety_feedback = response.prompt_feedback.safety_ratings
                         logger.warning(f"Batch {batch_index}: Prompt blocked by API. Reason: {block_reason}. Ratings: {safety_feedback}")
                         error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "Prompt blocked", "reason": str(block_reason), "safety_ratings": str(safety_feedback)}
                         append_error_log(error_detail)
                         api_error_occurred = True

                    elif not hasattr(response, 'candidates') or not response.candidates:
                         logger.warning(f"Batch {batch_index}: No candidates received from API.")
                         feedback_info = str(response.prompt_feedback) if hasattr(response, 'prompt_feedback') and response.prompt_feedback else 'N/A'
                         error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "No candidates in response", "feedback": feedback_info}
                         append_error_log(error_detail)
                         api_error_occurred = True
                    else:
                        # Process the first candidate
                        candidate = response.candidates[0]
                        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
                        safety_feedback = candidate.safety_ratings if hasattr(candidate, 'safety_ratings') else None

                        if finish_reason == types.FinishReason.SAFETY:
                            logger.warning(f"Batch {batch_index}: Candidate blocked explicitly due to safety. Finish Reason: {finish_reason}. Ratings: {safety_feedback}")
                            error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "Candidate safety block", "finish_reason": str(finish_reason), "safety_ratings": str(safety_feedback)}
                            append_error_log(error_detail)
                            api_error_occurred = True
                        elif finish_reason not in [types.FinishReason.STOP, types.FinishReason.MAX_TOKENS, None]: # Allow None finish reason if API sometimes omits it on success
                             logger.warning(f"Batch {batch_index}: Candidate stopped unexpectedly. Finish Reason: {finish_reason}. Safety Ratings: {safety_feedback}")
                             error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "Candidate stopped unexpectedly", "finish_reason": str(finish_reason), "safety_ratings": str(safety_feedback)}
                             append_error_log(error_detail)
                             api_error_occurred = True
                        else:
                            # Extract text safely
                            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                result_text = candidate.content.parts[0].text
                            elif hasattr(response, 'text'): # Fallback if text is directly on response
                                result_text = response.text
                            else:
                                result_text = None # No text found

                            if not result_text and finish_reason == types.FinishReason.STOP:
                                logger.warning(f"Batch {batch_index}: Received empty text response despite STOP finish reason.")
                                error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "Empty text response", "finish_reason": str(finish_reason)}
                                append_error_log(error_detail)
                                api_error_occurred = True
                            elif result_text:
                                logger.debug(f"Batch {batch_index}: Received text length: {len(result_text)}")

                                # Extract JSON from response using the helper function
                                cleaned_text = extract_json_from_response(result_text)

                                if cleaned_text:
                                    try:
                                        raw_parsed_response = json.loads(cleaned_text)
                                        # Validate and fix batch structure
                                        parsed_batch_response, batch_valid = validate_batch_response(
                                            raw_parsed_response,
                                            len(batch_article_data),
                                            batch_article_data
                                        )

                                        if not batch_valid:
                                            logger.error(f"Batch {batch_index}: Invalid batch structure, couldn't repair.")
                                            error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "Invalid batch structure"}
                                            append_error_log(error_detail)
                                            api_error_occurred = True
                                        else:
                                            logger.info(f"Batch {batch_index}: Successfully processed JSON response.")

                                    except json.JSONDecodeError as json_e:
                                        logger.error(f"Batch {batch_index}: Failed to decode JSON response: {json_e}")
                                        logger.error(f"Batch {batch_index}: Text that failed JSON parsing (first 500 chars):\n{cleaned_text[:500]}")
                                        error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "JSONDecodeError", "details": str(json_e), "raw_response_sample": cleaned_text[:500]}
                                        append_error_log(error_detail)
                                        api_error_occurred = True
                                else:
                                    logger.error(f"Batch {batch_index}: Failed to extract JSON from response text.")
                                    error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "JSON extraction failed", "raw_response_sample": result_text[:500] if result_text else "N/A"}
                                    append_error_log(error_detail)
                                    api_error_occurred = True
                            else:
                                logger.warning(f"Batch {batch_index}: No text extracted from response parts.")
                                error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "No text in response parts", "finish_reason": str(finish_reason)}
                                append_error_log(error_detail)
                                api_error_occurred = True

                except Exception as api_e:
                    logger.error(f"Batch {batch_index}: API call or response processing failed: {api_e}", exc_info=True)
                    # Include response details if available
                    response_details = str(response)[:500] if response else "N/A"
                    error_detail = {"batch_index": batch_index, "article_ids": processed_ids_in_this_batch, "error": "API call/processing failed", "exception_type": str(type(api_e)), "details": str(api_e), "response_snippet": response_details}
                    append_error_log(error_detail)
                    api_error_occurred = True

                # Increment error count and progress if API error occurred for the whole batch
                if api_error_occurred:
                    skipped_count = len(batch_article_data)
                    error_count += skipped_count
                    pbar.update(skipped_count)

                # --- Process Parsed Batch Response (only if no API error) ---
                if parsed_batch_response is not None and not api_error_occurred:
                    # --- Iterate through results for each article in the batch ---
                    for article_index, article_result_list in enumerate(parsed_batch_response):
                        # Ensure article_index is within bounds of processed_ids_in_this_batch
                        if article_index >= len(processed_ids_in_this_batch):
                             logger.error(f"Batch {batch_index}: article_index {article_index} out of bounds for processed IDs list (len {len(processed_ids_in_this_batch)}). Skipping result.")
                             error_count += 1
                             continue
                        expected_article_id = processed_ids_in_this_batch[article_index]
                        processed_count_in_run += 1

                        if not isinstance(article_result_list, list):
                            logger.warning(f"Batch {batch_index}, Article ID {expected_article_id} (Index {article_index}): Expected inner list of results, got {type(article_result_list)}. Writing error explanation.")
                            error_detail = {"batch_index": batch_index, "article_id": expected_article_id, "error": "Invalid inner result type", "details": f"Expected list, got {type(article_result_list)}"}
                            append_error_log(error_detail)
                            # Write a default explanation indicating the error
                            default_explanation = {
                                "article_id": expected_article_id, "is_relevant": False,
                                "start_year": None, "start_month": None, "start_day": None,
                                "end_year": None, "end_month": None, "end_day": None,
                                "fatalities_min": None, "fatalities_max": None,
                                "countries_involved": [], "countries_suffering_losses": [], "countries_causing_losses": [],
                                "explanation": f"Error processing article ID {expected_article_id}: API response structure invalid (expected list, got {type(article_result_list)}).",
                                "record_type": "explanation", "validation_status": "failed",
                                "validation_issues": ["Invalid inner result type from API"]
                            }
                            try:
                                with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f_out:
                                    f_out.write(json.dumps(default_explanation, ensure_ascii=False) + "\n")
                                explanation_record_count += 1
                            except IOError as e_write:
                                logger.error(f"Batch {batch_index}, Article ID {expected_article_id}: Error writing error explanation: {e_write}")
                            error_count += 1
                            continue # Skip to next article result

                        if not article_result_list:
                            logger.info(f"Batch {batch_index}, Article ID {expected_article_id}: Model returned an empty list '[]', writing default explanation.")
                            # Create and write a default explanation for empty lists
                            default_explanation = {
                                "article_id": expected_article_id, "is_relevant": False,
                                "start_year": None, "start_month": None, "start_day": None,
                                "end_year": None, "end_month": None, "end_day": None,
                                "fatalities_min": None, "fatalities_max": None,
                                "countries_involved": [], "countries_suffering_losses": [], "countries_causing_losses": [],
                                "explanation": f"No relevant events found in article ID {expected_article_id}. Model returned empty result list.",
                                "record_type": "explanation", "validation_status": "passed",
                                "validation_issues": ["Model returned empty result list"]
                            }
                            try:
                                with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f_out:
                                    f_out.write(json.dumps(default_explanation, ensure_ascii=False) + "\n")
                                explanation_record_count += 1
                            except IOError as e_write:
                                logger.error(f"Batch {batch_index}, Article ID {expected_article_id}: Error writing default explanation: {e_write}")
                                error_count += 1
                            continue # Skip to next article result

                        # --- Iterate through events/explanation for the article ---
                        for result_index, analysis_result in enumerate(article_result_list):
                            try:
                                validated_result, validation_passed = validate_and_normalize_result(
                                    analysis_result, expected_article_id, batch_index, result_index
                                )

                                log_level = logging.INFO
                                validation_status = validated_result.get('validation_status', 'unknown')
                                if validation_status == 'failed':
                                    log_level = logging.ERROR
                                elif validation_status == 'passed_with_warnings':
                                     log_level = logging.WARNING

                                if validation_passed: # Check if validation passed without critical errors
                                    record_type = validated_result.get('record_type', 'unknown')
                                    logger.log(log_level, f"Batch {batch_index}, Article ID {expected_article_id}: Writing '{record_type}' record (validation: {validation_status}). Issues: {validated_result.get('validation_issues')}")

                                    try:
                                        with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f_out:
                                            f_out.write(json.dumps(validated_result, ensure_ascii=False) + "\n")

                                        if record_type == 'event':
                                            relevant_event_count += 1
                                        elif record_type == 'explanation':
                                            explanation_record_count += 1

                                    except IOError as e_write:
                                        logger.error(f"Batch {batch_index}, Article ID {expected_article_id}: Error writing result: {e_write}")
                                        error_detail = {"batch_index": batch_index, "article_id": expected_article_id, "record_type": record_type, "error": "File write error", "details": str(e_write)}
                                        append_error_log(error_detail)
                                        error_count += 1
                                else:
                                    # Log critical validation failure as ERROR
                                    logger.error(f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}: Critical validation failed. Result not written. Issues: {validated_result.get('validation_issues', [])}")
                                    # Error detail already includes raw result from validation function
                                    error_detail = validated_result # Log the whole validation output as error detail
                                    error_detail["error"] = "Critical validation failed" # Ensure error field is set
                                    append_error_log(error_detail)
                                    error_count += 1

                            except Exception as val_e:
                                 logger.error(f"Batch {batch_index}, Article ID {expected_article_id}, Result {result_index}: Error during validation/normalization: {val_e}", exc_info=True)
                                 error_detail = {"batch_index": batch_index, "article_id": expected_article_id, "result_index": result_index, "error": "Validation/Normalization exception", "details": str(val_e)}
                                 append_error_log(error_detail)
                                 error_count += 1

                # --- Update Progress Bar & Delay ---
                # Update progress bar based on articles *attempted* in the batch
                # (those fetched and not skipped before API call)
                pbar.update(len(article_ids_in_batch)) # Update by the original number requested for the batch

                # Apply delay only if there are more batches to process
                if i + NUM_ARTICLES_PER_BATCH < len(ARTICLE_IDS):
                    elapsed_time = time.time() - batch_start_time
                    wait_time = DELAY_BETWEEN_BATCH_CALLS - elapsed_time
                    if wait_time > 0:
                        logger.info(f"Batch {batch_index} finished in {elapsed_time:.2f}s. Waiting {wait_time:.2f}s before next batch...")
                        time.sleep(wait_time)
                    else:
                         logger.info(f"Batch {batch_index} finished in {elapsed_time:.2f}s (-{abs(wait_time):.2f}s over). Proceeding immediately.")

    except duckdb.Error as db_err:
        logger.error(f"DuckDB Error accessing database {DB_PATH}: {db_err}", exc_info=True)
        raise
    except ValueError as val_err:
         logger.error(f"Configuration Error: {val_err}")
         raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during the main processing loop: {e}", exc_info=True)
        raise
    finally:
        if con:
            try:
                con.close()
                logger.info("Database connection closed.")
            except Exception as close_e:
                 logger.error(f"Error closing DuckDB connection: {close_e}")

    # --- Final Summary ---
    logger.info("="*40)
    logger.info("Processing Summary:")
    logger.info(f"Requested Article IDs: {len(ARTICLE_IDS)}")
    logger.info(f"Batch Size: {NUM_ARTICLES_PER_BATCH}")
    logger.info(f"Total Batches Processed: {batch_index if batch_index > 0 else 0}/{total_batches}")
    logger.info(f"Articles Found in DB & Fetched: {articles_actually_fetched}")
    logger.info(f"Articles Processed (analysis written or error logged): {processed_count_in_run + error_count}") # Should approx equal total IDs
    logger.info(f"Total Relevant Events Written ('event'): {relevant_event_count}")
    logger.info(f"Total Explanation Records Written ('explanation'): {explanation_record_count}")
    logger.info(f"Total Errors Logged (incl. skips, API issues, validation fails): {error_count}")
    logger.info(f"Output written to: {OUTPUT_FILENAME}")
    logger.info(f"Errors appended to: {ERROR_FILENAME}")
    logger.info(f"Prompts saved in: {PROMPT_SAVE_DIR}")
    logger.info("Processing complete.")
    logger.info("="*40)

def main():
    """Main entry point for the script, called by the pipeline."""
    if not ARTICLE_IDS:
        logger.critical("ARTICLE_IDS list is empty in the script. Please paste your IDs. Terminating.")
        return
    
    # Check if output file already exists
    if os.path.exists(OUTPUT_FILENAME):
        logger.info(f"Output file {OUTPUT_FILENAME} already exists. Skipping processing.")
        logger.info("If you want to regenerate responses, please rename or delete the existing file.")
        return
    
    # Only if we get here, try to process the articles
    # The actual initialization happens inside process_articles_in_batches
    process_articles_in_batches()
    
# --- Main Execution Guard ---
if __name__ == "__main__":
    main()