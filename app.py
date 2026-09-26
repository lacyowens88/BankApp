"""
Northbridge Bank - Credit Risk Portfolio Query Engine
Streamlit application version of the Credit Risk Query Engine notebook.

Preserves the original pipeline:
  1. Intent classification (verified template vs. generated SQL)
  2. Query construction (load template or generate fresh SQL)
  3. Validation gate (read-only check, schema check, EXPLAIN dry-run,
     LLM relevance check, template integrity check)
  4. One retry on the generated track if validation fails
  5. Escalation to a human analyst if validation still fails
  6. Execution (read-only) against the SQLite database
  7. Narrative response generation
"""

import json
import os
import re
import sqlite3
import warnings
from typing import Optional

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="\U0001F4CA",
    layout="wide",
)

DB_PATH = os.environ.get("CREDIT_RISK_DB_PATH", "credit_risk_portfolio.db")

# --------------------------------------------------------------------------
# Database schema (provided to the LLM in prompts) - unchanged from notebook
# --------------------------------------------------------------------------
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# --------------------------------------------------------------------------
# Verified Query Template Library - SQL unchanged from notebook
# (sql_7's stray leading quote in the original notebook has been corrected)
# --------------------------------------------------------------------------
sql_1 = """SELECT sm.sector_name,
     ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
     ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                    THEN lm.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_mn
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mn DESC"""

sql_2 = """SELECT lm.loan_category,
     ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
     COUNT(*) AS loan_count
FROM loan_master lm
GROUP BY lm.loan_category
ORDER BY total_outstanding_mn DESC"""

sql_3 = """SELECT ifrs9_stage,
     COUNT(*) AS loan_count,
     ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
     ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage"""

sql_4 = """SELECT sm.sector_name,
     ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_coverage_ratio DESC"""

sql_5 = """SELECT lm.borrower_name,
     sm.sector_name,
     ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_mn,
     lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY outstanding_mn DESC
LIMIT 10"""

sql_6 = """SELECT group_name,
     COUNT(*) AS loan_count,
     ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5"""

sql_7 = """SELECT lm.loan_account_number,
     lm.borrower_name,
     sm.sector_name,
     ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_mn,
     lm.days_past_due,
     lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC"""

sql_8 = """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END,
    CASE
        WHEN days_past_due = 0 THEN 0
        WHEN days_past_due BETWEEN 1 AND 30 THEN 1
        WHEN days_past_due BETWEEN 31 AND 60 THEN 2
        WHEN days_past_due BETWEEN 61 AND 90 THEN 3
        ELSE 4
    END
ORDER BY
    CASE
        WHEN days_past_due = 0 THEN 0
        WHEN days_past_due BETWEEN 1 AND 30 THEN 1
        WHEN days_past_due BETWEEN 31 AND 60 THEN 2
        WHEN days_past_due BETWEEN 61 AND 90 THEN 3
        ELSE 4
    END"""

sql_9 = """SELECT borrower_id,
     previous_rating,
     internal_rating AS current_internal_rating,
     pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC"""

sql_10 = """SELECT reporting_date,
     ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date"""

VERIFIED_QUERY_LIBRARY = {
    "VQ1": {
        "description": "Sector-wise total outstanding and NPA amount breakdown across all sectors",
        "sql": sql_1,
    },
    "VQ2": {
        "description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)",
        "sql": sql_2,
    },
    "VQ3": {
        "description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter",
        "sql": sql_3,
    },
    "VQ4": {
        "description": "Average provision coverage ratio by sector for the latest reporting quarter",
        "sql": sql_4,
    },
    "VQ5": {
        "description": "Top 10 largest loan exposures by outstanding amount at the borrower level",
        "sql": sql_5,
    },
    "VQ6": {
        "description": "Top 5 largest exposures aggregated at the business group level",
        "sql": sql_6,
    },
    "VQ7": {
        "description": "All overdue loan accounts with their days past due and asset classification",
        "sql": sql_7,
    },
    "VQ8": {
        "description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio",
        "sql": sql_8,
    },
    "VQ9": {
        "description": "Borrowers whose internal rating was downgraded in the latest rating cycle",
        "sql": sql_9,
    },
    "VQ10": {
        "description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time",
        "sql": sql_10,
    },
}

# --------------------------------------------------------------------------
# LLM setup
# --------------------------------------------------------------------------
def get_credentials():
    """Resolve OpenAI credentials from Streamlit secrets, env vars, or config.json."""
    api_key = None
    api_base = None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE")

    if not api_key and os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, api_base


@st.cache_resource(show_spinner=False)
def get_llms(api_key: str, api_base: Optional[str]):
    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm, evaluator_llm


@st.cache_resource(show_spinner=False)
def get_db_connection(db_path: str):
    # check_same_thread=False: Streamlit reruns/sessions can execute on different
    # threads than the one that created this cached connection. Safe here because
    # the connection is opened read-only (mode=ro) and only ever used for SELECTs.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return conn


# --------------------------------------------------------------------------
# Pipeline tools (ported from the notebook)
# --------------------------------------------------------------------------
def classify_intent(user_question, query_library, llm):
    """Classifies the user question and decides which route to take."""
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a bank's analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail  should NOT match a template that returns an aggregate count.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    """Generates a candidate SQL query for a novel question using the database schema."""
    generation_prompt = f"""
You are an expert in SQLite. Given the user's question and the database schema below, write a SINGLE SQLite SQL query to answer the question.

Here are some important rules:
- The query must be read-only, using only SELECT or WITH statements.
- Do NOT use any DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE) statements.
- Ensure the query is syntactically correct for SQLite.
- If the question involves 'NPA' (Non-Performing Assets), remember that NPA is defined as `asset_classification IN ('Substandard', 'Doubtful', 'Loss')`.
- If the question involves 'latest reporting quarter' or 'latest rating cycle', use the latest available dates: '2025-09-30' for both.
- Return ONLY the SQL query, without any additional text, explanations, or markdown fences (```sql).

User Question:
{user_question}

Database Schema:
{schema_context}

SQL Query:
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    """Validates a candidate SQL query through five checks before execution."""
    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None,
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result
    if ";" in candidate_sql.rstrip(";").rstrip():
        result["failed_check"] = "read_only_shape"
        result["details"] = "Multiple statements are not allowed"
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as", "case",
        "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
        and not tok.isdigit() and tok not in ("s", "l", "p", "r", "e6")
    ]
    if unknown:
        result["failed_check"] = "schema_conformance"
        result["details"] = f"Unrecognized identifier(s) not found in schema: {sorted(set(unknown))}"
        return result

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage - judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
### ROLE
You are a senior credit-risk analyst reviewing a SQL query before it is executed against the bank's portfolio database.

### CONTEXT
{track_context}

### USER QUESTION
{user_question}

### CANDIDATE SQL
{candidate_sql}

### INSTRUCTIONS
Judge whether the candidate SQL correctly answers the user question - including using the right
tables, metrics, joins, aggregations, and (where applicable) the NPA definition
(asset_classification IN ('Substandard', 'Doubtful', 'Loss')) and the latest-date logic
('2025-09-30').

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
        if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
            result["failed_check"] = "llm_relevance"
            result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    """Regenerates SQL after a validation failure, feeding the error back to the LLM."""
    retry_prompt = f"""
You are an expert in SQLite. The SQL query below was written to answer the user's question but
failed a validation check. Correct the query so it passes validation while still answering the
original question.

Rules:
- The query must be read-only, using only SELECT or WITH statements.
- Do NOT use any DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE) statements.
- Only reference tables and columns that exist in the database schema below.
- Ensure the query is syntactically correct for SQLite.
- Return ONLY the corrected SQL query, without any additional text, explanations, or markdown fences.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    """Executes a gate-passed SQL query and returns the result as a DataFrame."""
    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    """Generates a focused natural language response from the query result."""
    response_prompt = f"""
### ROLE
You are a credit-risk analytics assistant writing a concise, business-facing answer for a bank
user. You are given the user's question and the full result set returned by the query engine.

### INSTRUCTIONS
- Answer only what the user asked, using exact figures from the data below.
- If the data contains more rows/columns than the question needs, focus the narrative on the
  relevant subset, but do not invent numbers that are not in the data.
- Keep the response to 2-4 sentences of plain business language, no SQL, no code.
- Do not add caveats about being an AI model.

### USER QUESTION
{user_question}

### QUERY RESULT (route={route}, query_id={query_id})
{dataframe.to_string()}
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, log_fn=None):
    """Runs the complete query engine pipeline for a single user question."""
    def log(msg):
        if log_fn:
            log_fn(msg)

    result_log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    result_log["route"] = classification["route"]
    result_log["query_id"] = classification.get("query_id")
    result_log["match_reason"] = classification.get("match_reason")
    log(f"**[1] Intent Classification:** route=`{result_log['route']}`, query_id=`{result_log['query_id']}`  \nReason: {result_log['match_reason']}")

    # Step 2: Query construction
    if result_log["route"] == "verified" and result_log["query_id"] in query_library:
        candidate_sql = query_library[result_log["query_id"]]["sql"]
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    result_log["candidate_sql"] = candidate_sql
    log(f"**[2] Query Construction:** {'loaded from library' if result_log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, result_log["query_id"])
    result_log["gate_result"] = gate
    log(f"**[3] Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
    if not gate["passed"]:
        log(f"Failed check: `{gate.get('failed_check')}` — {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate["passed"] and result_log["route"] == "generated":
        log(f"Retrying after failure: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context, llm)
        result_log["candidate_sql"] = candidate_sql
        result_log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        result_log["gate_result"] = gate
        log(f"**Retry Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
        if not gate["passed"]:
            log(f"Retry failed check: `{gate.get('failed_check')}` — {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        result_log["escalated"] = True
        result_log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        result_log["confidence"] = "ESCALATED"
        log(f"**[!] Escalated to human analyst:** {gate['details']}")
        return {"log": result_log, "dataframe": None, **result_log}

    # Step 6: Execute
    result_log["executed_sql"] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result["dataframe"]
    result_log["row_count"] = len(df)
    log(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result["warnings"]:
        log(f"Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, result_log["route"], llm, result_log["query_id"])
    result_log["narrative"] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    result_log["confidence"] = gate.get("relevance_confidence")
    log(f"**[6] Response Generation:** confidence=`{result_log['confidence']}`")

    return {"log": result_log, "dataframe": df, **result_log}


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.title("\U0001F4CA Northbridge Bank - Credit Risk Portfolio Query Engine")
    st.caption(
        "Ask routine commercial-lending portfolio questions in plain English. "
        "Answers are backed by pre-approved SQL templates or validated, freshly generated "
        "read-only SQL, with full SQL, data, and confidence shown for auditability."
    )

    # ---- Sidebar: configuration ----
    with st.sidebar:
        st.header("Configuration")
        default_key, default_base = get_credentials()
        api_key = st.text_input("OpenAI API Key", value=default_key or "", type="password")
        api_base = st.text_input("OpenAI API Base URL (optional)", value=default_base or "")
        db_path = st.text_input("Database file path", value=DB_PATH)
        verbose = st.checkbox("Show pipeline trace", value=True)

        st.divider()
        st.subheader("Verified Query Library")
        for qid, entry in VERIFIED_QUERY_LIBRARY.items():
            with st.expander(qid):
                st.write(entry["description"])

        st.divider()
        test_file = st.file_uploader("Optional: upload test_queries.csv", type="csv")

    if not api_key:
        st.warning("Enter an OpenAI API key in the sidebar to run the query engine.")
        st.stop()

    if not os.path.exists(db_path):
        st.error(
            f"Database file not found at `{db_path}`. Place `credit_risk_portfolio.db` in the "
            "app's working directory, or set the correct path in the sidebar."
        )
        st.stop()

    llm, evaluator_llm = get_llms(api_key, api_base or None)
    conn = get_db_connection(db_path)

    # ---- Main: question input ----
    st.subheader("Ask a question")
    example_questions = [
        "Show me the sector-wise outstanding and NPA breakdown",
        "Show me the DPD Aging Profile",
        "What is the average interest rate for each sector?",
        "Which borrowers were downgraded in the latest rating cycle?",
        "Show me the ECL trend across reporting quarters",
    ]
    chosen_example = st.selectbox("Or pick an example question", ["(type my own)"] + example_questions)
    default_text = "" if chosen_example == "(type my own)" else chosen_example
    user_question = st.text_area("Your question", value=default_text, height=80)

    run_clicked = st.button("Run Query", type="primary")

    if run_clicked and user_question.strip():
        trace_container = st.container()
        trace_lines = []

        def log_fn(msg):
            trace_lines.append(msg)

        with st.spinner("Running the query engine pipeline..."):
            result = run_pipeline(
                user_question.strip(),
                conn,
                VERIFIED_QUERY_LIBRARY,
                DATABASE_SCHEMA,
                llm,
                evaluator_llm,
                log_fn=log_fn,
            )

        if verbose:
            with trace_container.expander("Pipeline trace", expanded=False):
                for line in trace_lines:
                    st.markdown(line)

        if result["escalated"]:
            st.error("This question was escalated to a human analyst.")
            st.write(result["narrative"])
            with st.expander("Escalation details"):
                st.write(result["gate_result"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Route", result["route"])
            col2.metric("Query ID", result["query_id"] or "—")
            conf = result["confidence"]
            col3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))

            st.subheader("Answer")
            st.write(result["narrative"])

            st.subheader("Data")
            st.dataframe(result["dataframe"], use_container_width=True)

            st.subheader("SQL Used")
            st.code(result["executed_sql"], language="sql")

            if result["retry_used"]:
                st.info("Note: the initial SQL failed validation and was automatically retried once.")

    elif run_clicked:
        st.warning("Please enter a question first.")

    # ---- Optional: batch evaluation against ground truth ----
    if test_file is not None:
        st.divider()
        st.subheader("Batch Evaluation Against Ground Truth")
        ground_truth = pd.read_csv(test_file)
        st.dataframe(ground_truth, use_container_width=True)

        if st.button("Run all test cases"):
            eval_rows = []
            progress = st.progress(0.0)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = run_pipeline(
                    gt["User Query"], conn, VERIFIED_QUERY_LIBRARY, DATABASE_SCHEMA, llm, evaluator_llm
                )
                eval_rows.append(
                    {
                        "Test Case": gt["Test Case"],
                        "Expected Route": gt["Expected Route"],
                        "Actual Route": tr["route"],
                        "Route Match": tr["route"] == gt["Expected Route"],
                        "Expected Query ID": gt.get("Expected Query ID"),
                        "Actual Query ID": tr["query_id"],
                        "Query ID Match": (
                            pd.isna(gt.get("Expected Query ID")) and pd.isna(tr["query_id"])
                        )
                        or tr["query_id"] == gt.get("Expected Query ID"),
                        "Confidence": tr["confidence"],
                        "Rows Returned": tr["row_count"],
                    }
                )
                progress.progress((i + 1) / len(ground_truth))

            evaluation_df = pd.DataFrame(eval_rows)
            path_accuracy = evaluation_df["Route Match"].mean() * 100
            verified_mask = evaluation_df["Expected Route"].astype(str).str.strip().str.lower() == "verified"
            query_accuracy = (
                evaluation_df.loc[verified_mask, "Query ID Match"].mean() * 100
                if verified_mask.any()
                else float("nan")
            )
            numeric_conf = pd.to_numeric(evaluation_df["Confidence"], errors="coerce")
            average_confidence = numeric_conf.mean()

            st.dataframe(evaluation_df, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence", f"{average_confidence:.2f}")


if __name__ == "__main__":
    main()
