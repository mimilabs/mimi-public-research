# Databricks notebook source
# MAGIC %pip install anthropic

# COMMAND ----------

!pip install tqdm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import json
import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm
from datetime import datetime
import pickle
from pathlib import Path
from pprint import pprint

anthropic_token = dbutils.secrets.get(scope="slackbot", key="anthropic") # replace this with your API key
output_path = "/Volumes/mimi_ws_1/hhsoig/src/claude_outputs"

# COMMAND ----------

class OIGArticleProcessor:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = [{"type": "text",
            "cache_control": {"type": "ephemeral"},
            "text": """You are a healthcare compliance data extraction system. Your role is to analyze healthcare enforcement action articles and extract key information into structured JSON format.

Output Schema:
{
    "article_id": number,                    // Sequential ID of the article
    "case_status": string,              // Settlement, Criminal Charges, Guilty Plea, Sentencing, Indictment, Arrest
    "subject_name": string,                  // Names of individual/organization
    "subject_type": string,                  // "Individual" or "Organization"
    "subject_role": string,                  // e.g., "CEO", "Physician", "Nurse"
    "location_city": string,                 // City
    "location_state": string,                // State
    "settlement_amount": number | null,      // Amount in USD
    "restitution_amount": number | null,     // Amount in USD
    "violation_types": string[],             // Array of violations
    "violation_start_date": string | null,   // ISO date or year
    "violation_end_date": string | null,     // ISO date or year
    "affected_programs": string[],           // e.g., ["Medicare", "Medicaid"]
    "resolution_type": string,               // e.g., "settlement", "guilty plea"
    "resolution_date": string | null,        // Date of resolution/action
    "prison_term": string | null,            // Prison sentence if applicable
    "probation_term": string | null,         // Probation term if applicable
    "victims_affected": number | null,       // Number of victims/patients
    "victim_type": string | null             // e.g., "elderly", "disabled"
}

Extraction Rules:
1. Monetary values: Convert to numbers (remove $ and ,)
2. Dates: Use ISO format (YYYY-MM-DD) where possible, or YYYY if only year available
3. Missing data: Use null, never leave empty
4. Subject types: "Individual" or "Organization" only

Common violation categories include but are not limited to:
- fraud
- kickbacks
- false claims
- documentation fraud
- drug diversion
- patient abuse

Focus on accurate extraction without commentary. If information is ambiguous, use available context or default to null if unclear."""
        }]

    def create_extraction_prompt(self, article):
        user_prompt = f"""Given this healthcare enforcement action article, extract key information into a JSON object following the system-defined schema.

Articles to process:

{article}"""

        return user_prompt

    def process_articles(self, article):
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": self.create_extraction_prompt(article)
                    }
                ]
            )
            # Extract JSON from response
            response_text = message.content[0].text
            # Find JSON block in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
        except Exception as e:
            print(f"Error processing notice: {str(e)}")
            return []
        
    def post_process_article(self, article):
        """Simple post-processing of article data with date parsing"""
        
        # Fields that should be joined if they're arrays
        array_fields = {
            'case_status', 
            'subject_name', 
            'subject_role',
            'subject_type',
            'location_city', 
            'location_state', 
            'resolution_type',
            'prison_term',
            'victim_type'
        }
        
        # Date fields to parse
        date_fields = {
            'violation_start_date', 'violation_end_date', 'resolution_date'
        }
        
        # Process each field
        for field, value in article.items():
            if value is None:
                continue
                
            # Join arrays for specified fields
            if field in array_fields and isinstance(value, list):
                article[field] = '; '.join(str(v) for v in value if v)
                
            # Convert numeric fields
            elif field in {'settlement_amount', 
                           'restitution_amount', 
                           'victims_affected', 
                           'restitution_amount'}:
                if isinstance(value, list):
                    try:
                        article[field] = float(max(value))
                    except ValueError:
                        article[field] = None
                elif isinstance(value, str):
                    try:
                        article[field] = float(value.replace('$', '').replace(',', ''))
                    except ValueError:
                        article[field] = None
                        
            # Parse dates
            elif field in date_fields and value:
                try:
                    parsed_date = parse(str(value)).date()
                    article[field] = parsed_date
                except (ValueError, TypeError):
                    article[field] = None

        return article


# COMMAND ----------

# initialize our Claude connector
processor = OIGArticleProcessor(anthropic_token)

# COMMAND ----------

# Find the articles that are not enrichced yet
pdf = spark.read.table('mimi_ws_1.hhsoig.enforcement_details').toPandas()

# COMMAND ----------

for index, row in tqdm(pdf.iterrows()):
    page_url = row.page_url
    article = f"article_id: {index}\n\n{row.title}\n{row.content}\n"
    result = processor.process_articles(article)
    d_original = row.to_dict()
    d_clean = processor.post_process_article({**result, **d_original})
    pprint(d_clean)
    break

# COMMAND ----------


