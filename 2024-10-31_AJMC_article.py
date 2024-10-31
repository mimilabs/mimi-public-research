# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Medicare Advantage Coverage and Deductible Analysis 2024-2025
# MAGIC
# MAGIC **Note: This research script is made publicly available to promote transparency in healthcare analytics and foster collaborative research. We encourage other researchers to build upon this work to further analyze Medicare Advantage market dynamics and their impact on beneficiaries.**
# MAGIC
# MAGIC Analysis script examining Medicare Advantage regional coverage changes and Part D drug deductible increases between 2024-2025. Focuses on tracking coverage region modifications and analyzing deductible changes, including the $91-$114 average increase per member and specific impacts on previously $0-deductible plans (~9 million members). Based on Mimilabs data.
# MAGIC
# MAGIC ## Research Transparency and Educational Use
# MAGIC - Source code and methodology published for reproducibility
# MAGIC - Encourages peer review and validation of findings
# MAGIC - Supports academic and industry researchers in conducting follow-up studies
# MAGIC - Facilitates understanding of Medicare Advantage market dynamics
# MAGIC - Promotes evidence-based policy discussions
# MAGIC
# MAGIC ## Limitations and Assumptions
# MAGIC
# MAGIC ### Data Exclusions
# MAGIC - Contract IDs and Plan IDs missing from both 2024 and 2025 landscape files are excluded from deductible comparison analysis, despite showing non-zero enrollments in some areas
# MAGIC
# MAGIC ### Geographic Considerations
# MAGIC - MAPD (Medicare Advantage Prescription Drug) data analyzed at county level
# MAGIC - PDP (Prescription Drug Plan) data analyzed at state level
# MAGIC - Data merging maintains respective geographic aggregation levels
# MAGIC
# MAGIC ### Member Transition Assumptions
# MAGIC - MAPD members assumed to transition only to other MAPD plans within same county
# MAGIC - PDP members assumed to transition only to other PDP plans within same state
# MAGIC - Zero member churn assumed in transition calculations
# MAGIC
# MAGIC ### Reference Data
# MAGIC - Enrollment numbers based on October 2024 data
# MAGIC - Plan transitions tracked using latest plan crosswalk file
# MAGIC
# MAGIC Originally published in the American Journal of Managed Care (AJMC). Link: https://www.ajmc.com/view/contributor-vulnerable-seniors-are-at-risk-with-looming-medicare-advantage-cuts-income-based-programs-can-minimize-the-damage

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Loading datasets

# COMMAND ----------

# MAGIC %run /Workspace/utils/basic

# COMMAND ----------

# MAGIC %pip install us

# COMMAND ----------

import us
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# State name format mismatch between files:
# - Landscape files: Full state names (e.g., "California")
# - Enrollment files: State abbreviations (e.g., "CA")
# Solution: Use 'us' Python package to create state name crosswalk
# Example: states_map = {state.name: state.abbr for state in us.states.STATES}
state_dict = {state.abbr: state.name for state in us.states.STATES}
state_dict['DC'] = 'District of Columbia'
fn = "PlanCrosswalk2024_10012024.txt"
volumepath = "/Volumes/mimi_ws_1/partcd/src/plancrosswalk/"
filepath = volumepath + fn
curr_year = 2024
next_year = 2025

# User Defined Function to convert state abbreviations to full state names
@F.udf(returnType=StringType())
def state_abbr_to_name(abbr):
    return state_dict.get(abbr, "Unknown")

# COMMAND ----------

# Column name mappings
COL_MAPS = {
    'enrollment': {'contract_id': 'cid_curr', 'plan_id': 'pid_curr'},
    'crosswalk': {
        'PREVIOUS_CONTRACT_ID': 'cid_curr', 'PREVIOUS_PLAN_ID': 'pid_curr',
        'CURRENT_CONTRACT_ID': 'cid_next', 'CURRENT_PLAN_ID': 'pid_next'
    },
    'landscape': {
        'curr': {'contract_id': 'cid_curr', 'plan_id': 'pid_curr', 'part_d_drug_deductible': 'dd_curr'},
        'next': {'contract_id': 'cid_next', 'plan_id': 'pid_next', 'part_d_drug_deductible': 'dd_next'}
    }
}

# Load data frames
df_enrollment = (get_df_latest(spark.read.table("mimi_ws_1.partcd.cpsc_combined"))
                .withColumn('state', state_abbr_to_name(F.col('state')))
                .withColumnsRenamed(COL_MAPS['enrollment']))

df_crosswalk = (spark.createDataFrame(pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1', dtype=str))
                .withColumnsRenamed(COL_MAPS['crosswalk']))

df_landscape_curr = (spark.read.table("mimi_ws_1.partcd.landscape_plan_premium_report")
                    .where(f"YEAR(mimi_src_file_date) = {curr_year}")
                    .withColumnsRenamed(COL_MAPS['landscape']['curr']))

df_landscape_next = (spark.read.table("mimi_ws_1.partcd.landscape_plan_premium_report")
                    .where(f"YEAR(mimi_src_file_date) = {next_year}")
                    .withColumnsRenamed(COL_MAPS['landscape']['next']))

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Plan Discontinuation Impact Analysis
# MAGIC
# MAGIC Analysis of Medicare Advantage membership affected by plan exits in 2024-2025. Quantifies the number of beneficiaries who will lose access to their current plans due to contract terminations or plan discontinuations. Based on October 2024 enrollment data matched against 2025 landscape files to identify discontinued contracts and affected membership.
# MAGIC
# MAGIC ## Methodology
# MAGIC - Members with automatic plan transitions (identified via plan crosswalk file) are excluded from the "lost plan" count
# MAGIC - These members will be automatically enrolled in their new designated plans
# MAGIC - Analysis assumes zero member churn during transition period
# MAGIC - Resulting estimates are conservative, as they only count members who have no automatic transition path to a new plan
# MAGIC
# MAGIC Note: This method provides a lower-bound estimate of disruption, focusing solely on members who must actively choose a new plan for 2025.

# COMMAND ----------

# Remove duplicate crosswalk entries, keep essential columns
df_crosswalk_dedup = (
    df_crosswalk
    .dropDuplicates(['cid_curr', 'pid_curr', 'cid_next', 'pid_next'])
    .withColumnRenamed('STATUS', 'cw_status')
    .select(['cid_curr', 'pid_curr', 'cid_next', 'pid_next', 'cw_status'])
)

# Calculate enrollment by plan type and crosswalk status
# Key metric: 'Terminated' status shows members needing to actively choose new plans
(
    df_enrollment
    .join(df_crosswalk_dedup, on=['cid_curr', 'pid_curr'], how='left')
    .groupBy('plan_type', 'cw_status')
    .agg(F.sum('enrollment').alias('count'))
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Medicare Part D Drug Deductible Changes 2024-2025
# MAGIC
# MAGIC Analysis of Part D drug deductible changes between 2024 and 2025 plan years. Limited to continuously existing plans present in both years' landscape files. Member transitions only occur as specified in the plan crosswalk file; no additional churn assumed.
# MAGIC
# MAGIC ## Analysis Scope
# MAGIC - Plans: Only those present in both 2024 and 2025 landscape files
# MAGIC - Membership: Based on October 2024 enrollment
# MAGIC - Transitions: Following plan crosswalk specifications only

# COMMAND ----------

# MAPD plans use county-level data
df_curr_county = (
    df_landscape_curr
    .select('state', 'county', 'cid_curr', 'pid_curr', 
            F.col('dd_curr').alias('dd_curr_county'))
)

# PDP plans use state-level data (All Counties)
df_curr_state = (
    df_landscape_curr
    .where("county = '(All Counties)'")
    .select('state', 'cid_curr', 'pid_curr', 
            F.col('dd_curr').alias('dd_curr_state'))
)

# Next year data with same geographic logic as current year
df_next_county = (  # MAPD: county-level
    df_landscape_next
    .select('state', 'county', 'cid_next', 'pid_next', 
            F.col('dd_next').alias('dd_next_county'))
)

df_next_state = (  # PDP: state-level
    df_landscape_next
    .where("county = 'All Counties'")
    .select('state', 'cid_next', 'pid_next',
            F.col('dd_next').alias('dd_next_state'))
)


# Calculate regional bounds for estimating deductibles of terminated plans
df_next_county_bounds = (  # MAPD: county-level bounds
    df_landscape_next
    .groupBy('state', 'county')
    .agg(
        F.max('dd_next').alias('dd_next_county_max'),
        F.min('dd_next').alias('dd_next_county_min')
    )
)

df_next_state_bounds = (  # PDP: state-level bounds
    df_landscape_next
    .groupBy('state')
    .agg(
        F.max('dd_next').alias('dd_next_state_max'),
        F.min('dd_next').alias('dd_next_state_min')
    )
)

# Combine data and calculate deductible changes
df_analysis = (
    df_enrollment
    .join(df_crosswalk_dedup, on=['cid_curr', 'pid_curr'], how='left')
    .join(df_curr_county, on=['state', 'county', 'cid_curr', 'pid_curr'], how='left')  # MAPD
    .join(df_curr_state, on=['state', 'cid_curr', 'pid_curr'], how='left')             # PDP
    .join(df_next_county, on=['state', 'county', 'cid_next', 'pid_next'], how='left')  # MAPD
    .join(df_next_state, on=['state', 'cid_next', 'pid_next'], how='left')             # PDP
    .join(df_next_county_bounds, on=['state', 'county'], how='left')
    .join(df_next_state_bounds, on=['state'], how='left')
    .withColumn('dd_curr', F.coalesce('dd_curr_county', 'dd_curr_state'))
    .withColumn('dd_next', F.coalesce('dd_next_county', 'dd_next_state'))
    # For terminated plans: use regional min/max as bounds for possible deductible changes
    .withColumn('dd_next_max', 
        F.when(F.col('cw_status') != 'Terminated/Non-renewed Contract', F.col('dd_next'))
         .otherwise(F.coalesce('dd_next_county_max', 'dd_next_state_max')))
    .withColumn('dd_next_min', 
        F.when(F.col('cw_status') != 'Terminated/Non-renewed Contract', F.col('dd_next'))
         .otherwise(F.coalesce('dd_next_county_min', 'dd_next_state_min')))
    # Calculate deductible differences using bounds
    .withColumn('dd_diff_max', F.col('dd_next_max') - F.col('dd_curr'))
    .withColumn('dd_diff_min', F.col('dd_next_min') - F.col('dd_curr'))
    .withColumn('wt_dd_diff_max', F.col('dd_diff_max') * F.col('enrollment'))
    .withColumn('wt_dd_diff_min', F.col('dd_diff_min') * F.col('enrollment'))
    .select([
        'plan_type', 'offers_part_d', 'parent_organization', 
        'organization_marketing_name', 'state', 'county',
        'cid_curr', 'pid_curr', 'enrollment',
        'dd_curr', 'dd_next_max', 'dd_next_min',
        'dd_diff_max', 'dd_diff_min',
        'wt_dd_diff_max', 'wt_dd_diff_min', 'cw_status'
    ])
)

# COMMAND ----------

# Calculate average deductible changes across all plans
# Range represents min/max possible changes based on regional bounds
(df_analysis
    .select(
        F.sum('wt_dd_diff_max')/F.sum('enrollment'),  # Upper bound of avg change
        F.sum('wt_dd_diff_min')/F.sum('enrollment'),  # Lower bound of avg change
        F.sum('enrollment')                           # Total members affected
    )
).show()

# Analyze deductible changes for MA/MAPD plans only (excluding PDP)
# Helps understand impact on Medicare Advantage population specifically
(df_analysis
    .where("plan_type != 'Medicare Prescription Drug Plan'")
    .select(
        F.sum('wt_dd_diff_max')/F.sum('enrollment'),  # MA/MAPD upper bound
        F.sum('wt_dd_diff_min')/F.sum('enrollment'),  # MA/MAPD lower bound
        F.sum('enrollment')                           # MA/MAPD members affected
    )
).show()

# Analyze impact on members who had $0 deductible in 2024
# Critical metric: shows how many members lose their "no deductible" benefit
(df_analysis
    .where("dd_curr = 0")
    .select(
        F.sum('wt_dd_diff_max')/F.sum('enrollment'),  # Max increase from $0
        F.sum('wt_dd_diff_min')/F.sum('enrollment'),  # Min increase from $0
        F.sum('enrollment')                           # Members losing $0 deductible
    )
).show()

# COMMAND ----------


