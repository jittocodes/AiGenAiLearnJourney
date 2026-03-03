import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ==========================================================
# 1️⃣ SOURCE SCHEMA (PRODUCTION GRADE – COMPLETE)
# ==========================================================


another_test_data = [
  {
    "source_system": "erp",
    "source_table": ["orders"],
    "source_columns": ["order_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "order_id",
    "transformation_logic": "direct map"
  },
  {
    "source_system": "erp",
    "source_table": ["orders"],
    "source_columns": ["order_date"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "order_date_key",
    "transformation_logic": "FORMAT_DATE('%Y%m%d', order_date)"
  },
  {
    "source_system": "erp",
    "source_table": ["orders"],
    "source_columns": ["customer_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "customer_id",
    "transformation_logic": "direct map"
  },
  {
    "source_system": "erp",
    "source_table": ["order_items"],
    "source_columns": ["quantity", "unit_price"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "gross_revenue",
    "transformation_logic": "SUM(quantity * unit_price)"
  },
  {
    "source_system": "erp",
    "source_table": ["order_items"],
    "source_columns": ["quantity", "unit_price", "discount_pct"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "net_revenue",
    "transformation_logic": "SUM(quantity * unit_price * (1 - discount_pct))"
  },
  {
    "source_system": "erp",
    "source_table": ["customers"],
    "source_columns": ["customer_type"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "customer_segment",
    "transformation_logic": "CASE WHEN customer_type IN ('VIP','PREMIUM') THEN 'HIGH_VALUE' ELSE 'STANDARD' END"
  },
  {
    "source_system": "crm",
    "source_table": ["leads_current", "leads_archive"],
    "source_columns": ["lead_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "latest_lead_id",
    "transformation_logic": "MAX_BY(lead_id, updated_at)"
  },
  {
    "source_system": "crm",
    "source_table": ["leads_current", "leads_archive"],
    "source_columns": ["status"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "latest_lead_status",
    "transformation_logic": "ANY_VALUE(status) OVER (PARTITION BY customer_id ORDER BY updated_at DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)"
  },
  {
    "source_system": "finance",
    "source_table": ["payments"],
    "source_columns": ["amount"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "total_payments",
    "transformation_logic": "SUM(amount)"
  },
  {
    "source_system": "finance",
    "source_table": ["refunds"],
    "source_columns": ["refund_amount"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "total_refunds",
    "transformation_logic": "SUM(refund_amount)"
  },
  {
    "source_system": "finance",
    "source_table": ["payments", "refunds"],
    "source_columns": ["transaction_date"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "transaction_month",
    "transformation_logic": "FORMAT_DATE('%Y-%m', transaction_date)"
  },
  {
    "source_system": "marketing",
    "source_table": ["campaign_clicks"],
    "source_columns": ["click_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "total_clicks",
    "transformation_logic": "COUNT(DISTINCT click_id)"
  },
  {
    "source_system": "marketing",
    "source_table": ["campaign_impressions"],
    "source_columns": ["impression_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "total_impressions",
    "transformation_logic": "COUNT(DISTINCT impression_id)"
  },
  {
    "source_system": "marketing",
    "source_table": ["campaign_clicks", "campaign_impressions"],
    "source_columns": ["campaign_id"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "ctr",
    "transformation_logic": "SAFE_DIVIDE(total_clicks, total_impressions)"
  },
  {
    "source_system": "erp",
    "source_table": ["orders"],
    "source_columns": ["order_date"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "is_recent_customer",
    "transformation_logic": "CASE WHEN DATE_DIFF(CURRENT_DATE(), order_date, DAY) <= 90 THEN TRUE ELSE FALSE END"
  },
  {
    "source_system": "erp",
    "source_table": ["orders"],
    "source_columns": ["order_date"],
    "target_dataset": "analytics",
    "target_table": "fact_customer_360",
    "target_column": "customer_lifetime_rank",
    "transformation_logic": "DENSE_RANK() OVER (PARTITION BY customer_id ORDER BY order_date)"
  }
]

SOURCE_SCHEMA = """
erp.orders(
  order_id STRING,
  customer_id STRING,
  order_date DATE,
  order_status STRING,
  created_at TIMESTAMP
);

erp.order_items(
  order_item_id STRING,
  order_id STRING,
  product_id STRING,
  quantity INT64,
  unit_price NUMERIC,
  discount_pct FLOAT64,
  created_at TIMESTAMP
);

erp.customers(
  customer_id STRING,
  customer_name STRING,
  customer_type STRING,
  signup_date DATE
);

crm.leads_current(
  lead_id STRING,
  customer_id STRING,
  campaign_id STRING,
  status STRING,
  updated_at TIMESTAMP,
  created_at TIMESTAMP
);

crm.leads_archive(
  lead_id STRING,
  customer_id STRING,
  campaign_id STRING,
  status STRING,
  updated_at TIMESTAMP,
  created_at TIMESTAMP
);

finance.payments(
  payment_id STRING,
  customer_id STRING,
  order_id STRING,
  amount NUMERIC,
  transaction_date DATE,
  payment_method STRING
);

finance.refunds(
  refund_id STRING,
  customer_id STRING,
  order_id STRING,
  refund_amount NUMERIC,
  transaction_date DATE,
  reason STRING
);

marketing.campaign_clicks(
  click_id STRING,
  customer_id STRING,
  campaign_id STRING,
  click_timestamp TIMESTAMP
);

marketing.campaign_impressions(
  impression_id STRING,
  customer_id STRING,
  campaign_id STRING,
  impression_timestamp TIMESTAMP
);
"""


# ==========================================================
# 2️⃣ MAPPING METADATA (YOUR FULL DATA)
# ==========================================================

another_test_data = [...]  # <-- paste your full mapping JSON here


# ==========================================================
# 3️⃣ LLM CONFIGURATION (STRICT JSON SCHEMA MODE)
# ==========================================================

cardinality_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CardinalityPlan",
            "schema": {
                "type": "object",
                "properties": {
                    "table_roles": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["table_roles"]
            }
        }
    }
)

planning_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "QueryPlan",
            "schema": {
                "type": "object",
                "properties": {
                    "base_table": {"type": "string"},
                    "ctes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cte_name": {"type": "string"},
                                "source_tables": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "strategy": {"type": "string"},
                                "group_by": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["cte_name", "source_tables", "strategy", "group_by"]
                        }
                    },
                    "join_keys": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["base_table", "ctes", "join_keys"]
            }
        }
    }
)


# ==========================================================
# 4️⃣ PROMPTS
# ==========================================================

cardinality_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a warehouse architect.

Using the schema and mapping:
- Detect base table.
- Detect one_to_many tables.
- Detect one_to_one dimension tables.
- Detect latest_record tables (using updated_at or MAX_BY logic).

Return JSON only.
"""),
    ("human", """
Source Schema:
{source_schema}

Mapping Metadata:
{mapping_json}
""")
])

planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a BigQuery planner.

Rules:
- Each one_to_many must be aggregated in separate CTE.
- latest_record tables must use ROW_NUMBER().
- Do not multiply rows of base table.
- Use UNION ALL where needed.
Return JSON only.
"""),
    ("human", """
Mapping:
{mapping_json}

Cardinality:
{cardinality_plan}
""")
])


# ==========================================================
# 5️⃣ JSON PARSER
# ==========================================================

def parse_json(msg):
    return json.loads(msg.content)


# ==========================================================
# 6️⃣ CHAINS
# ==========================================================

cardinality_chain = (
    cardinality_prompt
    | cardinality_llm
    | RunnableLambda(parse_json)
)

planning_chain = (
    planning_prompt
    | planning_llm
    | RunnableLambda(parse_json)
)


# ==========================================================
# 7️⃣ DETERMINISTIC SQL BUILDER
# ==========================================================

def build_sql(inputs):
    plan = inputs["query_plan"]

    sql = ""

    if plan["ctes"]:
        sql += "WITH\n"
        cte_blocks = []

        for cte in plan["ctes"]:
            block = f"{cte['cte_name']} AS (\n"
            block += f"  SELECT *\n"
            block += f"  FROM {', '.join(cte['source_tables'])}\n"
            if cte["group_by"]:
                block += f"  GROUP BY {', '.join(cte['group_by'])}\n"
            block += ")"
            cte_blocks.append(block)

        sql += ",\n".join(cte_blocks) + "\n"

    sql += f"SELECT *\nFROM {plan['base_table']}"

    for cte in plan["ctes"]:
        sql += f"\nLEFT JOIN {cte['cte_name']} USING({', '.join(plan['join_keys'])})"

    print("\n=========== FINAL SQL ===========\n")
    print(sql)
    print("\n=================================\n")

    return sql


# ==========================================================
# 8️⃣ EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\nSTEP 1: CARDINALITY DETECTION\n")

    cardinality_result = cardinality_chain.invoke({
        "source_schema": SOURCE_SCHEMA,
        "mapping_json": json.dumps(another_test_data, indent=2)
    })

    print(cardinality_result)

    print("\nSTEP 2: QUERY PLANNING\n")

    query_plan = planning_chain.invoke({
        "mapping_json": json.dumps(another_test_data, indent=2),
        "cardinality_plan": json.dumps(cardinality_result, indent=2)
    })

    print(query_plan)

    print("\nSTEP 3: SQL BUILD\n")

    build_sql({"query_plan": query_plan})
