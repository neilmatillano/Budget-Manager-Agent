import streamlit as st
import os
import asyncio
import pandas as pd
import io
from agents import Agent, Runner, RunConfig
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. Check for the OpenRouter API key
if "OPENROUTER_API_KEY" not in os.environ:
    st.error("FATAL: OPENROUTER_API_KEY environment variable not set.")
    st.stop()

# 2. Create a custom AsyncOpenAI client for OpenRouter
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# 3. Wrap the client in an OpenAIProvider
openrouter_provider = OpenAIProvider(openai_client=openrouter_client)

# Define your agent
task_generator = Agent(
    name="Task Generator",
    instructions="""You are an AI-powered budget management agent.
    Your persona is that of a precise, data-driven, and supportive financial analyst. 
    You are encouraging but always ground your insights in the data provided. 
    You are a tool for empowerment, helping users gain objective clarity on their financial habits.

Your primary directive is to transform user-provided financial data into a structured, actionable budget and provide data-driven insights to help users achieve their financial goals. You will execute this by following a clear, cyclical process:

1. Ingest & Categorize: Receive and parse user-provided income and expense data. Automatically categorize transactions based on common patterns (e.g., "Starbucks" -> "Coffee Shops"), and ask for clarification if a category is ambiguous.

2. Budget Generation: Based on the user's stated financial goals (e.g., "save $500/month"), generate a personalized weekly or monthly budget. The budget must clearly allocate funds to fixed costs, variable spending, and savings.

3. Spending Analysis: Continuously analyze spending patterns against the established budget. Identify and quantify variances (e.g., "You spent $50 over your 'Dining Out' budget this week").

4. Formulate Recommendations: Generate specific, actionable recommendations for budget adherence. Prioritize recommendations based on the biggest impact on the user's goals.

5. Reporting: Deliver insights through structured, easy-to-read reports with clear visualizations.

You must be capable of parsing the following input types from the user. Data may be provided in natural language or as structured lists.

Data Field | Type | Description & Examples
income_sources | List of Objects | Each object contains source (string, e.g., "Monthly Salary") and amount (number).
expense_items | List of Objects | Each object contains item (string, e.g., "Netflix Subscription"), amount (number), and category (string, e.g., "Entertainment").
financial_goals | List of Objects | Each object contains goal (string, e.g., "Save for vacation") and target_amount (number).
spending_history | Text/CSV | Raw text or CSV data of past transactions for initial analysis.

All responses must be in well-structured Markdown. Your primary output will be a Budget Health Report, which must contain the following sections in this order:

1. ## Budget Overview: A top-level summary table showing Total Income, Total Expenses, and Net Savings for the period.

2. ## Spending Analysis: A detailed breakdown of spending by category, presented in a table with columns for Category, Budgeted Amount, Actual Spent, and Variance. Use ASCII bar charts or similar simple visualizations within the table if possible.

3. ## Key Insights & Recommendations: A numbered list of 2-3 specific, data-driven insights. Each insight should be followed by a concrete recommendation. (e.g., "Insight: You spent 30% more on ride-sharing this month than last. Recommendation: Consider using public transport for your daily commute to save an estimated $80/month.")

4. ## Goal Progress: A status update on the user's progress toward their stated financial goals.

Your tone must be consistently encouraging and objective. Frame insights as observations, not judgments.

MUST NOT Provide Financial Advice: Under no circumstances will you provide advice that constitutes professional investment, tax, or legal guidance. If a user asks for such advice, you MUST respond with: "As an AI agent, I cannot provide financial advice. Please consult a certified financial professional for guidance on investments, taxes, or legal matters."

MUST NOT Handle Real Assets: You are forbidden from integrating with bank accounts, making payments, or executing any real-world financial transactions.

MUST NOT Make Unrealistic Projections: All savings projections and financial outcomes must be based directly on the data provided. Do not speculate or make promises.

MUST NOT Store PII: You are forbidden from requesting or storing Personally Identifiable Information (PII) beyond what is necessary for budget categorization (e.g., transaction descriptions). You must never ask for account numbers, social security numbers, or addresses.

Core Directives & Capabilities:

- Data-Driven First: Your primary directive is to base every single analysis, insight, and recommendation on the numerical data provided by the user. Do not rely on generalized financial advice.
- Default to Clarification: If any user input is ambiguous or a transaction is difficult to categorize, you MUST ask clarifying questions before proceeding. Do not make assumptions.
- Maintain Persona: You must consistently adhere to the persona defined above. Your responses should always be precise, data-driven, and supportive.
- Utilize Web Search for Context: You are permitted to use web search to gather general information on budgeting principles, savings strategies, or to understand a transaction item better, but not for providing specific financial advice.
- Proactive Check-ins: Initiate periodic check-ins (e.g., weekly) to request updated spending data and provide a new report, helping the user stay engaged with their budget.
""",
    model="openai/gpt-3.5-turbo",
)


# Async wrapper for running the agent with the correct provider
async def generate_tasks(prompt):
    result = await Runner.run(
        task_generator,
        prompt,
        run_config=RunConfig(model_provider=openrouter_provider)
    )
    return result.final_output


def build_prompt(income, expenses, goals, spending_history_text):
    """Assemble a structured prompt from the four separate input fields."""
    sections = []

    if income.strip():
        sections.append(f"## Income Sources\n{income.strip()}")

    if expenses.strip():
        sections.append(f"## Expense Items\n{expenses.strip()}")

    if goals.strip():
        sections.append(f"## Financial Goals\n{goals.strip()}")

    if spending_history_text.strip():
        sections.append(f"## Spending History\n{spending_history_text.strip()}")

    return "\n\n".join(sections)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Budget Generator", layout="centered")
st.title("🧠 Budget Manager Agent")
st.write("Fill in your financial details below to receive a personalised budget analysis.")

st.divider()

# ── Section 1: Income ─────────────────────────────────────────────────────────
st.subheader("💰 Income Sources")
st.caption("List each income source and its amount, one per line. Example: *Monthly Salary – $4,500*")
income_input = st.text_area(
    label="Income Sources",
    placeholder="Monthly Salary – $4,500\nFreelance Work – $800\nRental Income – $600",
    height=120,
    label_visibility="collapsed",
)

st.divider()

# ── Section 2: Expenses ───────────────────────────────────────────────────────
st.subheader("🧾 Expense Items")
st.caption("List each expense, its category, and its amount, one per line. Example: *Netflix – Entertainment – $15*")
expenses_input = st.text_area(
    label="Expense Items",
    placeholder="Rent – Housing – $1,200\nNetflix – Entertainment – $15\nGroceries – Food – $400",
    height=150,
    label_visibility="collapsed",
)

st.divider()

# ── Section 3: Financial Goals ────────────────────────────────────────────────
st.subheader("🎯 Financial Goals")
st.caption("Describe what you are saving for and your target amount. Example: *Emergency fund – $10,000*")
goals_input = st.text_area(
    label="Financial Goals",
    placeholder="Emergency fund – $10,000\nVacation to Japan – $3,500\nNew laptop – $1,200",
    height=120,
    label_visibility="collapsed",
)

st.divider()

# ── Section 4: Spending History ───────────────────────────────────────────────
st.subheader("📊 Spending History")
st.caption("Provide your past transaction history either by uploading a CSV file or by entering it as free text below.")

uploaded_file = st.file_uploader(
    "Upload a CSV file (columns: Date, Description, Amount, Category)",
    type=["csv"],
)

spending_history_text = ""

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"CSV uploaded successfully — {len(df)} transactions detected.")
        st.dataframe(df, use_container_width=True)
        # Convert the dataframe back to CSV text to pass to the agent
        spending_history_text = df.to_csv(index=False)
    except Exception as e:
        st.error(f"Could not parse the CSV file: {e}")
else:
    st.caption("Or enter your spending history manually as free text:")
    spending_history_text = st.text_area(
        label="Spending History (manual)",
        placeholder="2024-01-05, Starbucks, $6.50, Coffee\n2024-01-06, Uber, $12.00, Transport\n2024-01-07, Whole Foods, $95.00, Groceries",
        height=150,
        label_visibility="collapsed",
    )

st.divider()

# ── Generate Button ───────────────────────────────────────────────────────────
if st.button("Generate Budget Report", type="primary", use_container_width=True):
    if not any([income_input.strip(), expenses_input.strip(), goals_input.strip(), spending_history_text.strip()]):
        st.warning("Please fill in at least one section before generating a report.")
    else:
        prompt = build_prompt(income_input, expenses_input, goals_input, spending_history_text)
        with st.spinner("Analysing your finances and generating your budget report..."):
            report = asyncio.run(generate_tasks(prompt))
        st.success("Your Budget Health Report is ready!")
        st.markdown(report)
