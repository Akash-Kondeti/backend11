import os
import tempfile
from typing import Any
from datetime import datetime
import uuid
import re
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import docx
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import pdfplumber
    import docx
    import pandas as pd
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Please run 'pip install -r requirements.txt'")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CATEGORY_LIST = [
    "bank-transactions",
    "invoices",
    "bills",
    "inventory",
    "item-restocks",
    "manual-journals",
    "general-ledgers",
    "general-entries"
]

# In-memory transaction store
transactions = []

async def extract_text(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    text = ""
    try:
        if ext == 'pdf':
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ''
        elif ext in ['docx']:
            doc_file = docx.Document(tmp_path)
            text = '\n'.join([p.text for p in doc_file.paragraphs])
        elif ext in ['csv', 'xls', 'xlsx']:
            df = pd.read_csv(tmp_path) if ext == 'csv' else pd.read_excel(tmp_path)
            text = df.to_string()
        elif ext == 'txt':
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = "Unsupported file type."
    finally:
        os.remove(tmp_path)
    return text

async def summarize_and_classify(text: str) -> Any:
    category_descriptions = {
        "bank-transactions": "Bank statements, transaction lists, account activity, deposits, withdrawals, transfers.",
        "invoices": "Sales invoices, bills sent to customers, payment requests.",
        "bills": "Bills received, utility bills, vendor bills, payables.",
        "inventory": "Inventory lists, stock reports, itemized inventory.",
        "item-restocks": "Purchase orders, restock requests, inventory replenishment.",
        "manual-journals": "Manual journal entries, adjusting entries, non-standard accounting entries.",
        "general-ledgers": "General ledger reports, account summaries, trial balances.",
        "general-entries": "Miscellaneous entries, uncategorized financial records."
    }
    category_list_str = "\n".join([f"- {cat}: {desc}" for cat, desc in category_descriptions.items()])

    prompt = (
        "You are a financial document classifier. "
        "Classify the following document into one of these categories, using the most specific and relevant one:\n"
        f"{category_list_str}\n"
        "If the document is a bank statement or transaction list, extract the net total (sum of all credits minus debits, or the final balance if available) as the 'amount'.\n"
        "For other categories, extract the main amount (total, invoice amount, bill amount, or transaction amount) as a number (no currency symbol, just the number, or 0 if not found).\n"
        "Return ONLY a JSON object with three fields: 'summary' (a concise summary of the document), 'category' (the best matching category from the list), and 'amount' (the main amount as a number).\n"
        "Document:\n"
        f"{text[:4000]}"
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = ""
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            import json
            result = json.loads(match.group(0))
            return result
        else:
            # Fallback: try to parse category, summary, and amount from text
            lines = content.splitlines() if isinstance(content, str) else []
            summary = ""
            category = None
            amount = 0
            for line in lines:
                if 'category' in line.lower():
                    category = line.split(':', 1)[-1].strip().strip('"')
                elif 'summary' in line.lower():
                    summary = line.split(':', 1)[-1].strip().strip('"')
                elif 'amount' in line.lower():
                    try:
                        amount = float(line.split(':', 1)[-1].strip().replace(',', '').replace('$', ''))
                    except Exception:
                        amount = 0
            return {"summary": summary, "category": category, "amount": amount}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze-document/")
async def analyze_document(file: UploadFile = File(...)):
    try:
        text = await extract_text(file)
        if not text or text.strip() == "Unsupported file type.":
            return JSONResponse(status_code=400, content={"error": "Unsupported or empty file."})
        result = await summarize_and_classify(text)
        if "error" in result:
            return JSONResponse(status_code=500, content={"error": result["error"]})
        summary = result.get("summary", "")
        category = result.get("category", None)
        amount = result.get("amount", 0)
        now = datetime.now()
        upload_date = now.strftime("%d/%m/%Y")
        doc_id = str(uuid.uuid4())
        # Add transaction to in-memory store
        dashboard_category = None
        if category == "bank-transactions":
            dashboard_category = "Cash Balance"
        elif category == "invoices":
            dashboard_category = "Revenue"
        elif category == "bills":
            dashboard_category = "Expenses"
        elif category == "manual-journals":
            dashboard_category = "Net Burn"
        # Default type logic: treat invoices and bank-transactions as credit, bills and others as debit
        t_type = "credit" if category in ["invoices", "bank-transactions"] or (isinstance(amount, (int, float)) and amount >= 0) else "debit"
        transactions.append({
            "id": doc_id,
            "date": now.strftime("%Y-%m-%d"),
            "description": summary or file.filename,
            "name": file.filename,  # Add file name
            "amount": amount,
            "category": category,
            "type": t_type,
            "dashboardCategory": dashboard_category or ""
        })
        return {
            "id": doc_id,
            "name": file.filename,
            "status": "completed",
            "category": category,
            "confidence": 0.95,
            "uploadDate": upload_date,
            "summary": summary,
            "amount": amount,
            "dashboardCategory": dashboard_category or ""
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "id": str(uuid.uuid4()),
            "name": file.filename,
            "status": "failed",
            "category": None,
            "confidence": 0.0,
            "uploadDate": datetime.now().strftime("%d/%m/%Y"),
            "summary": None,
            "amount": 0,
            "error": str(e)
        })

@app.get("/dashboard-summary/")
def get_dashboard_summary():
    cash_balance = sum(
        t["amount"] if t["type"] == "credit" else -t["amount"]
        for t in transactions if t["dashboardCategory"] == "Cash Balance"
    )
    revenue = sum(
        t["amount"] for t in transactions if t["dashboardCategory"] == "Revenue"
    )
    expenses = sum(
        t["amount"] for t in transactions if t["dashboardCategory"] == "Expenses"
    )
    net_burn = sum(
        t["amount"] for t in transactions if t["dashboardCategory"] == "Net Burn"
    )
    return {
        "cashBalance": cash_balance,
        "revenue": revenue,
        "expenses": expenses,
        "netBurn": net_burn
    }

@app.post("/classify-transaction/")
async def classify_transaction(data: dict = Body(...)):
    description = data.get("description", "")
    if not description:
        return JSONResponse(status_code=400, content={"error": "Missing description."})
    # Use a simplified prompt for transaction description
    category_descriptions = {
        "bank-transactions": "Bank transactions, deposits, withdrawals, transfers.",
        "invoices": "Sales invoices, payment requests.",
        "bills": "Bills received, payables.",
        "inventory": "Inventory or stock related.",
        "item-restocks": "Restock or purchase orders.",
        "manual-journals": "Manual journal entries, adjustments.",
        "general-ledgers": "General ledger or account summaries.",
        "general-entries": "Miscellaneous or uncategorized entries."
    }
    category_list_str = "\n".join([f"- {cat}: {desc}" for cat, desc in category_descriptions.items()])
    prompt = (
        "Classify the following transaction description into one of these categories, using the most specific and relevant one:\n"
        f"{category_list_str}\n"
        "Return ONLY a JSON object with two fields: 'category' (the best matching category from the list), and 'dashboardCategory' (a high-level dashboard grouping if possible, or null).\n"
        "Description:\n"
        f"{description}"
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = ""
        import re, json
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            return result
        else:
            # Fallback: try to parse category and dashboardCategory from text
            lines = content.splitlines() if isinstance(content, str) else []
            category = None
            dashboardCategory = None
            for line in lines:
                if 'category' in line.lower():
                    category = line.split(':', 1)[-1].strip().strip('"')
                elif 'dashboardcategory' in line.lower():
                    dashboardCategory = line.split(':', 1)[-1].strip().strip('"')
            return {"category": category, "dashboardCategory": dashboardCategory}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-financial-statements/")
async def generate_financial_statements(data: dict = Body(...)):
    try:
        transactions = data.get("transactions", [])

        # Process transactions to generate financial statements
        balance_sheet = []
        profit_loss = []
        trial_balance = []
        cash_flow = []

        # Group transactions by category and type
        account_totals = {}

        for transaction in transactions:
            category = transaction.get("category", "general-entries")
            amount = transaction.get("amount", 0)
            t_type = transaction.get("type", "debit")
            dashboard_category = transaction.get("dashboardCategory", "")

            # Create account key
            account_key = f"{category}_{dashboard_category}" if dashboard_category else category

            if account_key not in account_totals:
                account_totals[account_key] = {"debit": 0, "credit": 0, "amount": 0}

            if t_type == "debit":
                account_totals[account_key]["debit"] += amount
                account_totals[account_key]["amount"] += amount
            else:
                account_totals[account_key]["credit"] += amount
                account_totals[account_key]["amount"] += amount

        # Generate Balance Sheet
        for account_key, totals in account_totals.items():
            category, dashboard_category = account_key.split("_", 1) if "_" in account_key else (account_key, "")

            # Determine account type for balance sheet
            if dashboard_category == "Cash Balance":
                balance_sheet.append({
                    "account": "Cash and Cash Equivalents",
                    "type": "asset",
                    "amount": totals["amount"],
                    "category": "Current Assets"
                })
            elif category == "invoices":
                balance_sheet.append({
                    "account": "Accounts Receivable",
                    "type": "asset",
                    "amount": totals["amount"],
                    "category": "Current Assets"
                })
            elif category == "bills":
                balance_sheet.append({
                    "account": "Accounts Payable",
                    "type": "liability",
                    "amount": totals["amount"],
                    "category": "Current Liabilities"
                })
            elif dashboard_category == "Revenue":
                profit_loss.append({
                    "account": "Revenue",
                    "type": "revenue",
                    "amount": totals["amount"]
                })
            elif dashboard_category == "Expenses":
                profit_loss.append({
                    "account": "Expenses",
                    "type": "expense",
                    "amount": totals["amount"]
                })

        # Generate Trial Balance
        for account_key, totals in account_totals.items():
            category, dashboard_category = account_key.split("_", 1) if "_" in account_key else (account_key, "")
            account_name = dashboard_category if dashboard_category else category.replace("-", " ").title()

            trial_balance.append({
                "account": account_name,
                "debit": totals["debit"],
                "credit": totals["credit"]
            })

        # Generate Cash Flow (simplified)
        cash_inflow = sum(t["amount"] for t in transactions if t.get("dashboardCategory") == "Revenue")
        cash_outflow = sum(t["amount"] for t in transactions if t.get("dashboardCategory") == "Expenses")

        cash_flow = [
            {"type": "Operating", "description": "Cash from Operations", "amount": cash_inflow - cash_outflow},
            {"type": "Operating", "description": "Cash Inflow", "amount": cash_inflow},
            {"type": "Operating", "description": "Cash Outflow", "amount": -cash_outflow}
        ]

        return {
            "balanceSheet": balance_sheet,
            "profitLoss": profit_loss,
            "trialBalance": trial_balance,
            "cashFlow": cash_flow
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)}) 