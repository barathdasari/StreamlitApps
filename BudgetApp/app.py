import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Monthly Budget App", layout="centered")

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame(columns=["Date", "Type", "Category", "Amount"])

st.title("💰 Simple Monthly Budgeting App")

# Input Section
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        trans_type = st.selectbox("Transaction Type", ["Income", "Expense"])
        category = st.selectbox("Category", 
            ["Salary", "Freelance", "Groceries", "Rent", "Utilities", "Entertainment", "Misc"]
            if trans_type == "Expense"
            else ["Salary", "Freelance", "Investment"])
    with col2:
        amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0)
        date = st.date_input("Date", value=datetime.today())

    submitted = st.form_submit_button("Add Transaction")
    if submitted and amount > 0:
        new_row = {"Date": date, "Type": trans_type, "Category": category, "Amount": amount}
        st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([new_row])], ignore_index=True)
        st.success(f"{trans_type} of ₹{amount:.2f} added!")

# Summary Section
st.subheader("📊 Monthly Summary")

df = st.session_state.transactions
df["Date"] = pd.to_datetime(df["Date"])
df_month = df[df["Date"].dt.month == datetime.today().month]

total_income = df_month[df_month["Type"] == "Income"]["Amount"].sum()
total_expense = df_month[df_month["Type"] == "Expense"]["Amount"].sum()
balance = total_income - total_expense

col1, col2, col3 = st.columns(3)
col1.metric("Income", f"₹{total_income:,.2f}")
col2.metric("Expense", f"₹{total_expense:,.2f}")
col3.metric("Balance", f"₹{balance:,.2f}", delta_color="inverse")

# Pie chart
if not df_month[df_month["Type"] == "Expense"].empty:
    expense_by_cat = df_month[df_month["Type"] == "Expense"].groupby("Category")["Amount"].sum()
    st.subheader("💸 Expense Breakdown")
    st.pyplot(expense_by_cat.plot.pie(autopct="%1.1f%%", figsize=(5, 5), title="Expenses by Category", ylabel="").get_figure())

# Show table
st.subheader("📅 Transaction Log")
st.dataframe(df_month.sort_values(by="Date", ascending=False).reset_index(drop=True))

# Clear button
if st.button("Clear All Data"):
    st.session_state["transactions"] = pd.DataFrame(columns=["Date", "Type", "Category", "Amount"])
    st.success("All data cleared.")