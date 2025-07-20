import streamlit as st
import pandas as pd
from scipy import stats

st.title("Basic Statistical Testing App")

# Step 1: Upload and Preview Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Step 2: Let user select columns for testing
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    group_col = st.selectbox("Select the grouping column (categorical)", df.columns)
    value_col = st.selectbox("Select the value column (numerical)", num_cols)

    groups = df[group_col].dropna().unique()
    if len(groups) == 2:
        group1 = df[df[group_col] == groups[0]][value_col].dropna()
        group2 = df[df[group_col] == groups[1]][value_col].dropna()

        # Step 3: Conduct t-test
        t_stat, p_val = stats.ttest_ind(group1, group2)
        st.write(f"**t-statistic:** {t_stat:.4f}")
        st.write(f"**p-value:** {p_val:.4g}")

        if p_val < 0.05:
            st.success("Result: Significant difference detected (p < 0.05)")
        else:
            st.info("Result: No significant difference detected (p >= 0.05)")
    else:
        st.warning("The selected grouping column must have exactly two unique values for a t-test.")

