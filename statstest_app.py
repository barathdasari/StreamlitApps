import streamlit as st
import pandas as pd
from scipy import stats
import numpy as np
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Custom function to calculate power for proportion tests
def proportions_power(p1, p2, nobs1, alpha=0.05, ratio=1.0):
    """
    Calculate power for test of two independent proportions
    
    Parameters
    ----------
    p1, p2 : float
        Proportions for the two groups
    nobs1 : int
        Number of observations in first group
    alpha : float, optional
        Significance level for the test
    ratio : float, optional
        Ratio of sample sizes, nobs2/nobs1
        
    Returns
    -------
    power : float
        Power of the test
    """
    nobs2 = nobs1 * ratio
    p_pooled = (p1 * nobs1 + p2 * nobs2) / (nobs1 + nobs2)
    std_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1 / nobs1 + 1 / nobs2))
    
    # Critical value
    z_crit = norm.ppf(1 - alpha / 2)
    
    # Non-centrality parameter
    z_power = (p1 - p2) / np.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2)
    
    # Power calculation
    power = norm.cdf(z_power - z_crit) + norm.cdf(-z_power - z_crit)
    return power

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
        
        # Step 3: Select test type and parameters
        st.subheader("Statistical Testing Options")
        
        # Test type selection
        test_type = st.radio(
            "Select Test Type", 
            ["Test of Means", "Test of Proportions"]
        )
        
        # Alpha value selection
        alpha = st.select_slider(
            "Select Significance Level (α)", 
            options=[0.01, 0.05, 0.1],
            value=0.05
        )
        
        if test_type == "Test of Means":
            # Test for normality
            _, p_norm1 = stats.shapiro(group1) if len(group1) < 5000 else (0, 0.001)
            _, p_norm2 = stats.shapiro(group2) if len(group2) < 5000 else (0, 0.001)
            
            is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
            
            st.write("Data Distribution Assessment:")
            if is_normal:
                st.write("✅ Data appears to be normally distributed")
                test_method = st.selectbox(
                    "Select Statistical Test", 
                    ["Independent t-test", "Welch's t-test"]
                )
            else:
                st.write("⚠️ Data does not appear to be normally distributed")
                test_method = st.selectbox(
                    "Select Statistical Test", 
                    ["Mann-Whitney U Test", "Independent t-test", "Welch's t-test"]
                )
                
            # Display histograms
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(group1, kde=True, ax=ax[0])
            ax[0].set_title(f"Distribution of {groups[0]}")
            sns.histplot(group2, kde=True, ax=ax[1])
            ax[1].set_title(f"Distribution of {groups[1]}")
            st.pyplot(fig)
            
            # Conduct selected test
            if test_method == "Independent t-test":
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
                test_name = "Independent t-test"
                
                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                                    (len(group1) + len(group2) - 2))
                effect_size = abs(mean1 - mean2) / pooled_std
                
                # Calculate power
                power_analysis = TTestIndPower()
                power = power_analysis.power(effect_size=effect_size, 
                                            nobs1=len(group1), 
                                            alpha=alpha, 
                                            ratio=len(group2)/len(group1))
                
            elif test_method == "Welch's t-test":
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                test_name = "Welch's t-test"
                
                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                effect_size = abs(mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
                
                # Calculate power
                power_analysis = TTestIndPower()
                power = power_analysis.power(effect_size=effect_size, 
                                            nobs1=len(group1), 
                                            alpha=alpha, 
                                            ratio=len(group2)/len(group1))
                
            else:  # Mann-Whitney U Test
                u_stat, p_val = stats.mannwhitneyu(group1, group2)
                test_name = "Mann-Whitney U Test"
                t_stat = u_stat  # For consistency in output
                
                # Approximation of effect size for Mann-Whitney
                n1, n2 = len(group1), len(group2)
                effect_size = abs(u_stat - (n1 * n2 / 2)) / (n1 * n2 / 4)
                
                # Power calculation (approximation)
                power = TTestIndPower().power(effect_size=effect_size*0.8, 
                                             nobs1=len(group1), 
                                             alpha=alpha, 
                                             ratio=len(group2)/len(group1))
                
        else:  # Test of Proportions
            st.write("For proportion testing, data should be binary (0/1, True/False, etc.)")
            
            # Check if data is binary
            is_binary1 = set(group1.unique()).issubset({0, 1, True, False})
            is_binary2 = set(group2.unique()).issubset({0, 1, True, False})
            
            if is_binary1 and is_binary2:
                # Calculate proportions
                p1 = group1.mean()
                p2 = group2.mean()
                n1 = len(group1)
                n2 = len(group2)
                
                # Chi-square or z-test for proportions
                test_method = st.selectbox(
                    "Select Statistical Test", 
                    ["Z-test for Proportions", "Chi-Square Test"]
                )
                
                if test_method == "Z-test for Proportions":
                    z_stat, p_val = stats.proportions_ztest([sum(group1), sum(group2)], 
                                                           [len(group1), len(group2)])
                    t_stat = z_stat  # For consistency in output
                    test_name = "Z-test for Proportions"
                    
                    # Effect size (Cohen's h)
                    effect_size = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
                    
                    # Calculate power
                    power = proportions_power(p1=p1, p2=p2, nobs1=n1, alpha=alpha, ratio=n2/n1)
                    
                else:  # Chi-square test
                    observed = np.array([[sum(group1), len(group1) - sum(group1)], 
                                        [sum(group2), len(group2) - sum(group2)]])
                    chi2_stat, p_val, _, _ = stats.chi2_contingency(observed)
                    t_stat = chi2_stat  # For consistency in output
                    test_name = "Chi-Square Test"
                    
                    # Effect size (Cramer's V for 2x2)
                    n = n1 + n2
                    effect_size = np.sqrt(chi2_stat / n)
                    
                    # Calculate power
                    power = TTestIndPower().power(effect_size=effect_size*2, 
                                                 nobs1=n/2, 
                                                 alpha=alpha)
            else:
                st.error("For proportion testing, data must contain only binary values (0/1).")
                st.stop()
        
        # Step 4: Display results
        st.subheader("Statistical Test Results")
        st.write(f"**Test performed:** {test_name}")
        st.write(f"**Test statistic:** {t_stat:.4f}")
        st.write(f"**p-value:** {p_val:.4g}")
        st.write(f"**Effect size:** {effect_size:.4f}")
        st.write(f"**Statistical power:** {power:.4f}")
        
        if p_val < alpha:
            st.success(f"Result: Significant difference detected (p < {alpha})")
        else:
            st.info(f"Result: No significant difference detected (p ≥ {alpha})")
            
        if power < 0.8:
            st.warning("⚠️ The statistical power is below 0.8. This test may not be powerful enough to detect small differences.")
    else:
        st.warning("The selected grouping column must have exactly two unique values for these tests.")