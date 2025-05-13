# Streamlit app: Distribution Fit Identifier for specified distributions with AD & P-value
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox_normmax, boxcox, anderson, kstest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

st.title("Distribution Fit & Goodness-of-Fit Tests")

# Step 1: Upload data
uploaded_file = st.file_uploader("Upload CSV with numeric columns:", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = st.multiselect("Select column for analysis", numeric_cols, default=numeric_cols[:1])

    # List of distributions to test
    dist_info = [
        ('Normal', 'norm', {}),
        ('Gamma (2P)', 'gamma', {'floc': 0}),
        ('Gamma (3P)', 'gamma', {}),
        ('Exponential', 'expon', {}),
        ('Weibull (2P)', 'weibull_min', {'floc': 0}),
        ('Weibull (3P)', 'weibull_min', {}),
        ('Logistic', 'logistic', {}),
        ('Log-Logistic', 'fisk', {}),
        ('Smallest Extreme Value', 'gumbel_r', {})
    ]

    for col in cols:
        st.header(f"Analysis for '{col}'")
        data = df[col].dropna().values

        # Raw normality test
        sw_stat, sw_p = stats.shapiro(data)
        st.write(f"**Shapiro–Wilk**: statistic={sw_stat:.4f}, p-value={sw_p:.4f}")

        # Decide on transform
        transform = 'None'
        if sw_p < 0.05:
            st.warning("Data not normal: applying transform.")
            try:
                lam = boxcox_normmax(data + 1e-8)
                x = boxcox(data + 1e-8, lam)
                transform = f"Box-Cox (λ={lam:.3f})"
            except:
                x = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
                transform = 'Yeo-Johnson'
            st.write(f"Transformation applied: {transform}")
        else:
            st.success("Data appears normal; no transform.")
            x = data

        # Plot original vs transformed
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        axs[0].hist(data, bins=30, color='skyblue', edgecolor='black')
        axs[0].set_title('Original Data')
        axs[1].hist(x, bins=30, color='lightgreen', edgecolor='black')
        axs[1].set_title(f'Transformed Data ({transform})')
        st.pyplot(fig)

        # Goodness-of-fit
        st.subheader("Goodness-of-Fit Results")
        results = []
        for name, alias, fit_kwargs in dist_info:
            try:
                dist = getattr(stats, alias)
                # Fit distribution
                params = dist.fit(x, **fit_kwargs)
                # AD statistic if supported
                try:
                    ad_stat = anderson(x, dist=alias).statistic
                except:
                    ad_stat = np.nan
                # KS test for p-value
                ks_stat, ks_p = kstest(x, alias, args=params)
                results.append({'name': name, 'alias': alias, 'params': params,
                                'AD': ad_stat, 'p_value': ks_p})
                st.write(f"{name}: AD={ad_stat if not np.isnan(ad_stat) else 'N/A'}, KS p-value={ks_p:.4f}")
            except Exception as e:
                st.write(f"{name}: Test failed ({e})")

        # Select best: lowest AD (ignoring N/A), then highest p-value
        df_res = pd.DataFrame(results)
        # Replace NAs with large number to deprioritize
        df_res['AD_sort'] = df_res['AD'].fillna(df_res['AD'].max()*10)
        best = df_res.sort_values(['AD_sort', 'p_value'], ascending=[True, False]).iloc[0]
        st.success(f"**Best Fit**: {best['name']} (AD={best['AD']:.4f}, p-value={best['p_value']:.4f})")

        # Comparison chart
        fig2, ax2 = plt.subplots(figsize=(8,4))
        names = df_res['name']
        ad_vals = df_res['AD_sort']
        pv_vals = df_res['p_value']
        idx = np.arange(len(names))
        w = 0.4
        ax2.bar(idx-w/2, ad_vals, w, label='AD Stat')
        ax2.bar(idx+w/2, pv_vals, w, label='KS p-value')
        ax2.set_xticks(idx)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_title('Goodness-of-Fit Comparison')
        ax2.legend()
        st.pyplot(fig2)
        st.write(f"**Transformation used:** {transform}")
