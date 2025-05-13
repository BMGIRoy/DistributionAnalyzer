# Streamlit app: Distribution Fit Identifier for specified distributions
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

    # Helper: evaluate distributions on data x
    def evaluate_fits(x):
        res = []
        for name, alias, fit_kwargs in dist_info:
            try:
                dist = getattr(stats, alias)
                # Fit distribution
                params = dist.fit(x, **fit_kwargs)
                # Compute AD if supported
                try:
                    ad_stat = anderson(x, dist=alias).statistic
                except:
                    ad_stat = np.nan
                # KS test p-value
                ks_stat, ks_p = kstest(x, alias, args=params)
                res.append({'name': name, 'alias': alias, 'params': params,
                            'AD': ad_stat, 'p_value': ks_p})
            except:
                continue
        return res

    for col in cols:
        st.header(f"Analysis for '{col}'")
        data = df[col].dropna().values

        # Fit raw data
        st.subheader("Fits on Raw Data")
        raw_results = evaluate_fits(data)
        if raw_results:
            for r in raw_results:
                # Handle possible NaN AD values
                display_ad = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
                st.write(f"{r['name']}: AD={display_ad}, KS p-value={r['p_value']:.4f}")
            # Rank raw
            df_raw = pd.DataFrame(raw_results)
            df_raw['AD_sort'] = df_raw['AD'].fillna(df_raw['AD'].max()*10)
            best_raw = df_raw.sort_values(['AD_sort', 'p_value'], ascending=[True, False]).iloc[0](['AD_sort', 'p_value'], ascending=[True, False]).iloc[0]
            st.write(f"Best raw fit: **{best_raw['name']}** (AD={best_raw['AD']:.4f if not np.isnan(best_raw['AD']) else 'N/A'}, p-value={best_raw['p_value']:.4f})")
        else:
            st.warning("No valid raw fits")
            raw_results = []
            best_raw = None

        # Decide on final dataset (raw or transformed)
        if best_raw and best_raw['p_value'] >= 0.05:
            x = data
            transform = 'None'
            final_results = raw_results
            st.success("Accepting raw-data fit; no transformation needed.")
        else:
            # Transform data
            st.subheader("Data Transformation")
            try:
                lam = boxcox_normmax(data + 1e-8)
                x = boxcox(data + 1e-8, lam)
                transform = f"Box-Cox (λ={lam:.3f})"
            except:
                x = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
                transform = 'Yeo-Johnson'
            st.write(f"Applied transformation: {transform}")
            # Fit transformed
            st.subheader("Fits on Transformed Data")
            transformed_results = evaluate_fits(x)
            for r in transformed_results:
                display_ad = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
                st.write(f"{r['name']}: AD={display_ad}, KS p-value={r['p_value']:.4f}")
            df_tr = pd.DataFrame(transformed_results)(transformed_results)
            df_tr['AD_sort'] = df_tr['AD'].fillna(df_tr['AD'].max()*10)
            best_tr = df_tr.sort_values(['AD_sort', 'p_value'], ascending=[True, False]).iloc[0]
            best_raw = best_tr  # use for final
            final_results = transformed_results

        # Plot histogram
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        axs[0].hist(data, bins=30, color='skyblue', edgecolor='black'); axs[0].set_title('Original')
        axs[1].hist(x, bins=30, color='lightgreen', edgecolor='black'); axs[1].set_title(f'Transformed ({transform})')
        st.pyplot(fig)

        # Report best
        st.success(f"**Best Fit**: {best_raw['name']} (AD={best_raw['AD']:.4f if not np.isnan(best_raw['AD']) else 'N/A'}, p-value={best_raw['p_value']:.4f})")

        # Comparison chart
        df_final = pd.DataFrame(final_results)
        df_final['AD_sort'] = df_final['AD'].fillna(df_final['AD'].max()*10)
        fig2, ax2 = plt.subplots(figsize=(8,4))
        idx = np.arange(len(df_final))
        ax2.bar(idx-0.2, df_final['AD_sort'], 0.4, label='AD Stat')
        ax2.bar(idx+0.2, df_final['p_value'], 0.4, label='KS p-value')
        ax2.set_xticks(idx)
        ax2.set_xticklabels(df_final['name'], rotation=45, ha='right')
        ax2.set_title('Goodness-of-Fit Comparison')
        ax2.legend()
        st.pyplot(fig2)

        st.write(f"**Transformation used:** {transform}")
