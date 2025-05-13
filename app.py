# Streamlit app: Distribution Fit & Goodness-of-Fit Tests
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
if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Data Preview")
st.dataframe(df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the uploaded file.")
    st.stop()

cols = st.multiselect(
    "Select column(s) for analysis", numeric_cols, default=[numeric_cols[0]]
)

# Define distributions to test
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

# Helper function: evaluate fits
def evaluate_fits(x):
    results = []
    for name, alias, fit_kwargs in dist_info:
        try:
            dist = getattr(stats, alias)
            params = dist.fit(x, **fit_kwargs)
            # Anderson-Darling
            try:
                ad_stat = anderson(x, dist=alias).statistic
            except Exception:
                ad_stat = np.nan
            # KS test
            ks_stat, ks_p = kstest(x, alias, args=params)
            results.append({
                'name': name,
                'alias': alias,
                'params': params,
                'AD': ad_stat,
                'p_value': ks_p
            })
        except Exception:
            continue
    return results

for col in cols:
    st.header(f"Analysis for '{col}'")
    data = df[col].dropna().values

    # Raw data fit
    st.subheader("Fits on Raw Data")
    raw_results = evaluate_fits(data)
    if raw_results:
        df_raw = pd.DataFrame(raw_results)
        # display each
        for r in raw_results:
            ad_disp = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
            st.write(f"{r['name']}: AD={ad_disp}, KS p-value={r['p_value']:.4f}")
        # choose best raw
        df_raw['AD_sort'] = df_raw['AD'].fillna(df_raw['AD'].max()*10)
        best_raw = df_raw.sort_values(['AD_sort','p_value'], ascending=[True,False]).iloc[0]
        raw_ad = f"{best_raw['AD']:.4f}" if not np.isnan(best_raw['AD']) else "N/A"
        raw_p = best_raw['p_value']
        st.write(f"**Best raw fit**: {best_raw['name']} (AD={raw_ad}, p-value={raw_p:.4f})")
    else:
        st.warning("No raw-data fits succeeded.")
        best_raw = None

    # Decide transformation
    if best_raw is not None and best_raw['p_value'] >= 0.05:
        x = data
        transform = 'None'
        final_results = raw_results
        best_fit = best_raw
        st.success("Raw data fit accepted; no transformation applied.")
    else:
        # apply transform
        st.subheader("Data Transformation")
        try:
            lam = boxcox_normmax(data + 1e-8)
            x = boxcox(data + 1e-8, lam)
            transform = f"Box-Cox (λ={lam:.3f})"
        except Exception:
            x = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
            transform = 'Yeo-Johnson'
        st.write(f"Transformation applied: {transform}")
        # fit transformed
        st.subheader("Fits on Transformed Data")
        transformed_results = evaluate_fits(x)
        df_tr = pd.DataFrame(transformed_results)
        for r in transformed_results:
            ad_disp = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
            st.write(f"{r['name']}: AD={ad_disp}, KS p-value={r['p_value']:.4f}")
        df_tr['AD_sort'] = df_tr['AD'].fillna(df_tr['AD'].max()*10)
        best_tr = df_tr.sort_values(['AD_sort','p_value'], ascending=[True,False]).iloc[0]
        best_fit = best_tr
        final_results = transformed_results

    # Plot histograms
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].hist(data, bins=30, color='skyblue', edgecolor='black'); axs[0].set_title('Original Data')
    axs[1].hist(x, bins=30, color='lightgreen', edgecolor='black'); axs[1].set_title(f'Transformed ({transform})')
    st.pyplot(fig)

    # Final best fit display
    ad_disp = f"{best_fit['AD']:.4f}" if not np.isnan(best_fit['AD']) else "N/A"
    p_disp = best_fit['p_value']
    st.success(f"**Final Best Fit**: {best_fit['name']} (AD={ad_disp}, p-value={p_disp:.4f})")

    # Comparison chart
    df_fin = pd.DataFrame(final_results)
    df_fin['AD_sort'] = df_fin['AD'].fillna(df_fin['AD'].max()*10)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    idx = np.arange(len(df_fin))
    ax2.bar(idx-0.2, df_fin['AD_sort'], 0.4, label='AD Stat')
    ax2.bar(idx+0.2, df_fin['p_value'], 0.4, label='KS p-value')
    ax2.set_xticks(idx)
    ax2.set_xticklabels(df_fin['name'], rotation=45, ha='right')
    ax2.set_title('Goodness-of-Fit Comparison')
    ax2.legend()
    st.pyplot(fig2)

    st.write(f"**Transformation used:** {transform}")
