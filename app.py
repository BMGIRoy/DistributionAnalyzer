# Streamlit app: Distribution Fit & Goodness-of-Fit Tests with Transform Comparison
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox_normmax, boxcox, anderson, kstest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

st.title("Distribution Fit & Goodness-of-Fit Tests with Transform Comparison")

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
    st.error("No numeric columns found.")
    st.stop()

cols = st.multiselect(
    "Select column(s) for analysis", numeric_cols, default=[numeric_cols[0]]
)

# Define distributions to test, including Uniform
dist_info = [
    ('Normal', 'norm', {}),
    ('Uniform', 'uniform', {}),
    ('Gamma (2P)', 'gamma', {'floc': 0}),
    ('Gamma (3P)', 'gamma', {}),
    ('Exponential', 'expon', {}),
    ('Weibull (2P)', 'weibull_min', {'floc': 0}),
    ('Weibull (3P)', 'weibull_min', {}),
    ('Logistic', 'logistic', {}),
    ('Log-Logistic', 'fisk', {}),
    ('Smallest Extreme Value', 'gumbel_r', {})
]

# Helper: evaluate fits for a given array x
def evaluate_fits(x):
    results = []
    for name, alias, fit_kwargs in dist_info:
        try:
            dist = getattr(stats, alias)
            params = dist.fit(x, **fit_kwargs)
            # Anderson-Darling (if supported)
            try:
                ad_stat = anderson(x, dist=alias).statistic
            except Exception:
                ad_stat = np.nan
            # KS test
            ks_stat, ks_p = kstest(x, alias, args=params)
            results.append({'name': name, 'alias': alias, 'params': params,
                            'AD': ad_stat, 'p_value': ks_p})
        except Exception:
            continue
    return results

for col in cols:
    st.header(f"Analysis for '{col}'")
    data = df[col].dropna().values

    # Raw data fits
    st.subheader("Fits on Raw Data")
    raw_results = evaluate_fits(data)
    for r in raw_results:
        ad_disp = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
        st.write(f"{r['name']}: AD={ad_disp}, KS p-value={r['p_value']:.4f}")
    if raw_results:
        df_raw = pd.DataFrame(raw_results)
        df_raw['AD_sort'] = df_raw['AD'].fillna(df_raw['AD'].max()*10)
        best_raw = df_raw.sort_values(['AD_sort','p_value'], ascending=[True,False]).iloc[0]
        ad_raw = f"{best_raw['AD']:.4f}" if not np.isnan(best_raw['AD']) else "N/A"
        st.write(f"**Best raw fit**: {best_raw['name']} (AD={ad_raw}, p-value={best_raw['p_value']:.4f})")
    else:
        st.warning("No valid fits on raw data.")

    # Box-Cox transform
    st.subheader("Fits on Box-Cox Transformed Data")
    try:
        lam = boxcox_normmax(data + 1e-8)
        x_box = boxcox(data + 1e-8, lam)
        st.write(f"Box-Cox λ = {lam:.4f}")
    except Exception as e:
        st.write(f"Box-Cox transform failed: {e}")
        x_box = data
    box_results = evaluate_fits(x_box)
    for r in box_results:
        ad_disp = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
        st.write(f"{r['name']}: AD={ad_disp}, KS p-value={r['p_value']:.4f}")
    if box_results:
        df_box = pd.DataFrame(box_results)
        df_box['AD_sort'] = df_box['AD'].fillna(df_box['AD'].max()*10)
        best_box = df_box.sort_values(['AD_sort','p_value'], ascending=[True,False]).iloc[0]
        ad_box = f"{best_box['AD']:.4f}" if not np.isnan(best_box['AD']) else "N/A"
        st.write(f"**Best Box-Cox fit**: {best_box['name']} (AD={ad_box}, p-value={best_box['p_value']:.4f})")

    # Johnson (Yeo-Johnson) transform
    st.subheader("Fits on Yeo-Johnson Transformed Data")
    x_john = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
    john_results = evaluate_fits(x_john)
    for r in john_results:
        ad_disp = f"{r['AD']:.4f}" if not np.isnan(r['AD']) else "N/A"
        st.write(f"{r['name']}: AD={ad_disp}, KS p-value={r['p_value']:.4f}")
    if john_results:
        df_john = pd.DataFrame(john_results)
        df_john['AD_sort'] = df_john['AD'].fillna(df_john['AD'].max()*10)
        best_john = df_john.sort_values(['AD_sort','p_value'], ascending=[True,False]).iloc[0]
        ad_john = f"{best_john['AD']:.4f}" if not np.isnan(best_john['AD']) else "N/A"
        st.write(f"**Best Johnson fit**: {best_john['name']} (AD={ad_john}, p-value={best_john['p_value']:.4f})")

    # Summary chart for raw, boxcox, johnson bests
    summary = []
    if raw_results:
        summary.append(('Raw', best_raw['AD'], best_raw['p_value']))
    if box_results:
        summary.append(('Box-Cox', best_box['AD'], best_box['p_value']))
    if john_results:
        summary.append(('YJ', best_john['AD'], best_john['p_value']))
    if summary:
        labels, ad_vals, pv_vals = zip(*summary)
        idx = np.arange(len(labels))
        width = 0.3
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.bar(idx-width, ad_vals, width, label='AD Stat')
        ax2.bar(idx, pv_vals, width, label='KS p-value')
        ax2.set_xticks(idx)
        ax2.set_xticklabels(labels)
        ax2.legend()
        st.pyplot(fig2)

    # Show histogram of final transformed data (raw if normal else best)
    st.subheader("Histogram of Final Selected Data")
    final_data = data if (raw_results and best_raw['p_value']>=0.05) else (
        x_box if (box_results and best_box['p_value']>=best_john['p_value']) else x_john
    )
    fig, ax = plt.subplots()
    ax.hist(final_data, bins=30, density=True, alpha=0.6)
    ax.set_title('Final Data Histogram')
    st.pyplot(fig)
