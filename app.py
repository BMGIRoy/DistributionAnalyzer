# app.py
# Streamlit app: Unified Distribution Fit Table with AD & KS Tests
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox_normmax, boxcox, anderson, kstest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Page layout
st.set_page_config(page_title="Distribution Fit Analyzer", layout="wide")
st.title("📊 Distribution Fit Analyzer with AD & KS Tests")

# --- Upload Data ---
uploader = st.file_uploader("Upload CSV with numeric columns:", type=["csv"])
if not uploader:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

data_df = pd.read_csv(uploader)
st.subheader("Data Preview")
st.dataframe(data_df.head())

# --- Select Column ---
numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the uploaded file.")
    st.stop()
col = st.selectbox("Select column for analysis:", numeric_cols)
series = data_df[col].dropna().values

# --- Define Distributions ---
dist_info = [
    ('Normal', 'norm', {}),
    ('Uniform', 'uniform', {}),
    ('Gamma (2P)', 'gamma', {'floc': 0}),
    ('Gamma (3P)', 'gamma', {}),
    ('Exponential', 'expon', {}),
    ('Lognormal', 'logN', {}),
    ('Weibull (2P)', 'weibull_min', {'floc': 0}),
    ('Weibull (3P)', 'weibull_min', {}),
    ('Logistic', 'logistic', {}),
    ('Log-Logistic', 'fisk', {}),
    ('Smallest EV', 'gumbel_r', {})
]

# --- Prepare Transformations ---
y_raw = series

# Box-Cox (positive-only)
try:
    lam = boxcox_normmax(y_raw + 1e-8)
    y_box = boxcox(y_raw + 1e-8, lam)
    box_label = f"Box-Cox (λ={lam:.3f})"
except Exception:
    y_box = None
    box_label = None

# Yeo-Johnson (handles zero/negatives)
y_john = PowerTransformer(method='yeo-johnson').fit_transform(y_raw.reshape(-1,1)).flatten()
john_label = "Yeo-Johnson"

# --- Fit & Evaluate Function ---
def evaluate_fits(y, transform_label):
    results = []
    for name, alias, kwargs in dist_info:
        try:
            dist = getattr(stats, alias)
            params = dist.fit(y, **kwargs)
            # Anderson-Darling for uniform by manual calculation
            if alias == 'uniform':
                loc, scale = params[0], params[1]
                # transform to [0,1]
                u = np.sort((y - loc) / scale)
                n = len(u)
                # clip to avoid log(0)
                u = np.clip(u, 1e-10, 1 - 1e-10)
                i = np.arange(1, n+1)
                ad_stat = -n - np.sum((2*i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n
            else:
                # Anderson-Darling for supported distributions
                try:
                    ad_stat = anderson(y, dist=alias).statistic
                except Exception:
                    ad_stat = np.nan
            # KS test
            ks_stat, ks_p = kstest(y, alias, args=params)
            results.append({
                'Distribution': name,
                'Transform': transform_label,
                'AD_stat': ad_stat,
                'KS_p': ks_p
            })
        except Exception:
            continue
    return results

# --- Gather Results ---
all_results = []
all_results += evaluate_fits(y_raw, 'Raw')
if y_box is not None:
    all_results += evaluate_fits(y_box, box_label)
all_results += evaluate_fits(y_john, john_label)

# --- Display Summary Table ---
res_df = pd.DataFrame(all_results)
st.subheader("Goodness-of-Fit Summary")
st.dataframe(res_df)

# --- Select Best Fit ---
max_ad = res_df['AD_stat'].max(skipna=True)
res_df['AD_sort'] = res_df['AD_stat'].fillna(max_ad * 10)
best = res_df.sort_values(['AD_sort', 'KS_p'], ascending=[True, False]).iloc[0]

st.success(
    f"🏆 Best Fit: {best['Distribution']} with {best['Transform']} "
    f"(AD={best['AD_stat']:.4f}, KS p-value={best['KS_p']:.4f})"
)

# --- Visualize AD vs KS p-value ---
fig, ax = plt.subplots(figsize=(8, 4))
for _, row in res_df.iterrows():
    ax.scatter(row['AD_stat'], row['KS_p'], label=f"{row['Distribution']}\n{row['Transform']}")
ax.set_xlabel('Anderson-Darling Statistic')
ax.set_ylabel('KS Test p-value')
ax.set_title('Fit Comparison')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
st.pyplot(fig)

# --- Final Histogram ---
transform_map = {'Raw': y_raw, box_label: y_box, john_label: y_john}
y_final = transform_map.get(best['Transform'], y_raw)
st.subheader("Histogram of Final Selected Data")
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.hist(y_final, bins=30, density=True, alpha=0.6, color='teal', edgecolor='black')
ax2.set_title(f"{best['Distribution']} after {best['Transform']}")
st.pyplot(fig2)
