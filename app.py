# app.py
# Streamlit app: Unified Distribution Fit Analyzer
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
y_raw = data_df[col].dropna().values

# --- Define Distributions ---
dist_info = [
    ('Normal', 'norm', {}),
    ('Uniform', 'uniform', {}),
    ('Gamma (2P)', 'gamma', {'floc': 0}),
    ('Gamma (3P)', 'gamma', {}),
    ('Exponential', 'expon', {}),
    ('Lognormal', 'lognorm', {}),
    ('Weibull (2P)', 'weibull_min', {'floc': 0}),
    ('Weibull (3P)', 'weibull_min', {}),
    ('Logistic', 'logistic', {}),
    ('Log-Logistic', 'fisk', {}),
    ('Smallest EV', 'gumbel_r', {})
]

# --- Prepare Transformations ---
# Raw series
y_raw = series
# Box-Cox (positive data)
try:
    lam = boxcox_normmax(y_raw + 1e-8)
    y_box = boxcox(y_raw + 1e-8, lam)
    box_label = f"Box-Cox (λ={lam:.3f})"
except Exception:
    y_box = None
    box_label = None
# Yeo-Johnson

y_john = PowerTransformer(method='yeo-johnson').fit_transform(y_raw.reshape(-1,1)).flatten()
john_label = "Yeo-Johnson"

# Map transforms for easy lookup
transform_map = {'Raw': y_raw}
if y_box is not None:
    transform_map[box_label] = y_box
transform_map[john_label] = y_john

# --- Fit & Evaluate Function ---
def evaluate_fits(y, transform_label):
    results = []
    for name, alias, kwargs in dist_info:
        try:
            dist = getattr(stats, alias)
            params = dist.fit(y, **kwargs)
            # Anderson-Darling
            if alias == 'uniform':
                loc, scale = params[0], params[1]
                u = np.sort((y - loc) / scale)
                n = len(u)
                u = np.clip(u, 1e-10, 1-1e-10)
                i = np.arange(1, n+1)
                ad_stat = -n - np.sum((2*i-1)*(np.log(u)+np.log(1-u[::-1])))/n
            else:
                try:
                    ad_stat = anderson(y, dist=alias).statistic
                except Exception:
                    ad_stat = np.nan
            # KS test
            ks_stat, ks_p = kstest(y, alias, args=params)
            results.append({
                'Distribution': name,
                'Transform': transform_label,
                'Alias': alias,
                'Params': params,
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
st.dataframe(res_df[['Distribution','Transform','AD_stat','KS_p']])

# --- Select Best Fit ---
max_ad = res_df['AD_stat'].max(skipna=True)
res_df['AD_sort'] = res_df['AD_stat'].fillna(max_ad*10)
best = res_df.sort_values(['AD_sort','KS_p'], ascending=[True,False]).iloc[0]

st.success(f"🏆 Best Fit: {best['Distribution']} after {best['Transform']} (AD={best['AD_stat']:.4f}, p-value={best['KS_p']:.4f})")

# --- Side-by-Side Raw vs Transformed with Best-Fit PDF and 95% CI ---
st.subheader("Raw vs Transformed with Best-Fit PDF & 95% CI")
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
# Left: raw data with best-fit PDF & CI if raw transform
raw_subset = res_df[(res_df.Transform == 'Raw')]
# find best raw row
raw_max_ad = raw_subset.AD_stat.max(skipna=True)
best_raw = raw_subset.assign(AD_sort=raw_subset.AD_stat.fillna(raw_max_ad*10))\
    .sort_values(['AD_sort','KS_p'], ascending=[True,False]).iloc[0]
# raw PDF overlay
y1 = y_raw
dist1 = getattr(stats, best_raw.Alias)
params1 = best_raw.Params
ax1.hist(y1, bins=30, density=True, alpha=0.6)
x1 = np.linspace(y1.min(), y1.max(), 200)
ax1.plot(x1, dist1.pdf(x1, *params1), 'r-', label=f"{best_raw.Distribution} PDF")
try:
    ci1 = dist1.ppf([0.025,0.975], *params1)
    ax1.axvline(ci1[0], linestyle='--', label='95% CI')
    ax1.axvline(ci1[1], linestyle='--')
except Exception:
    pass
ax1.set_title('Raw Data')
ax1.legend()
# Right: transformed data with best-fit PDF & CI
y2 = transform_map.get(best['Transform'], y_raw)
dist2 = getattr(stats, best.Alias)
params2 = best.Params
ax2.hist(y2, bins=30, density=True, alpha=0.6)
x2 = np.linspace(y2.min(), y2.max(), 200)
ax2.plot(x2, dist2.pdf(x2, *params2), 'r-', label=f"{best.Distribution} PDF")
try:
    ci2 = dist2.ppf([0.025,0.975], *params2)
    ax2.axvline(ci2[0], linestyle='--', label='95% CI')
    ax2.axvline(ci2[1], linestyle='--')
except Exception:
    pass
ax2.set_title(f"{best.Distribution} after {best.Transform}")
ax2.legend()
st.pyplot(fig)

# --- Predicted Future Values ---
st.subheader("Predicted Next 5 Values")
# generate predictions using best fit
dist_best = getattr(stats, best.Alias)
params_best = best.Params
preds = dist_best.rvs(*params_best, size=5)
st.write([round(float(p),4) for p in preds])

# --- Final Histogram ---
st.subheader("Final Data Histogram")
fig2, ax3 = plt.subplots(figsize=(6,3))
ax3.hist(y2, bins=30, density=True, alpha=0.6, edgecolor='black')
ax3.set_title(f"{best['Distribution']} after {best['Transform']}")
st.pyplot(fig2)
