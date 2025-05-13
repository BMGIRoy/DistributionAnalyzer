# Streamlit app: Automatic Distribution Fit via AD & p-value with conditional transformation
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox_normmax, boxcox, anderson, kstest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

st.title("Distribution Fit Identifier using AD and p-value")

# Step 1: Upload data
uploaded_file = st.file_uploader("Upload CSV with numeric columns:", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = st.multiselect("Select column for analysis", numeric_cols, default=numeric_cols[:1])

    for col in cols:
        st.header(f"Analysis for '{col}'")
        data = df[col].dropna().values

        # Function to run AD and KS tests
        def evaluate(x):
            results = []
            for dist_name in ['norm','expon','logistic','weibull_min','gamma','lognorm']:
                try:
                    ad_stat = anderson(x, dist=dist_name).statistic
                    ks_stat, ks_p = kstest(x, dist_name, args=getattr(stats, dist_name).fit(x))
                    results.append({'dist': dist_name, 'AD': ad_stat, 'p_value': ks_p})
                except Exception:
                    continue
            return results

        # Evaluate on raw data
        st.subheader("Goodness-of-Fit on Raw Data")
        raw_results = evaluate(data)
        if raw_results:
            for r in raw_results:
                st.write(f"{r['dist']}: AD={r['AD']:.4f}, KS p-value={r['p_value']:.4f}")
            best_raw = sorted(raw_results, key=lambda r: (r['AD'], -r['p_value']))[0]
            st.write(f"Best on raw data: **{best_raw['dist']}** (AD={best_raw['AD']:.4f}, p-value={best_raw['p_value']:.4f})")
        else:
            st.warning("No distribution tests passed on raw data.")
            best_raw = None

        # Decide transformation
        if best_raw and best_raw['dist']=='norm':
            st.success("Normal distribution is best fit on raw data; no transform needed.")
            x = data
            transform = 'None'
            final_results = raw_results
            best_fit = best_raw
        else:
            st.warning("Normal not best on raw data; applying transformation.")
            try:
                lam = boxcox_normmax(data + 1e-8)
                x = boxcox(data + 1e-8, lam)
                transform = f"Box-Cox (λ={lam:.3f})"
            except Exception:
                x = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
                transform = 'Yeo-Johnson'
            st.write(f"Transformation: {transform}")
            # Evaluate on transformed data
            st.subheader("Goodness-of-Fit on Transformed Data")
            transformed_results = evaluate(x)
            for r in transformed_results:
                st.write(f"{r['dist']}: AD={r['AD']:.4f}, KS p-value={r['p_value']:.4f}")
            best_fit = sorted(transformed_results, key=lambda r: (r['AD'], -r['p_value']))[0]

        # Display histograms
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].hist(data, bins=30, color='skyblue', edgecolor='black')
        ax[0].set_title("Original Data")
        ax[1].hist(x, bins=30, color='lightgreen', edgecolor='black')
        ax[1].set_title(f"Transformed Data ({transform})")
        st.pyplot(fig)

        # Show final best fit
        st.success(f"**Final Best Fit**: {best_fit['dist']} (AD={best_fit['AD']:.4f}, p-value={best_fit['p_value']:.4f})")

        # Bar chart comparison of AD and p-value
        dists = [r['dist'] for r in (final_results if best_raw and best_raw['dist']=='norm' else transformed_results)]
        ad_vals = [r['AD'] for r in (final_results if best_raw and best_raw['dist']=='norm' else transformed_results)]
        pv_vals = [r['p_value'] for r in (final_results if best_raw and best_raw['dist']=='norm' else transformed_results)]
        idx = np.arange(len(dists))
        width=0.35
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.bar(idx-width/2, ad_vals, width, label='AD Stat')
        ax2.bar(idx+width/2, pv_vals, width, label='KS p-value')
        ax2.set_xticks(idx)
        ax2.set_xticklabels(dists)
        ax2.set_ylabel('Score')
        ax2.set_title('Distribution Fit Comparison')
        ax2.legend()
        st.pyplot(fig2)

        # Show transformation used
        st.info(f"Transformation applied: {transform}")
        # Clear for next loop
        final_results = None
