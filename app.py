# Streamlit app: Automatic Distribution Fit via AD & p-value
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

        # Test normality
        sw_stat, sw_p = stats.shapiro(data)
        st.write(f"**Shapiro–Wilk**: statistic={sw_stat:.4f}, p-value={sw_p:.4f}")

        # Transform if not normal
        if sw_p >= 0.05:
            st.success("Data is normal; no transform applied.")
            x = data
            transform = "None"
        else:
            st.warning("Data not normal; applying Box-Cox or Yeo-Johnson.")
            try:
                lam = boxcox_normmax(data + 1e-8)
                x = boxcox(data + 1e-8, lam)
                transform = f"Box-Cox (λ={lam:.3f})"
            except Exception:
                x = PowerTransformer(method='yeo-johnson').fit_transform(data.reshape(-1,1)).flatten()
                transform = "Yeo-Johnson"
            st.write(f"Transformation: {transform}")

        # Show histograms
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].hist(data, bins=30, color='skyblue', edgecolor='black')
        ax[0].set_title("Original")
        ax[1].hist(x, bins=30, color='lightgreen', edgecolor='black')
        ax[1].set_title("Transformed")
        st.pyplot(fig)

        # Goodness-of-fit via AD and KS
        st.subheader("Goodness-of-Fit Tests")
        dists = ['norm','expon','logistic','gumbel','extreme1']
        results = []
        for dist_name in dists:
            try:
                # Anderson-Darling
                ad = anderson(x, dist=dist_name).statistic
                # Kolmogorov-Smirnov for p-value
                ks_stat, ks_p = kstest(x, dist_name, args=getattr(stats,dist_name).fit(x))
                results.append({'dist':dist_name,'AD':ad,'p_value':ks_p})
                st.write(f"{dist_name}: AD={ad:.4f}, KS p-value={ks_p:.4f}")
            except Exception as e:
                st.write(f"{dist_name}: test failed ({e})")

        # Select best distribution: minimal AD, then maximal p-value
        if results:
            # sort by AD asc, p_value desc
            results_sorted = sorted(results, key=lambda r: (r['AD'], -r['p_value']))
            best = results_sorted[0]
            st.success(f"**Best Fit**: {best['dist']} (AD={best['AD']:.4f}, p-value={best['p_value']:.4f})")

            # Bar chart comparison
            ad_vals = [r['AD'] for r in results]
            pv_vals = [r['p_value'] for r in results]
            labels = [r['dist'] for r in results]
            x_idx = np.arange(len(labels))
            width=0.35
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.bar(x_idx-width/2, ad_vals, width, label='AD Stat')
            ax2.bar(x_idx+width/2, pv_vals, width, label='KS p-value')
            ax2.set_xticks(x_idx)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Score')
            ax2.set_title('AD vs p-value by Distribution')
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("No distributions could be tested.")
