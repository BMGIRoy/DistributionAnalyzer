# Full integrated Streamlit app with AIC, BIC, CI, and visualization
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox_normmax, boxcox, anderson, kstest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import base64
import os
from datetime import datetime

st.title("Advanced Distribution Analyzer & Transformer")

uploaded_file = st.file_uploader("Upload your CSV file with numeric columns:", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect("Select column(s) to analyze", numeric_cols, default=numeric_cols[:1])

    report = []
    plot_images = []
    transformation_plots = []

    for column in selected_cols:
        st.header(f"Analysis for: {column}")
        values = data[column].dropna().values

        stat, p_value = stats.shapiro(values)
        st.write(f"Shapiro-Wilk Test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")

        if p_value > 0.05:
            st.success("Data appears to be normally distributed.")
            transformed_data = values
            transformation_used = "None (Normal)"
        else:
            st.warning("Data is not normally distributed. Trying Box-Cox transformation.")
            try:
                lmbda = boxcox_normmax(values + 1e-4)
                transformed_data = boxcox(values + 1e-4, lmbda=lmbda)
                transformation_used = f"Box-Cox (lambda={lmbda:.4f})"
            except:
                st.warning("Box-Cox failed. Using Johnson (Yeo-Johnson) transformation.")
                pt = PowerTransformer(method='yeo-johnson')
                transformed_data = pt.fit_transform(values.reshape(-1, 1)).flatten()
                transformation_used = "Johnson (Yeo-Johnson)"

        fig2, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(values, bins=30, color='skyblue', edgecolor='black')
        axs[0].set_title("Original Data")
        axs[1].hist(transformed_data, bins=30, color='lightgreen', edgecolor='black')
        axs[1].set_title("Transformed Data")
        st.pyplot(fig2)

        trans_buf = io.BytesIO()
        fig2.savefig(trans_buf, format="png")
        trans_buf.seek(0)
        transformation_plots.append((column, trans_buf.read()))

        distributions = ['norm', 'lognorm', 'expon', 'weibull_min', 'gamma']
        best_fit = None
        best_loglik = -np.inf
        aic_bic_comparison = []

        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(transformed_data)
                loglik = np.sum(dist.logpdf(transformed_data, *params))
                k = len(params)
                n = len(transformed_data)
                aic = 2 * k - 2 * loglik
                bic = k * np.log(n) - 2 * loglik
                aic_bic_comparison.append((dist_name, aic, bic))
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_fit = (dist_name, params)
            except:
                continue

        dist_name, params = best_fit
        st.success(f"Best fit: {dist_name} with parameters: {np.round(params, 4)}")

        dist = getattr(stats, dist_name)
        ks_stat, ks_p = kstest(transformed_data, dist_name, args=params)
        ad_result = anderson(transformed_data, dist='norm')
        st.write(f"Kolmogorov-Smirnov Test: Statistic = {ks_stat:.4f}, p-value = {ks_p:.4f}")
        st.write(f"Anderson-Darling Test: Statistic = {ad_result.statistic:.4f}, Critical Values = {ad_result.critical_values}, Significance Levels = {ad_result.significance_level}")

        predicted = dist.rvs(*params, size=3)
        st.write(f"Predicted Values: {np.round(predicted, 4)}")

        st.subheader("Confidence Intervals (Approximate)")
        alpha = 0.05
        ci_low = dist.ppf(alpha / 2, *params)
        ci_high = dist.ppf(1 - alpha / 2, *params)
        st.write(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
        confidence_interval = [round(ci_low, 4), round(ci_high, 4)]

        st.subheader("AIC/BIC Comparison Across Distributions")
        labels, aics, bics = zip(*aic_bic_comparison)
        x = np.arange(len(labels))
        width = 0.35
        fig3, ax3 = plt.subplots()
        ax3.bar(x - width/2, aics, width, label='AIC')
        ax3.bar(x + width/2, bics, width, label='BIC')
        ax3.set_ylabel('Score')
        ax3.set_title('AIC & BIC by Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()
        st.pyplot(fig3)

        fig, ax = plt.subplots()
        ax.hist(transformed_data, bins=30, density=True, alpha=0.6, label="Transformed Data")
        x = np.linspace(min(transformed_data), max(transformed_data), 1000)
        pdf = dist.pdf(x, *params)
        ax.plot(x, pdf, 'r-', label=f"{dist_name} PDF")
        ax.legend()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{encoded}" download="{column}_plot.png">Download {column} plot</a>'
        st.markdown(href, unsafe_allow_html=True)

        buf.seek(0)
        plot_images.append((column, buf.read()))

        report.append({
            "Column": column,
            "Normality p-value": round(p_value, 4),
            "Transformation": transformation_used,
            "Best Fit": dist_name,
            "Parameters": np.round(params, 4).tolist(),
            "KS Statistic": round(ks_stat, 4),
            "KS p-value": round(ks_p, 4),
            "AD Statistic": round(ad_result.statistic, 4),
            "Predicted Values": np.round(predicted, 4).tolist(),
            "95% CI": confidence_interval,
            "AIC": round(aic, 2),
            "BIC": round(bic, 2)
        })

    report_df = pd.DataFrame(report)
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="Download Summary as CSV",
        data=csv,
        file_name='distribution_report.csv',
        mime='text/csv'
    )

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Distribution Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)

        for entry, (colname, img_bytes), (_, trans_bytes) in zip(report, plot_images, transformation_plots):
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Analysis Report - {colname}", ln=True, align='C')
            for key, value in entry.items():
                pdf.multi_cell(0, 10, f"{key}: {value}")
            trans_path = f"{colname}_transform.png"
            with open(trans_path, "wb") as f:
                f.write(trans_bytes)
            pdf.image(trans_path, w=170)
            os.remove(trans_path)
            image_path = f"{colname}_plot.png"
            with open(image_path, "wb") as f:
                f.write(img_bytes)
            pdf.image(image_path, w=170)
            os.remove(image_path)

        pdf_output = "distribution_analysis_report.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{pdf_output}">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
