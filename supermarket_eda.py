# Description: Interactive Streamlit dashboard for analyzing supermarket sales data with advanced features.
# python -m streamlit run app.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Supermarket Sales EDA", layout="wide")
st.title("🛍️ Supermarket Sales EDA Dashboard")

@st.cache_data
# This function handles loading the dataset from the user and ensures
# that required reference columns (like 'date' and 'total') are properly formatted for analysis.
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    return df

# This class handles all dashboard visualizations and user interactions for the EDA.
class Dashboard:
    def __init__(self, df):
        self.df = df

    def show_kpis(self):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${self.df['total'].sum():,.2f}")
        col2.metric("Average Rating", f"{self.df['rating'].mean():.2f}")
        col3.metric("Transactions", len(self.df))

    def sales_by_product(self):
        st.subheader("🧺 Product Line Sales")
        grouped = self.df.groupby('product_line')['total'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped.plot(kind='barh', color='#1f77b4', ax=ax, edgecolor='black')
        ax.set_xlabel("Total Sales")
        st.pyplot(fig)

        st.subheader("🧁 Product Line Sales Share")
        fig2, ax2 = plt.subplots()
        ax2.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

    def sales_by_city(self):
        st.subheader("🌆 Sales Distribution by City")
        grouped = self.df.groupby('city')['total'].sum()
        fig, ax = plt.subplots()
        ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

    def gender_comparison(self):
        st.subheader("🧍 Sales by Gender")
        data = self.df.groupby('gender')['total'].sum()
        fig, ax = plt.subplots()
        sns.barplot(x=data.index, y=data.values, palette="Set2", edgecolor="black", ax=ax)
        st.pyplot(fig)

    def customer_type_breakdown(self):
        st.subheader("👥 Customer Type")
        data = self.df.groupby('customer_type')['total'].sum()
        st.bar_chart(data)

    def payment_distribution(self):
        st.subheader("💳 Payment Methods")
        data = self.df['payment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    def correlation_matrix(self):
        st.subheader("🔥 Correlation Heatmap")
        corr = self.df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='gray', ax=ax)
        st.pyplot(fig)

    def monthly_sales_trend(self):
        st.subheader("📆 Monthly Sales Trend")
        if 'date' in self.df.columns:
            monthly = self.df.resample('M', on='date')['total'].sum().reset_index()
            fig, ax = plt.subplots()
            ax.plot(monthly['date'], monthly['total'], marker='o', linestyle='--', linewidth=2, color='#FF5733')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.set_title("Monthly Sales Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Sales")
            fig.autofmt_xdate()
            st.pyplot(fig)

    def quick_insights(self):
        st.subheader("💡 Quick Insights")
        top_city = self.df.groupby('city')['total'].sum().idxmax()
        top_payment = self.df['payment'].value_counts().idxmax()
        top_product = self.df.groupby('product_line')['total'].sum().idxmax()
        top_product_sales = self.df.groupby('product_line')['total'].sum().max()
        latest = self.df.sort_values('date', ascending=False).head(30)
        last_30_total = latest['total'].sum()
        st.info(f"🏙️ Highest Sales City: {top_city}")
        st.info(f"💳 Most Used Payment Method: {top_payment}")
        st.info(f"📦 Highest Selling Product: {top_product.title()} (${top_product_sales:,.2f})")
        st.info(f"📈 Sales in Last 30 Records: ${last_30_total:,.2f}")

    def about_section(self):
        with st.expander("ℹ️ About this Dashboard"):
            st.markdown("""
            This interactive dashboard allows you to explore supermarket sales data with breakdowns by city, gender, product lines, and more. 
            It also includes forecasting features and downloadable reports to support business insights.
            """)

    def search_city_and_product(self):
        st.subheader("🔍 Search City or Product")
        city = st.text_input("Enter city name (e.g., Yangon):").title()
        product = st.text_input("Enter product line name (e.g., Health and beauty):").title()
        if city:
            if city in self.df['city'].str.title().unique():
                city_df = self.df[self.df['city'].str.title() == city]
                st.success(f"Total sales in {city}: ${city_df['total'].sum():,.2f}")
            else:
                st.warning("City not found.")
        if product:
            if product in self.df['product_line'].str.title().unique():
                product_df = self.df[self.df['product_line'].str.title() == product]
                st.success(f"Total sales for '{product}': ${product_df['total'].sum():,.2f}")
            else:
                st.warning("Product not found.")

    def download_data_button(self):
        st.subheader("⬇️ Download Filtered Data")
        csv = self.df.to_csv(index=False)
        st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

def main():
    df = load_data("supermarket_sales.csv")
    st.sidebar.title("Filters")

    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    city_filter = st.sidebar.multiselect("City", df['city'].unique(), default=df['city'].unique())
    product_filter = st.sidebar.multiselect("Product Line", df['product_line'].unique(), default=df['product_line'].unique())
    gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())

    filtered = df[
        df['city'].isin(city_filter) &
        df['product_line'].isin(product_filter) &
        df['gender'].isin(gender_filter)
    ]

    st.subheader("📄 Preview of Filtered Data")
    st.dataframe(filtered.head())
    if 'date' in filtered.columns:
        st.write(f"📅 Data from {filtered['date'].min().date()} to {filtered['date'].max().date()}")

    dash = Dashboard(filtered)
    dash.about_section()
    dash.show_kpis()
    dash.sales_by_product()
    dash.sales_by_city()
    dash.gender_comparison()
    dash.customer_type_breakdown()
    dash.payment_distribution()
    dash.correlation_matrix()
    dash.monthly_sales_trend()
    dash.quick_insights()
    dash.search_city_and_product()
    dash.download_data_button()
    st.success("✅ Dashboard generated successfully!")

if __name__ == '__main__':
    main()
