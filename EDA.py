#EDA


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 1: LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv('hotel_bookings_cleaned.csv')

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# STEP 2.1: CANCELLATION DISTRIBUTION
# PIE CHART
# ==========================================

plt.figure(figsize=(7, 7))

df['is_canceled'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Not Cancelled', 'Cancelled'],
    wedgeprops={'edgecolor':'black','linewidth':1},
    colors=['#36A2EB','#FF6384']
)

plt.title('Hotel Booking Cancellation Distribution')
plt.ylabel('')

plt.show()


# ==========================================
# STEP 2.2: HOTEL TYPE ANALYSIS
# BAR GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

df['hotel'].value_counts().plot(kind='bar',color=['#1F4E79','#20B2AA'])

plt.title('Bookings by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Number of Bookings')

plt.show()


# ==========================================
# STEP 2.3: MONTHLY BOOKINGS
# LINE GRAPH
# ==========================================

monthly_bookings = df.groupby(
    'arrival_date_month'
).size()

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

monthly_bookings = monthly_bookings.reindex(month_order)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_bookings.index,
    monthly_bookings.values,
    marker='o',
    color='red'
)

plt.title('Monthly Hotel Bookings')
plt.xlabel('Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)

plt.show()


# ==========================================
# STEP 2.4: LEAD TIME DISTRIBUTION
# HISTOGRAM
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x='lead_time',
    bins=30,
    kde=True,
    color='slateblue',
    edgecolor='black'
)

plt.title('Distribution of Booking Lead Time')
plt.xlabel('Lead Time (Days)')
plt.ylabel('Frequency')

plt.show()


# ==========================================
# STEP 2.5: CUSTOMER TYPE VS CANCELLATION
# BOX PLOT
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x='customer_type',
    y='lead_time',
    hue='is_canceled',
    palette=['#B5E2FA','#ED254E']
)

plt.title('Customer Type and Lead Time vs Cancellation')
plt.xlabel('Customer Type')
plt.ylabel('Lead Time (Days)')

plt.show()


# ==========================================
# STEP 2.6: ADR DISTRIBUTION
# HISTOGRAM
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x='adr',
    bins=40,
    kde=True,
    color='#e63946',
    edgecolor='black'
)

plt.title('Distribution of Average Daily Rate')
plt.xlabel('Average Daily Rate')
plt.ylabel('Frequency')

plt.show()


# ==========================================
# STEP 2.7: CORRELATION HEATMAP
# ==========================================

df_numeric = df.select_dtypes(include=['number'])

plt.figure(figsize=(14, 9))

sns.heatmap(
    df_numeric.corr(),
    cmap='icefire',
    annot=True,
    fmt='.2f'
)

plt.title('Correlation Heatmap of Hotel Booking Features')

plt.show()
