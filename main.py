import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.tseries.offsets import DateOffset

# Function to process a single CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df['BILL_DATE'] = pd.to_datetime(df['BILL_DATE'], format='%Y%m%d')
    df = df.groupby("BILL_DATE")["QTY"].sum().reset_index()
    df = df.sort_values('BILL_DATE')
    return df

# Create features for the dataset
def create_features(df):
    df['year'] = df['BILL_DATE'].dt.year
    df['month'] = df['BILL_DATE'].dt.month
    df['day'] = df['BILL_DATE'].dt.day
    df['day_of_week'] = df['BILL_DATE'].dt.dayofweek
    df['day_of_year'] = df['BILL_DATE'].dt.dayofyear
    return df

# Streamlit UI
st.title('📈 ทำนายการขายรายเดือน 💸 ')

# Load the main dataset
csv_file = "C:\\Users\\santi\\Desktop\\CMU\\Equipment.csv"
df = process_csv(csv_file)

# Option to manually input or upload a CSV file
data_input_method = st.radio("กรุณาเลือกวิธีเพิ่มข้อมูล:", ('อัปโหลดไฟล์ CSV', 'กรอกตอนนี้'))

# Option 1: Upload CSV file for the last 7 days of new data
if data_input_method == 'อัปโหลดไฟล์ CSV':
    uploaded_file = st.file_uploader("อัปโหลดประวัติการดำเนินการเมื่อ 7 วันก่อน", type=['csv'])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data['BILL_DATE'] = pd.to_datetime(new_data['BILL_DATE'], format='%Y%m%d')

# Option 2: Manually input data for the last 7 days
elif data_input_method == 'กรอกตอนนี้':
    st.write("กรุณากรอกวันที่และจำนวนใน 7 วันที่ผ่านมา")
    input_data = {
        'BILL_DATE': ['20240701', '20240702', '20240703', '20240704', '20240705', '20240706', '20240707'],
        'QTY': [None]*7  # Placeholder for manual input
    }
    
    for i, date in enumerate(input_data['BILL_DATE']):
        input_data['QTY'][i] = st.number_input(f"ป้อนจำนวนสำหรับวันที่ {date}:", min_value=0, value=1000)

    # Create new DataFrame for manual input data
    new_data = pd.DataFrame(input_data)
    new_data['BILL_DATE'] = pd.to_datetime(new_data['BILL_DATE'], format='%Y%m%d')

try:
    # Combine new data with existing data
    df_updated = pd.concat([df, new_data]).sort_values('BILL_DATE').reset_index(drop=True)
    df_updated = create_features(df_updated)

    # Train-test split
    split_date = '2024-04-10'
    train = df_updated[df_updated['BILL_DATE'] <= split_date]
    test = df_updated[df_updated['BILL_DATE'] > split_date]

    X_train = train[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
    y_train = train['QTY']
    X_test = test[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
    y_test = test['QTY']

    # Train the model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Generate future dates for forecasting
    last_date = df_updated['BILL_DATE'].max()
    future_dates = [last_date + DateOffset(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({'BILL_DATE': future_dates})
    future_df = create_features(future_df)
    X_future = future_df[['year', 'month', 'day', 'day_of_week', 'day_of_year']]
    future_predictions = xgb_model.predict(X_future)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    plt.plot(train['BILL_DATE'], y_train, label='True QTY (Train)', color='blue', linewidth=2)
    plt.plot(test['BILL_DATE'], y_test, label='True QTY (Test)', color='green', linewidth=2)
    plt.plot(test['BILL_DATE'], y_pred, label='Predicted QTY (Test)', color='red', linestyle='--', linewidth=2)
    plt.plot(future_df['BILL_DATE'], future_predictions, label='Predicted QTY (Next 1 Month)', color='orange', linestyle='-.', linewidth=2)

    # Plot the new data in purple
    plt.plot(new_data['BILL_DATE'], new_data['QTY'], label='Added Data (Last 7 Days)', color='purple', linewidth=2, marker='o')

    # Formatting the plot
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate(rotation=45)

    plt.title(f'QTY Forecast with Last 7 Days Data', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Quantity (QTY)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(False)

    st.pyplot(plt)
except:
    st.markdown(
        """
    <div style='border: 2px solid red; border-radius: 5px; padding: 10px; background-color: white; font-family: "Times New Roman", Times, serif;'>
        <h1 style='text-align: center; color: red; font-family: "Times New Roman", Times, serif;'>
            ❌ ยังไม่อัปโหลดไฟล์ .csv ❌
        </h1>
    </div>
        """, unsafe_allow_html=True)
