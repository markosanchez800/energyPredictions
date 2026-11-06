# EDA graphs for the household power consumption dataset
# run this file and it will save all the graphs into a graphs/ folder

from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # so it doesn't try to open windows, just saves the pngs
import matplotlib.pyplot as plt
import os

# same cleaning steps as energyPredictions.py so the graphs match the model data
def loadData():
    dataset = fetch_ucirepo(id=235)
    df = dataset.data.features.copy()

    df['dateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])

    # some values are '?' in the raw data so convert everything to numbers and drop the bad rows
    for col in df.columns.drop('dateTime'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    df['hour'] = df['dateTime'].dt.hour
    df['day'] = df['dateTime'].dt.dayofweek
    df['month'] = df['dateTime'].dt.month

    df = df.sort_values('dateTime')
    return df


# Graph 1: daily average of Global_active_power over the whole time period
def dailyAverageGraph(df):
    daily = df.set_index('dateTime')['Global_active_power'].resample('D').mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily.values, linewidth=0.8)
    plt.title('Daily Average Global Active Power (Dec 2006 - Nov 2010)')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/dailyAverage.png')
    plt.close()


# Graph 2: distribution of Global_active_power (histogram)
def distributionGraph(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Global_active_power'], bins=100, edgecolor='black', linewidth=0.3)
    plt.title('Distribution of Global Active Power')
    plt.xlabel('Global Active Power (kW)')
    plt.ylabel('Number of Minutes')
    plt.tight_layout()
    plt.savefig('graphs/distribution.png')
    plt.close()


# Graph 3: average usage by hour of the day
def hourlyAverageGraph(df):
    hourly = df.groupby('hour')['Global_active_power'].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(hourly.index, hourly.values)
    plt.title('Average Global Active Power by Hour of Day')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Global Active Power (kW)')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('graphs/hourlyAverage.png')
    plt.close()


# Graph 4: average usage by day of the week
def dayOfWeekGraph(df):
    weekly = df.groupby('day')['Global_active_power'].mean()
    dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    plt.figure(figsize=(8, 5))
    plt.bar(dayNames, weekly.values)
    plt.title('Average Global Active Power by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Global Active Power (kW)')
    plt.tight_layout()
    plt.savefig('graphs/dayOfWeek.png')
    plt.close()


# Graph 5: average usage by month (shows the seasonal pattern)
def monthlyAverageGraph(df):
    monthly = df.groupby('month')['Global_active_power'].mean()
    monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(10, 5))
    plt.bar(monthNames, monthly.values)
    plt.title('Average Global Active Power by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Global Active Power (kW)')
    plt.tight_layout()
    plt.savefig('graphs/monthlyAverage.png')
    plt.close()


# Graph 6: how much of the power goes to each sub meter by hour
# sub_metering_1 = kitchen, sub_metering_2 = laundry room, sub_metering_3 = water heater + AC
def subMeteringGraph(df):
    subs = df.groupby('hour')[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(subs.index, subs['Sub_metering_1'], marker='o', label='Sub_metering_1 (kitchen)')
    plt.plot(subs.index, subs['Sub_metering_2'], marker='o', label='Sub_metering_2 (laundry)')
    plt.plot(subs.index, subs['Sub_metering_3'], marker='o', label='Sub_metering_3 (water heater / AC)')
    plt.title('Average Sub Metering by Hour of Day')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Energy (Wh)')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/subMetering.png')
    plt.close()


if __name__ == '__main__':
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    df = loadData()
    print('rows after cleaning:', len(df))

    dailyAverageGraph(df)
    distributionGraph(df)
    hourlyAverageGraph(df)
    dayOfWeekGraph(df)
    monthlyAverageGraph(df)
    subMeteringGraph(df)

    print('saved all graphs to graphs/')
