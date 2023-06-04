import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
#%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime

# ---------------------------------------------------------------------Data preparation
# Read the data
df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/electricity2020-2023-2.csv")

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')

# Format the Date column to "2016-01-01 04:00"
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')

# drop columns
df.drop(['Prod. Photovoltaique (kWh)',	'Rayonnement (W/m2)', 'Temperature', 'consommation eau de pluie (m3)', 'consommation eau reseau (m3)','precipitations (mm)'], axis=1, inplace=True)

df = df.rename(columns={'Other Electricity': 'Other'})

# fill missing values with 0
df.fillna(value=0, inplace=True)

#CORRECTION------------------------------------
df.at[35797, 'Cooling'] = 11.823385
df.at[35798, 'Cooling'] = 11.823523
df.at[35799, 'Cooling'] = 11.823667
df.at[35800, 'Cooling'] = 11.823397
df.at[35801, 'Cooling'] = 23.644847
df.at[51160, 'Other'] = 23.644847
df.at[62467, 'Heating'] = 56.319
df.at[8870, 'Other'] = 70.24578125
df.at[43766, 'Other'] = 52.874
#----------------------------------------------

# create the 'Total' column
df['Consumption'] = df['Other'] + df['Heating'] + df['Lighting'] + df['Cooling'] + df['Plug'] + df['Ventilation']

# reorder the columns
df = df.reindex(columns=['Date', 'Heating', 'Lighting', 'Cooling', 'Plug', 'Ventilation', 'Other', 'Consumption'])

start_date = '2020-08-24'
end_date = '2023-04-26'

# convert the Date column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# filter the DataFrame to only include rows within the specified date range
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

df.head(10)
#----------------------------------------------------------------------------- # main graph

#-----------------------------------------------------------------------GRAPH
import plotly.graph_objects as go
import pandas as pd

# Convert the date column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Create the figure and traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=df['Consumption'], name='Consumption',
                         line=dict(color='red', width=2), visible=True))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Other'], name='Other',
                        line=dict(color='blue', width=2), yaxis='y2', visible=True))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Heating'], name='Heating',
                        line=dict(color='orange', width=2), yaxis='y3', visible=True))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Lighting'], name='Lighting',
                        line=dict(color='green', width=2), yaxis='y3', visible=True))

# Set the axis titles
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Consumption', titlefont=dict(color='red')),
    yaxis2=dict(title='Other', titlefont=dict(color='blue'), overlaying='y', side='right'),
    yaxis3=dict(title='Heating', titlefont=dict(color='orange'), overlaying='y', side='right'),
    yaxis4=dict(title='Lighting', titlefont=dict(color='green'), overlaying='y', side='right'),
    plot_bgcolor='white'
)

# Add hover information
fig.update_traces(hovertemplate='%{y:.2f}')

# Add checkboxes for trace selection
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label='Consumption',
                     method='update',
                     args=[{'visible': [True, False, False, False]}]),
                dict(label='Other',
                     method='update',
                     args=[{'visible': [False, True, False, False]}]),
                dict(label='Heating',
                     method='update',
                     args=[{'visible': [False, False, True, False]}]),
                dict(label='Lighting',
                     method='update',
                     args=[{'visible': [False, False, False, True]}]),
                dict(label='All',
                     method='update',
                     args=[{'visible': [True, True, True, True]}])
            ]),
            direction='down',
            xanchor='left',
            yanchor='top',
            y=1.15,
            x=0.05
        ),
    ]
)

# Show the figure
fig.show()

#-------------------------------------------------------------------------------- room capacity
# Read the Data
schedule = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/fusionschedule2020-2023.csv" , encoding='ISO-8859-1')

# Combine Date and Hour columns into a single column
schedule['Date'] = pd.to_datetime(schedule['Date'] + ' ' + schedule['Hour'])

# convert the Date column to datetime type
schedule['Date'] = pd.to_datetime(schedule['Date'], format='%Y-%m-%d %H:%M')

# Drop the original Hour columns
schedule = schedule.drop(['Hour'], axis=1)

start_date = '2020-08-24'
end_date = '2023-04-26'

# filter the DataFrame to only include rows within the specified date range
schedule = schedule[(schedule['Date'] >= start_date) & (schedule['Date'] <= end_date)]

schedule.head(10)

#----------------------------------------------------------------------- weather

# Read the data
weather = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/grenoble%202020-08-24%20to%202023-04-26.csv")

weather['datetime'] = pd.to_datetime(weather['datetime'])
# Format the datetime column
weather['Date'] = weather['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# Drop the columns
weather = weather.drop(['name', 'datetime', 'feelslike', 'dew', 'precipprob', 'preciptype', 'snow', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'snowdepth', 'icon', 'conditions', 'severerisk', 'stations'], axis=1)

weather = weather.rename(columns={'temp': 'Temperature', 'humidity': 'Humidity', 'precip':'Precipitation', 'cloudcover':'Cloudcover', 'visibility':'Visibility' ,'solarradiation':'Solarradiation', 'solarenergy':'Solarenergy', 'uvindex':'Uvindex'})

# reorder the columns
weather = weather.reindex(columns=['Date', 'Temperature', 'Humidity', 'Precipitation', 'Cloudcover', 'Visibility', 'Solarradiation', 'Solarenergy', 'Uvindex'])

weather.head(10)

#-----------------------------------------------------------------------merged
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
schedule['Date'] = pd.to_datetime(schedule['Date'], format='%Y-%m-%d %H:%M')
weather['Date'] = pd.to_datetime(weather['Date'], format='%Y-%m-%d %H:%M')

merged_df = pd.merge(df, schedule, on=['Date'], how='outer')
merged_df = pd.merge(merged_df, weather, on=['Date'], how='outer')
merged_df.fillna(0, inplace=True)
merged_df.head(10)
#---------------------------------------------------------------------- Boxplot
df_boxplot= merged_df.copy()

import plotly.subplots as sp
import plotly.express as px
import pandas as pd


df_boxplot = df_boxplot.set_index('Date')  # set 'Date' column as index
df_boxplot.index = pd.to_datetime(df_boxplot.index)  # convert to datetime data type

print(df_boxplot.index.isnull().sum()) # prints the number of missing values
print(df_boxplot.index.duplicated().sum()) # prints the number of duplicated values


# create dataframes for spring, summer, autumn, winter
df_spring = pd.concat([df_boxplot[(df_boxplot.index >= '2021-03-01') & (df_boxplot.index <= '2021-05-31')],
                       df_boxplot[(df_boxplot.index >= '2022-03-01') & (df_boxplot.index <= '2022-05-31')],
                       df_boxplot[(df_boxplot.index >= '2023-03-01') & (df_boxplot.index <= '2022-04-26')]])

df_summer = pd.concat([df_boxplot[(df_boxplot.index >= '2020-08-24') & (df_boxplot.index <= '2020-08-31')],
                       df_boxplot[(df_boxplot.index >= '2021-06-01') & (df_boxplot.index <= '2021-08-31')],
                       df_boxplot[(df_boxplot.index >= '2022-06-01') & (df_boxplot.index <= '2022-08-31')]])

df_autumn = pd.concat([df_boxplot[(df_boxplot.index >= '2020-09-01') & (df_boxplot.index <= '2020-11-30')],
                       df_boxplot[(df_boxplot.index >= '2021-09-01') & (df_boxplot.index <= '2021-11-30')],
                       df_boxplot[(df_boxplot.index >= '2022-09-01') & (df_boxplot.index <= '2022-11-30')]])

df_winter = pd.concat([df_boxplot[(df_boxplot.index >= '2020-12-01') & (df_boxplot.index <= '2021-02-28')],
                       df_boxplot[(df_boxplot.index >= '2021-12-01') & (df_boxplot.index <= '2022-02-28')],
                       df_boxplot[(df_boxplot.index >= '2022-12-01') & (df_boxplot.index <= '2023-02-28')]])


# Create a list of day names to use for X axis labels
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create subplots for each season
fig_boxplot = sp.make_subplots(rows=2, cols=2, subplot_titles=('Spring Consumption', 'Summer Consumption',
                                                       'Autumn Consumption', 'Winter Consumption'),
                       shared_xaxes=True, vertical_spacing=0.1)

# Add each boxplot to the subplots
fig_boxplot.add_trace(px.box(df_spring, x=df_spring.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='red')), row=1, col=1)

fig_boxplot.add_trace(px.box(df_summer, x=df_summer.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='blue')), row=1, col=2)

fig_boxplot.add_trace(px.box(df_autumn, x=df_autumn.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='orange')), row=2, col=1)

fig_boxplot.add_trace(px.box(df_winter, x=df_winter.index.day_name(), y='Consumption', category_orders={'x': day_names}).data[0].update(marker=dict(color='green')), row=2, col=2)


# Update the layout
fig_boxplot.update_layout(plot_bgcolor='white', 
                  #title_text="Consumption by Season"
                  )


#fig.update_xaxes(title_text="Day of the Week", row=1, col=1)
#fig.update_xaxes(title_text="Day of the Week", row=1, col=2)
#fig.update_xaxes(title_text="Day of the Week", row=2, col=1)
#fig.update_xaxes(title_text="Day of the Week", row=2, col=2)

fig_boxplot.update_yaxes(title_text="Consumption", row=1, col=1)
fig_boxplot.update_yaxes(title_text="", row=1, col=2)
fig_boxplot.update_yaxes(title_text="Consumption", row=2, col=1)
fig_boxplot.update_yaxes(title_text="", row=2, col=2)

# Display the plot
fig_boxplot.show()
#------------------------------------------------------------------ day index
# Add a new column for day of the week
merged_df['Dayofweek'] = merged_df['Date'].dt.day_name()

# --------------------------------------Add Dayindex---------------------
def get_dayindex_map(month):
    if month in [12, 1, 2]:  # Winter season
        return {'Monday': 1, 'Tuesday': 0.973, 'Wednesday': 0.996, 'Thursday': 0.967, 'Friday': 0.939, 'Saturday': 0.6, 'Sunday': 0.628}
    elif month in [3, 4, 5]:  # Spring season
        return {'Monday': 0.679, 'Tuesday': 0.669, 'Wednesday': 0.684, 'Thursday': 0.635, 'Friday': 0.615, 'Saturday': 0.360, 'Sunday': 0.371}
    elif month in [6, 7, 8]:  # Summer season
        return {'Monday': 0.578, 'Tuesday': 0.607, 'Wednesday': 0.584, 'Thursday': 0.581, 'Friday': 0.566, 'Saturday': 0.402, 'Sunday': 0.408}
    else:  # Autumn season
        return {'Monday': 0.651, 'Tuesday': 0.644, 'Wednesday': 0.636, 'Thursday': 0.619, 'Friday': 0.631, 'Saturday': 0.359, 'Sunday': 0.366}

# Apply the function to modify the Dayindex column
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Dayindex'] = merged_df.apply(lambda row: get_dayindex_map(row['Month'])[row['Dayofweek']], axis=1)
merged_df = merged_df.drop(['Month', 'Dayofweek'], axis=1)
#----------------------------------------------------------------
merged_df.rename(columns={'Capacity': 'Occupants'}, inplace=True)
merged_df.head(10)
#-------------------------------------------------------------------- #delete 23 rows
merged_df = merged_df.drop(merged_df.tail(23).index)
merged_df.tail(10)
#------------------------------------------------------------------------------- print average 3 days

# Select the last 72 rows of the columns
last_72_rows = merged_df[['Heating', 'Consumption', 'Temperature', 'Occupants']].tail(72)

# Calculate the average values
average_values = last_72_rows.mean()

# Print the average values
print("Average values of the last 72 rows:")
print(round(average_values['Heating'], 2))
print(round(average_values['Consumption'], 2))
print(round(average_values['Temperature'], 2))
print('{:.0f}'.format(average_values['Occupants']))


#-------------------------------------------------------------------------------- graph of the last 3 days
import plotly.graph_objects as go
import pandas as pd

# Select the last 168 rows of the Consumption column
last_72_heating = merged_df['Heating'].tail(72)
last_72_consumption = merged_df['Consumption'].tail(72)
last_72_temperature = merged_df['Temperature'].tail(72)
last_72_occupants = merged_df['Occupants'].tail(72)

# Create the figure and trace
fig_heating = go.Figure()
fig_consumption = go.Figure()
fig_temperature = go.Figure()
fig_occupants = go.Figure()

fig_heating.add_trace(go.Scatter(x=merged_df['Date'].tail(72), y=last_72_heating, name='Heating',
                         line=dict(color='red', width=2), fill='tozeroy'))
fig_consumption.add_trace(go.Scatter(x=merged_df['Date'].tail(72), y=last_72_consumption, name='Consumption',
                         line=dict(color='blue', width=2), fill='tozeroy'))
fig_temperature.add_trace(go.Scatter(x=merged_df['Date'].tail(72), y=last_72_temperature, name='Temperature',
                         line=dict(color='orange', width=2), fill='tozeroy'))
fig_occupants.add_trace(go.Scatter(x=merged_df['Date'].tail(72), y=last_72_occupants, name='Occupants',
                         line=dict(color='green', width=2), fill='tozeroy'))

# Set the axis titles
fig_heating.update_layout(
    plot_bgcolor='white', height=150,
    xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
    yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
)
fig_consumption.update_layout(
    plot_bgcolor='white', height=150,
    xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
    yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
)
fig_temperature.update_layout(
    plot_bgcolor='white',height=150,
    xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
    yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
)
fig_occupants.update_layout(
    plot_bgcolor='white',height=150,
    xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
    yaxis=dict(showticklabels=False)  # Hide y-axis tick labels
)

# Add hover information
fig_heating.update_traces(hovertemplate='%{y:.2f}')
fig_consumption.update_traces(hovertemplate='%{y:.2f}')
fig_temperature.update_traces(hovertemplate='%{y:.2f}')
fig_occupants.update_traces(hovertemplate='%{y:.2f}')

# Set the margin and padding to full screen
fig_heating.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig_consumption.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig_temperature.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig_occupants.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# Show the figure
fig_heating.show()
fig_consumption.show()
fig_temperature.show()
fig_occupants.show()


#--------------------------------------------------------------------------------exogeneous visualization

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create the subplots
fig_exogeneous = make_subplots(rows=2, cols=3, subplot_titles=('Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility'))

# Add the traces to the subplots

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Number of Room'], name='Number of Room',
                         line=dict(color='Coral', width=2), fill='tozeroy'), row=1, col=1)

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Dayindex'], name='Dayindex',
                         line=dict(color='orange', width=2), fill='tozeroy'), row=1, col=2)

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Occupants'], name='Occupants',
                         line=dict(color='Crimson', width=2), fill='tozeroy'), row=1, col=3)

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Temperature'], name='Temperature',
                         line=dict(color='blue', width=2), fill='tozeroy'), row=2, col=1)

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cloudcover'], name='Cloudcover',
                         line=dict(color='DarkCyan', width=2), fill='tozeroy'), row=2, col=2)

fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Visibility'], name='Visibility',
                         line=dict(color='purple', width=2), fill='tozeroy'), row=2, col=3)


# Set the axis titles
fig_exogeneous.update_xaxes(title_text='Date', row=1, col=1)
fig_exogeneous.update_yaxes(title_text='Number of Room', title_font=dict(color='Coral'), row=1, col=1)
fig_exogeneous.update_yaxes(title_text='Dayindex', title_font=dict(color='orange'), row=1, col=2)
fig_exogeneous.update_yaxes(title_text='Occupants', title_font=dict(color='Crimson'), row=1, col=3)
fig_exogeneous.update_yaxes(title_text='Temperature', title_font=dict(color='blue'), row=2, col=1)
fig_exogeneous.update_yaxes(title_text='Cloudcover', title_font=dict(color='DarkCyan'), row=2, col=2)
fig_exogeneous.update_yaxes(title_text='Visibility', title_font=dict(color='purple'), row=2, col=3)

# Add hover information
fig_exogeneous.update_traces(hovertemplate='%{y:.2f}')

# Update the layout
fig_exogeneous.update_layout(plot_bgcolor='white', showlegend=False)

# Show the figure
fig_exogeneous.show()

#-----------------------------------------------------------------------------------Importances

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df.set_index('Date', inplace=True)

# Define features and target
features = ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']
target = 'Consumption'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(merged_df[features], merged_df[target], test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the performance
y_pred = rf.predict(X_test)
print('R2 score:', r2_score(y_test, y_pred))

# Get feature importances from the model
importances = rf.feature_importances_

# Create a bar plot of feature importances using Plotly
colors = ['Coral', 'orange', 'Crimson', 'blue', 'DarkCyan', 'purple']  # Specify different colors for each bar
fig_rf = go.Figure([go.Bar(x=features, y=importances, text=importances, textposition='auto', textfont=dict(size=12),
                           marker=dict(color=colors))])

# Update layout and axes properties
fig_rf.update_layout(
    #title='Feature Importances',
    #width=1400,  # set width of the plot
    #height=400,  # set height of the plot
    font=dict(size=12),
    plot_bgcolor='white',
    xaxis=dict(tickangle=45)
)

fig_rf.show()
'''
#----------------------------------------------------------------------------------- Forecast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Convert the date and hour columns to datetime format
data = merged_df.copy()

# Rename the index level "Date" to "Datetime"
data.index.names = ['Datetime']

# Split the data into input (X) and output (Y) variables
X = data[['Number of Room', 'Dayindex', 'Occupants']].values
Y = data['Consumption'].values

# Normalize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Split the data into training and validation sets
train_X = X[:-24]
train_Y = Y[:-24]
val_X = X[-24:]
val_Y = Y[-24:]

# Train the model and store the history object
history = model.fit(train_X, train_Y, epochs=2, batch_size=10, verbose=2, validation_data=(val_X, val_Y))

# Ask the user how many hours ahead to predict
num_hours = 23

# Generate the list of dates and hours to predict
#last_datetime = pd.to_datetime('2023-04-26 00:00')
last_datetime = data.index.max()
next_day = last_datetime + pd.DateOffset(hours=1)
datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
selected_datetimes = [str(d) for d in datetime_range]

# Make predictions for the selected dates and hours
input_data = np.zeros((num_hours, X.shape[1]))
numberofroom_arr = [0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 5, 25, 25, 21, 19, 11, 2, 2, 0, 0, 0, 0]  # input values for number of rooms
dayindex_arr = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635,  0.635,  0.635,  0.635, 0.635]  # input values for day index
occupants_arr = [0, 0, 0, 0, 0, 0, 0, 0, 923, 923, 923, 923, 633, 1068, 1068, 964, 908, 791, 371, 371, 0, 0, 0, 0]  # input values for number of occupants
for i in range(num_hours):
    numberofroom = numberofroom_arr[i]
    dayindex = dayindex_arr[i]
    occupants = occupants_arr[i]
    input_data[i] = [numberofroom, dayindex, occupants]

input_data = (input_data - X_mean) / X_std
predictions = model.predict(input_data)

# Print the predictions
for i in range(num_hours):
    print('Predicted consumption for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))

 #--------------------------------------------------------------------------------------- visualization

# Get the predictions for the training data
train_predictions = model.predict(X)

# Plot the true consumption values and the corresponding predicted values
fig_tp = go.Figure()
fig_tp.add_trace(go.Scatter(x=data.index, y=Y, name='True Consumption', showlegend=False, line_color='orange'))
fig_tp.add_trace(go.Scatter(x=data.index, y=train_predictions.flatten(), name='Predicted Consumption', showlegend=False, line_color='red'))
fig_tp.update_layout(
                  plot_bgcolor='white',
                  xaxis_title='Date and Time', yaxis_title='Consumption')
fig_tp.show()

 # ---------------------------------------------------------------------------------------Train Loss

import plotly.graph_objects as go

# Get the training and validation loss values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a figure
fig_tvl = go.Figure()

# Add the training loss trace
fig_tvl.add_trace(go.Scatter(
    x=list(range(1, len(train_loss) + 1)),
    y=train_loss,
    showlegend=True,
    mode='lines',
    name='Training Loss'
))

# Add the validation loss trace
fig_tvl.add_trace(go.Scatter(
    x=list(range(1, len(val_loss) + 1)),
    y=val_loss,
    showlegend=True,
    mode='lines',
    name='Validation Loss',
    
))

# Update the layout
fig_tvl.update_layout(
    plot_bgcolor='white',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend=dict(x=0.02, y=0.98),
    margin=dict(l=40, r=20, t=60, b=40),
)

# Show the figure
fig_tvl.show()




 #----------------------------------------------------------------------------------------Prediction chart
# Print the predictions
for i in range(num_hours):
    print('Electricity consumption forecast for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))

# Evaluate the model on the training set
rmse = sqrt(mean_squared_error(Y, model.predict(X)))
mse = mean_squared_error(Y, model.predict(X))
mae = mean_absolute_error(Y, model.predict(X))
r2 = r2_score(Y, model.predict(X))
print('RMSE: {:.2f}'.format(rmse))
print('MSE: {:.2f}'.format(mse))
print('MAE: {:.2f}'.format(mae))
print('R2 score: {:.2f}'.format(r2))

# Show the chart of the last three days and the predicted days
last_three_days = data.iloc[-1:]
predicted_days = pd.DataFrame(predictions, columns=['Consumption'], index=datetime_range)

fig_prediction = go.Figure()
fig_prediction.add_trace(go.Bar(x=last_three_days.index, y=last_three_days['Consumption'], name='Previous days', showlegend=False,))
fig_prediction.add_trace(go.Bar(x=predicted_days.index, y=predicted_days['Consumption'], name='Predicted days', showlegend=False,))
fig_prediction.update_layout(plot_bgcolor='white', xaxis_title='Date', yaxis_title='Electricity consumption')
fig_prediction.show()
'''

#---------------------------------------FINISH

import dash
from dash import dash_table
from dash import dcc
from dash import html

external_stylesheets = ['style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'GreEn-ER Electricity'

# Define the layout
app.layout = html.Div(
    children=[
        # Header
        html.Div(
            children=[
                # Left column with image icon
                html.Div(
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi1b2TrokIZaf_b_sousXB9cM84AfuHI9ZLxwlnNAvx85NBN5ZSeNWQAxxwCYSXakj7guqwBKn-O1H85BpdsBMjqTQVmV9ACpLvjZmqDQ5oygps-mlWV1OxHfmm0XXJCh96RgRk0M7xkqcoPwp37A4lB1QEzFXVeSUXxdG3frpaYZ5M_RG3z-0mMYgoUg/s1600/download-removebg-preview.png"),
                    style={'width': '15%','display': 'inline-block', "text-align" :"left"}
                ),
                # Center column with title
                html.Div(
                    [
                        html.Span("GreEn-ER Building Electricity", style={'font-family' : 'calibri', 'font-weight': '600', 'font-size': '40px', 'display': 'block'}),
                        html.Span("Author : Bahauddin Habibullah | Supervisor : Benoit Delinchant", style={'font-family' : 'calibri', 'font-size': '16px', 'display': 'block'}),
                    ],
                    style={'width': '65%', 'display': 'inline-block', 'text-align': 'left'}
                ),
                # Right column with image icon
                html.Div(
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEixIt_aFT2aA2Y_FUL6N1uAAX8CW-NdlNxP6BG-ggzuhCgMdzBzeQpYb5Wb6HqtlkSentBuKjIzIY-TtlR1TPnkFyh1jSrmwyKXgUzlw0aljCT-m1O44MFo8is_tIlg59JVf4biACzqIICfONNqicCIMvA1TQzl0QlVmzkgylnfiyNVf3As0Er8jMHK0w/s1600/download__1_-removebg-preview.png"),
                    style={'width': '20%', 'display': 'inline-block', 'text-align': 'right'}
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'padding' : '40px 2%',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}
        ),

        # first row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Heating", style={'font-family' : 'calibri', 'font-weight': '300','font-size': '20px', 'display': 'block'}),
                     html.Span("Last 3 days average", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{round(average_values['Heating'], 2)}", style={'margin-bottom':'-40px', 'margin-top':'-30px','font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     dcc.Graph(figure=fig_heating,  style={'margin':'-10px',})
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Consumption", style={'font-family' : 'calibri', 'font-weight': '300',  'font-size': '20px', 'display': 'block'}),
                     html.Span("Last 3 days average", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{round(average_values['Consumption'], 2)}", style={'margin-bottom':'-40px', 'margin-top':'-30px','font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     dcc.Graph(figure=fig_consumption, style={'margin':'-10px',})
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Temperature", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Last 3 days average", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{round(average_values['Temperature'], 2)}", style={'margin-bottom':'-40px', 'margin-top':'-30px','font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4','font-size': '40px', 'display': 'block'}),
                     dcc.Graph(figure=fig_temperature, style={'margin':'-10px',})
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("Occupants", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Last 3 days average", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'display': 'block'}),
                     #dcc.Markdown(f"Number of missing values: {df_boxplot.index.isnull().sum()}", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '40px', 'display': 'block'}),
                     dcc.Markdown(f"{'{:.0f}'.format(average_values['Occupants'])}", style={'margin-bottom':'-40px', 'margin-top':'-30px','font-family' : 'calibri', 'font-weight': '600', 'color':'#008bd4', 'font-size': '40px', 'display': 'block'}),
                     dcc.Graph(figure=fig_occupants, style={'margin':'-10px',})
                     #html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjTJLFaFGNCbJ7Yn67lMwndmVMKUV4kUXhY56ml8tD7OVHnuFfAyLV01p-s75NvhIKsRI0_9IfTcRyOSJ74xwpiDTQkNiatJUGBIxrpw9Ky6QgoErYXSmXD5po11R7go80HOptRZPukiWgPzekOF2EjIdG0WyUbZ9RcZPMfq0fsXsHtBmW-0wXa0xFOKg/s1600/download%20(1).png"),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),



        # Graphs second row
        html.Div(
            children=[
                html.Div(
                    [html.Span("GreEn-ER Electricity consumption", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Time series of elctricity prediction and consumption over the past 3 years", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig)
                    ],
                    style ={'width': '71%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
                html.Div(
                    [html.Span("The Region", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                    html.Img(src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgaD593j9Yiignmfq2YtP_rMTKRoNeVxdZ76JXDVj0JlN3qO5kimtLIYi8zA1GXRuMWcImIAzU1h8cnNeFqbQoRZvUOreHj2CxmaM6isGIPnUyX9a79WXulfOj8sFM80gCAJvhGi-SBi6WHMTPoytA_tAQTHNP8gVUgrZVsxTaI0nZ48tOiCGrtmxODKg/s320/939px-Auvergne-Rh%C3%B4ne-Alpes_region_map_(DPJ-2020).svg.png", style={'max-width': '100%'}),
                    html.Span("Auvergne-Rhône-Alpes is a region in southeast-central France created by the 2014 territorial reform of French regions; it resulted from the merger of Auvergne and Rhône-Alpes. ", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    ],
                    style ={'width': '21%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '1px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                )
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),

        # Graphs third row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Exogeneous Data", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Record of external variables for electricity building forecast", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_exogeneous)
                    ],
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),

                html.Div(
                    [html.Span("Features Importance", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Random Forest method to find features importances", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_rf)
                    ],
                    style ={'width': '46%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),
            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),
'''
        # Graphs fourth row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Forecast: True and Prediction", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Neural Network ML Model for the electricity prediction", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_tp)
                    ],
                    style ={'width': '29%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),

                html.Div(
                    [html.Span("Training Loss and Validation", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("10 Epoch - 10 Batch size - 2 Verbose", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_tvl)
                    ],
                    style ={'width': '29%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),

                html.Div(
                    [html.Span("Prediction", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Due to the lower epoch defined (for fast online load), the prediction may be innacurate", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                    dcc.Graph(figure=fig_prediction)
                    ],
                    style ={'width': '29%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),

            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),
      '''

        # Graphs fifth row
        html.Div(
            children=[
                html.Div(
                    [html.Span("Dataset", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '20px', 'display': 'block'}),
                     html.Span("Combined Dataset - Class, Electricity, and Weather", style={'font-family' : 'calibri', 'font-weight': '300', 'font-size': '14px', 'line-height' : '40px', 'display': 'block'}),
                     dash_table.DataTable(data=merged_df.to_dict('records'), page_size=10, 
                     #style_data={'whiteSpace': 'normal','font-family': 'calibri','height': 'auto','font-size': '10px','text-align': 'left'},
                     #style_header={'font-family': 'calibri','font-size': '11px','font-weight': 'bold','background-color': '#f3f3f3','border-top-left-radius': '10px','border-top-right-radius': '10px','padding': '5px',}
                     
                     style_table={
                        'overflowX': 'auto',
                        'width': '100%',
                        'height': '400px',
                        'margin-top': '10px',
                        'margin-bottom': '10px',
                        'border': '0px solid lightgray',
                        'border-radius': '0px',
                    },
                    style_cell={
                        'font-family': 'calibri',
                        'font-size': '10px',
                        'text-align': 'left',
                        'padding': '5px',
                    },
                    style_header={
                        'font-family': 'calibri',
                        'font-size': '11px',
                        'font-weight': 'bold',
                        'background-color': '#f3f3f3',
                        'border-top-left-radius': '10px',
                        'border-top-right-radius': '10px',
                        'padding': '5px',
                    },),

                    ],
                    style ={'width': '96%',
                            'padding' :'1%',
                            'margin': '1%',
                            'font-size': '25px',
                            'border-radius': '10px',
                            'background': '#fff',
                            'display': 'inline-block'},
                ),



            ],
            style={'max-width': '1500px',
                   'margin': '0 auto',
                   'display': 'flex',
                   'flex-wrap': 'wrap'}  # Set the background color to #000 (black)
        ),



        


        
    ],
    style={'background-color': '#fcfcfc', 'margin' :'-8px'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, host="0.0.0.0", port=8080)


