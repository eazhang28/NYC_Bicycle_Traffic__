import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

dataset_2 = pandas.read_csv('miniproject-f23-eazhang28/nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

# Numerical Conversions
day_to_numeric = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
months_to_numeric = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Turn Weekdays into Numeric Values
dataset_2['Weekday'] = dataset_2['Day'].map(day_to_numeric)

# Turn Date into Months without the Days
dataset_2[['Day', 'Month']] = dataset_2['Date'].str.split('-', expand=True)
dataset_2.drop('Date', axis=1, inplace=True)
dataset_2.drop('Day',axis=1, inplace=True )
dataset_2['Month'] = dataset_2['Month'].map(months_to_numeric)

# Which Three Bridges Should We Install Sensors On?
# Features: Brooklyn Bridge, Manhattan Bridge, Queesboro Bridge, Williamsburg Bridge
# Target: Total

def run_ridge_model(X, y, i):
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=61)

    reg = RidgeCV(store_cv_values=True)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.plot(y_test, '*', label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Model %d Performance' % i)
    plt.xlabel('Observations')
    plt.ylabel('Traffic (Count of Bicyclists)')
    plt.legend()
    plt.show()
    
    return mse, r2

bridges_combinations = [
    ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge'],
    ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'],
    ['Brooklyn Bridge', 'Queensboro Bridge', 'Williamsburg Bridge'],
    ['Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
]

mse_values = []
r2_values = []

for i, bridges in enumerate(bridges_combinations, 1):
    X_data = dataset_2[bridges].to_numpy()
    intercepts = np.ones((X_data.shape[0], 1))
    X_data = np.hstack((X_data, intercepts))
    
    y_data = dataset_2[['Total']].astype(int).to_numpy()
    
    mse, r2 = run_ridge_model(X_data, y_data, i)
    mse_values.append(mse)
    r2_values.append(r2)
    
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"Results for Model {i}:")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")
    
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("Summary:")
print("MSE values:", mse_values)
print("R^2 values:", r2_values)
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

# What Weather Conditions Should We Focus On?
# Features: High Temp, Low Temp, Precipitation
# Target: Total

X2 = dataset_2[['High Temp', 'Low Temp', 'Precipitation']].values
y2 = dataset_2[['Total']].astype(int).values

polyfts = PolynomialFeatures(degree=2,include_bias=False)
X2_poly = polyfts.fit_transform(X2)
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2_poly)

X2_train, X2_test, y2_train, y2_test, = tts(X2_scaled, y2, test_size=.3, random_state=11)

reg2 = Ridge(alpha=0.01)
reg2.fit(X2_train,y2_train)

y2_pred = reg2.predict(X2_test)
mse2 = mean_squared_error(y2_test, y2_pred)
coefdet2 = r2_score(y2_test, y2_pred)

alpha_values = [0, 0.001 , 0.01, 0.1, 1, 10, 100, 1000]
cv_scores = []
for alpha in alpha_values:
    reg_cv = Ridge(alpha=alpha)
    scores = cvs(reg_cv, X2_scaled, y2, cv=5, scoring='r2')
    cv_scores.append((alpha, scores.mean()))

best_alpha = max(cv_scores, key=lambda x: x[1])[0]
print('Best alpha from cross-validation:', best_alpha)

param_grid = {'alpha': [0, 0.001 , 0.01, 0.1, 1, 10, 100, 1000]}
ridge_grid = gscv(Ridge(), param_grid, cv=5, scoring='r2')
ridge_grid.fit(X2_scaled, y2)

best_alpha = ridge_grid.best_params_['alpha']
print('Best alpha from grid search:', best_alpha)

#using best alpha
final_reg = Ridge(alpha=best_alpha)
final_reg.fit(X2_scaled, y2)

y2_pred_final = final_reg.predict(X2_test)

mse_final = mean_squared_error(y2_test, y2_pred_final)
coefdet_final = r2_score(y2_test, y2_pred_final)
print('Final Ridge Regression MSE: %f' % mse_final)
print('Final Ridge Regression R2: %f' % coefdet_final)

plt.figure(5)
plt.plot(y2_test,'*')
plt.plot(y2_pred_final)
plt.title('Weather Conditions')
plt.xlabel('Observations')
plt.ylabel('Traffic (Count of Bicyclists)')
plt.show()

# What Day is it Based on the Traffic Numbers?
# Features: Total
# Target: Weekday

X = dataset_2[['Total']].to_numpy()
y = dataset_2[['Weekday']].astype(int)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# TRAIN TEST SPLIT 
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

#BASELINE LOGISTIC REGRESSION
scaler = StandardScaler()
X3_trained_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)
reg3 = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000)
reg3.fit(X_train, y_train)
y3_pred = reg3.predict(X_test)
accuracy = accuracy_score(y_test, y3_pred)
print('Q3 Accuracy: %f' % accuracy)
plt.figure(6)
plt.plot(y_test,'x')
plt.plot(y3_pred)
plt.xlabel('Observations')
plt.ylabel('Day of the Week (Numerical)')
plt.title('Multinomial Logisitic Regression Test vs. Prediction')
plt.show()

#RESPLIT
X_train, X_test, y_train, y_test = tts(X, y_encoded, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_encoded.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_decoded = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_decoded, y_pred)
print(f'Model Accuracy: {accuracy}')

# Plot training history
plt.figure(7)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and ValidationAccuracy over Epochs')
plt.legend()
plt.show()
