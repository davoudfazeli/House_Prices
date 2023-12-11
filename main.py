

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Read the CSV file
data = pd.read_csv("train.csv")

x_total = np.array(data[["YearBuilt", "GarageArea", "GarageCars", "PoolArea", "FullBath",
                          "TotalBsmtSF","WoodDeckSF","OpenPorchSF"]])
y_total = np.array(data['SalePrice'])

# Normalize input data
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x_total)
mx = np.mean(x_total, axis=0)
sx = np.std(x_total, axis=0)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_total, random_state=50, test_size=0.2)

model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

output = model.fit(x_train, y_train, epochs=300, validation_split=0.2, callbacks=[early_stopping], verbose=0)

plt.plot(output.history['loss'], label='Training Loss')
plt.plot(output.history['val_loss'], label='Validation Loss')
plt.title('Train Results')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)
my_house_spc = np.array([[2004, 60, 3, 0, 12, 120, 30, 10]])
my_spc = (my_house_spc-mx)/sx
# Sequential.save_model(model, 'trained_model.h5')
print("My House Price:",model.predict(my_spc))