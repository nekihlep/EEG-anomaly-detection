from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

X_train = np.load('processed_data/X_train_scaled.npy')
X_test  = np.load('processed_data/X_test_scaled.npy')

input_layer = Input(shape=(114,))
encoded = Dense(64, activation='relu')(input_layer)
bottleneck = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(bottleneck)
output_layer = Dense(114, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, bottleneck)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1)

ae_features_train = encoder.predict(X_train)
ae_features_test  = encoder.predict(X_test)

np.save('processed_data/ae_features_train.npy', ae_features_train)
np.save('processed_data/ae_features_test.npy', ae_features_test)

print(f"Train: {ae_features_train.shape}, Test: {ae_features_test.shape}")