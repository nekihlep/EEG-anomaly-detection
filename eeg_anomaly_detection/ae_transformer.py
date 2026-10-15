"""Keras autoencoder as a train-only sklearn transformer."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class AutoencoderFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, latent_dim=32, epochs=50, batch_size=32, random_state=42):
        self.latent_dim=latent_dim
        self.epochs=epochs
        self.batch_size=batch_size
        self.random_state=random_state

    def fit(self, X, y=None):
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        tf.config.experimental.enable_op_determinism()
        X=np.asarray(X,dtype=np.float32)
        self.n_features_in_=X.shape[1]
        inputs=tf.keras.Input(shape=(X.shape[1],))
        hidden=tf.keras.layers.Dense(64,activation='relu')(inputs)
        latent=tf.keras.layers.Dense(self.latent_dim,activation='relu')(hidden)
        hidden=tf.keras.layers.Dense(64,activation='relu')(latent)
        outputs=tf.keras.layers.Dense(X.shape[1],activation='linear')(hidden)
        model=tf.keras.Model(inputs,outputs)
        model.compile(optimizer='adam',loss='mse')
        rng=np.random.default_rng(self.random_state)
        self.loss_history_=[]
        for epoch in range(self.epochs):
            indices=rng.permutation(len(X))
            model.reset_metrics()
            for start in range(0,len(X),self.batch_size):
                batch=X[indices[start:start+self.batch_size]]
                loss=model.train_on_batch(batch,batch)
            self.loss_history_.append(float(loss))
        # Store just encoder weights: portable through joblib/Pipeline caching.
        self.encoder_weights_=[layer.get_weights() for layer in model.layers[1:3]]
        self.reconstruction_mse_=float(np.mean((model(X,training=False).numpy()-X)**2))
        return self

    def transform(self, X):
        check_is_fitted(self,'encoder_weights_')
        X=np.asarray(X,dtype=np.float32)
        if X.shape[1]!=self.n_features_in_:
            raise ValueError('Unexpected number of spectral features')
        for weights,bias in self.encoder_weights_:
            X=np.maximum(X@weights+bias,0)
        return X
