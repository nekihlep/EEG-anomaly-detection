"""Export 32 latent features fitted on the training partition only.
For cross-validation use step 12: the encoder must be refitted per fold.
"""
from pathlib import Path
import pickle
import numpy as np
from ae_transformer import AutoencoderFeatures

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'processed_data'

def main():
    train=np.load(DATA/'X_train_scaled.npy')
    test=np.load(DATA/'X_test_scaled.npy')
    encoder=AutoencoderFeatures().fit(train)
    np.save(DATA/'ae_features_train.npy',encoder.transform(train))
    np.save(DATA/'ae_features_test.npy',encoder.transform(test))
    with open(DATA/'models'/'ae_encoder.pkl','wb') as f:
        pickle.dump(encoder,f)
    print('Latent features:',encoder.transform(train).shape,encoder.transform(test).shape)

if __name__=='__main__':main()
