#!/usr/bin/python3

from signal import siginterrupt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu')
        ])
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(256, activation='sigmoid')
        ])

    def call(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class AnomalyDetector:
    def __init__(self):
        self.auto_encoder = AutoEncoder()
        self.auto_encoder.compile(optimizer='adam', loss='mae')

    def train(self, train_data_path, test_data_path, epochs=50):
        train_data = pd.read_csv(train_data_path, header=None).to_numpy()
        test_data = pd.read_csv(test_data_path, header=None).to_numpy()
        self.auto_encoder.fit(
            x=train_data, y=train_data, epochs=epochs, validation_data=(test_data, test_data), batch_size=1024)

    def test(self, test_data_path):
        test_data = pd.read_csv(test_data_path, header=None).to_numpy()
        test_output = self.auto_encoder.predict(test_data, batch_size=1024)
        plt.plot(test_data, 'b')
        plt.plot(test_output.transpose()[0], 'r')
        plt.show()

    def save_models(self):
        self.auto_encoder.encoder.save('encoder')
        self.auto_encoder.decoder.save('decoder')
        encoder = tf.lite.TFLiteConverter.from_saved_model('encoder').convert()
        decoder = tf.lite.TFLiteConverter.from_saved_model('decoder').convert()
        with open('encoder.tflite', 'wb') as f:
            f.write(encoder)
        with open('decoder.tflite', 'wb') as f:
            f.write(decoder)


if __name__ == '__main__':
    detector = AnomalyDetector()
    detector.train(train_data_path='data/train.csv',
                   test_data_path='data/test.csv')
    detector.test(test_data_path='data/anomaly.csv')
    detector.test(test_data_path='data/test.csv')
    detector.save_models()
