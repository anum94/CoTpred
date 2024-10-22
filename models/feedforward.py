import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import ModelCheckpoint
def feedforward_network(X, y):



    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print (model.summary())
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2,
              callbacks = [WandbMetricsLogger(log_freq=10), checkpoint]
              )
    print (f"History: {history}"
           )

    best_model = tf.keras.models.load_model('best_model.keras')
    #print (history.params)
    print (history.history.keys["accuracy"])
    print(history.history.keys["loss"])
    print(history.history.keys["val_accuracy"])
    print(history.history.keys["val_loss"])

    # Evaluate the model
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')
    return accuracy , loss
