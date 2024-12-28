import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.backends.mkl import verbose
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import HeNormal, HeUniform

from utils.wandb import wandb_plot_line
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
def feedforward_network(X, y, exec_str, epochs = 5, i = -1, weights_init = "HE", batch_size = 8,
                        lr = 0.01, external_test_set = False, confidence_th = 0.5, optimizer = "adam"):

    best_model_path = os.path.join(exec_str,"models", f'best_model_hs_{str(i)}.keras')

    # Split the data into training and testing sets
    if external_test_set:
        feature_path = os.path.join("runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/test_features_1000", f"regression_features_layer_{i}.txt")
        X_test = np.loadtxt(feature_path, dtype=float)

        y_test = pd.read_excel("runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/deepmind-aqua_rat_balanced_1000_labelled.xlsx")["anum_decisions"]
        y_test = np.array(y_test).reshape(-1, 1)

        X_train = np.array(X)
        y_train = np.array(y).reshape(-1, 1)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    try:
        # Standardize the features
        #print (f"# training samples: {len(X_train)}, # Test samples: {len(y_test)}")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #model = KerasClassifier(build_fn=build_clf)
        #best_hp = get_best_hp(model)

        # Define the model
        if weights_init == 'HE_normal':
            model = Sequential([
                Dense(256, input_shape=(X_train.shape[1],), activation='relu', kernel_initializer=HeNormal()),
                Dense(128, activation='relu', kernel_initializer=HeNormal()),
                Dense(64, activation='relu', kernel_initializer=HeNormal()),
                Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
            ])

        if weights_init == 'HE_uniform':
            model = Sequential([
                Dense(256, input_shape=(X_train.shape[1],), activation='relu', kernel_initializer=HeUniform()),
                Dense(128, activation='relu', kernel_initializer=HeUniform()),
                Dense(64, activation='relu', kernel_initializer=HeUniform()),
                Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
            ])

        else:
            model = Sequential([
                Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
                Dense(128, activation='relu',),
                Dense(64, activation='relu',),
                Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
            ])

        # Compile the model
        if optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=lr)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        #print (model.summary())



        checkpoint = ModelCheckpoint(best_model_path, monitor='val_accuracy',
                                     save_best_only=True, mode='max', verbose=0)


        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                  callbacks = [WandbMetricsLogger(log_freq=10), checkpoint], verbose=False,
                  )
        plot_history(history)


        best_model = tf.keras.models.load_model(best_model_path)

        # Evaluate the model
        loss, accuracy = best_model.evaluate(X_test, y_test, verbose=False)
        #print(f'Test accuracy: {accuracy:.4f}')

        # Get predictions
        log_prob = best_model.predict(X_test, verbose=1)
        pred = (log_prob > confidence_th).astype("int32")

        compute_metrics(predictions=pred, true_labels=y_test, pred_prob = log_prob)
    except Exception as e:
        print(e)
        accuracy = 0
        loss = 0

    return accuracy , loss

def compute_metrics(predictions, true_labels, pred_prob):

    # Calculate Precision
    precision = precision_score(true_labels, predictions)
    #print(f'Precision: {precision}')

    # Calculate Recall
    recall = recall_score(true_labels, predictions)
    #print(f'Recall: {recall}')

    # Calculate F1 Score
    f1 = f1_score(true_labels, predictions)
    #print(f'F1 Score: {f1}')

    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    #print(f'Confusion Matrix:\n{conf_matrix}')

    # Calculate AUC
    auc = roc_auc_score(true_labels, pred_prob)
    #print(f'Area Under Curve (AUC): {auc}')

def plot_history(history):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]


    epochs = range(1, len(loss) + 1)

    wandb_plot_line(epochs,acc,"Training Accuracy per epoch", "epoch","acc")
    wandb_plot_line(epochs, loss, "Training Loss epoch", "epoch", "loss")
    wandb_plot_line(epochs, val_acc, "Validation Accuracy per epoch", "epoch", "val_acc")
    wandb_plot_line(epochs, val_loss, "Validation Loss per epoch", "epoch", "val_loss")


