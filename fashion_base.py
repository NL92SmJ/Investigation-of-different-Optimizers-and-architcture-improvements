# Module importieren
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Ordner für Plots erstellen
os.makedirs("fashion_mnist_plots", exist_ok=True)

print("Lade Fashion-MNIST Datensatz...")
(train_data, train_col), (test_data, test_col) = tf.keras.datasets.fashion_mnist.load_data()

# 1. Normalisierung (Pixelwerte von 0-255 auf 0.0-1.0 skalieren)
train_data = train_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# 2. Kanal-Dimension für Conv2D hinzufügen -> Shape: (28, 28, 1)
train_data = train_data[..., tf.newaxis]
test_data = test_data[..., tf.newaxis]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- 3. MODELLARCHITEKTUR (CNN) ---
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Regulierung gegen Overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 10 Klassen für Klassifikation
])

# Kompilieren (Sparse Categorical Crossentropy für Integer-Labels)
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Early Stopping (Stoppt, wenn sich die Validation Loss 3 Epochen lang nicht verbessert)
cb_early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

print("\nStarte Training...")
# --- 4. TRAINING ---
history = model.fit(
    train_data, train_col,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[cb_early],
    verbose=1
)

# --- 5. EVALUATION ---
test_loss, test_acc = model.evaluate(test_data, test_col, verbose=0)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

# Vorhersagen für Testdaten generieren
predictions = model.predict(test_data)
pred_classes = np.argmax(predictions, axis=1)

# Ergebnisse als DataFrame speichern
df_results = pd.DataFrame({
    "Metrik": ["Test Loss", "Test Accuracy"],
    "Wert": [round(test_loss, 4), round(test_acc, 4)]
})
df_results.to_csv("fashion_mnist_summary.csv", index=False)
print("Tabelle erfolgreich als 'fashion_mnist_summary.csv' gespeichert!")

# --- 6. PLOT 1: Training History (Loss & Accuracy) ---
plt.figure(figsize=(12, 5))

# Loss-Verlauf
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss-Verlauf (Training vs. Validation)')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Accuracy-Verlauf
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Accuracy-Verlauf (Training vs. Validation)')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("fashion_mnist_plots/training_history.png")
plt.close()

# --- 7. PLOT 2: Confusion Matrix (Heatmap) ---
plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_col, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Fashion-MNIST Confusion Matrix')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Echte Klasse')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("fashion_mnist_plots/confusion_matrix.png")
plt.close()

print("Alle Evaluation-Plots wurden im Ordner 'fashion_mnist_plots' gespeichert!")