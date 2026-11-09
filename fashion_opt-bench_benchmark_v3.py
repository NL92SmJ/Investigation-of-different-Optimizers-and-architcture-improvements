import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import os

# Ordner für Plots erstellen
os.makedirs("benchmark_results/plots", exist_ok=True)

print("="*60)
print("Lade und vorbereite Fashion-MNIST Datensatz...")
print("="*60)
(train_data, train_col), (test_data, test_col) = tf.keras.datasets.fashion_mnist.load_data()

# Normalisierung & Kanal-Dimension hinzufügen
train_data = (train_data.astype("float32") / 255.0)[..., tf.newaxis]
test_data = (test_data.astype("float32") / 255.0)[..., tf.newaxis]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_train_samples = train_data.shape[0] * 0.8  # 80% Training bei 0.2 validation_split

# ==========================================
# TEIL 1: OPTIMIERER-BENCHMARK
# ==========================================
print("\n--- STARTE TEIL 1: Optimierer-Benchmark ---")

optimizers_dict = {
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "AdamW": tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
    "SGD_Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001)
}

optimizer_results = []
optimizer_histories = {}

for name, opt in optimizers_dict.items():
    print(f"Trainiere mit Optimierer: {name}...")
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(train_data, train_col, epochs=12, batch_size=64, validation_split=0.2, callbacks=[cb_early], verbose=0)
    duration = time.time() - start_time
    
    epochs_trained = len(history.history['loss'])
    optimizer_histories[name] = history.history
    
    test_loss, test_acc = model.evaluate(test_data, test_col, verbose=0)
    images_per_sec = round((num_train_samples * epochs_trained) / duration, 1)
    
    optimizer_results.append({
        "Optimizer": name,
        "Test_Accuracy": round(test_acc * 100, 2),
        "Test_Loss": round(test_loss, 4),
        "Epochen": epochs_trained,
        "Dauer_Sek": round(duration, 1),
        "Images_per_sec": images_per_sec
    })

df_opt = pd.DataFrame(optimizer_results)
df_opt.to_csv("benchmark_results/optimizer_benchmark.csv", index=False)


# ==========================================
# TEIL 2: ARCHITEKTUR- & REGULARISIERUNGS-BENCHMARK
# ==========================================
print("\n--- STARTE TEIL 2: Architektur- & Feature-Benchmark ---")

# Data Augmentation Pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
])

architectures_dict = {
    "Baseline_Adam": {"use_bn": False, "use_aug": False, "use_lr_scheduler": False},
    "BatchNorm": {"use_bn": True, "use_aug": False, "use_lr_scheduler": False},
    "DataAugmentation": {"use_bn": False, "use_aug": True, "use_lr_scheduler": False},
    "LR_Scheduler": {"use_bn": False, "use_aug": False, "use_lr_scheduler": True},
    "Full_Optimized": {"use_bn": True, "use_aug": True, "use_lr_scheduler": True}
}

arch_results = []
arch_histories = {}  # Speichert die Trainingsverläufe der Architekturen
best_model = None
best_acc = 0.0
best_model_name = ""

for name, config in architectures_dict.items():
    print(f"Teste Architektur-Konfiguration: {name}...")
    
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    
    if config["use_aug"]:
        x = data_augmentation(x)
        
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(x)
    if config["use_bn"]:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same')(x)
    if config["use_bn"]:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same')(x)
    if config["use_bn"]:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Callback Setup
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    if config["use_lr_scheduler"]:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
        callbacks.append(lr_scheduler)
        
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(train_data, train_col, epochs=12, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=0)
    duration = time.time() - start_time
    
    # History abspeichern
    arch_histories[name] = history.history
    
    test_loss, test_acc = model.evaluate(test_data, test_col, verbose=0)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_model_name = name

    arch_results.append({
        "Configuration": name,
        "Test_Accuracy": round(test_acc * 100, 2),
        "Test_Loss": round(test_loss, 4),
        "Dauer_Sek": round(duration, 1)
    })

df_arch = pd.DataFrame(arch_results)
df_arch.to_csv("benchmark_results/architecture_benchmark.csv", index=False)


# ==========================================
# TEIL 3: DIAGRAMME & VISUALISIERUNGEN ERZEUGEN
# ==========================================
print("\n--- Generiere Visualisierungen ---")

# 1. Multi-Line-Plot: Optimierer (Train Loss, Validation Loss & Validation Accuracy)
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
for name, hist in optimizer_histories.items():
    plt.plot(hist['loss'], label=f"{name} (Train)", linewidth=2)
plt.title('Training Loss (Optimierer-Vergleich)')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 3, 2)
for name, hist in optimizer_histories.items():
    plt.plot(hist['val_loss'], label=f"{name} (Val)", linewidth=2, linestyle='--')
plt.title('Validation Loss (Optimierer-Vergleich)')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 3, 3)
for name, hist in optimizer_histories.items():
    plt.plot(hist['val_accuracy'], label=name, linewidth=2)
plt.title('Validation Accuracy (Optimierer-Vergleich)')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("benchmark_results/plots/multi_optimizer_curves.png")
plt.close()

# 2. Heatmap: Performance vs. Zeit (Optimierer)
plt.figure(figsize=(8, 4))
df_heat = df_opt.set_index("Optimizer")[["Test_Accuracy", "Dauer_Sek", "Images_per_sec"]]
sns.heatmap(df_heat, annot=True, fmt='g', cmap='YlGnBu', cbar=True)
plt.title('Optimizer Performance & Throughput Heatmap')
plt.ylabel('Optimierer')
plt.tight_layout()
plt.savefig("benchmark_results/plots/optimizer_heatmap.png")
plt.close()

# 3. Multi-Line-Plot für den Architekten-Vergleich (Validation Loss & Accuracy)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for name, hist in arch_histories.items():
    plt.plot(hist['val_loss'], label=name, linewidth=2)
plt.title('Validation Loss (Architektur-Vergleich)')
plt.xlabel('Epoche')
plt.ylabel('Val Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 2, 2)
for name, hist in arch_histories.items():
    plt.plot(hist['val_accuracy'], label=name, linewidth=2)
plt.title('Validation Accuracy (Architektur-Vergleich)')
plt.xlabel('Epoche')
plt.ylabel('Val Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("benchmark_results/plots/architecture_comparison_curves.png")
plt.close()

# 4. Confusion Matrix für das beste Modell aus Teil 2
print(f"Erstelle Confusion Matrix für das beste Modell: {best_model_name}...")
predictions = best_model.predict(test_data)
pred_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_col, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Bestes Modell ({best_model_name}: {best_acc*100:.2f}%)')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Echte Klasse')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("benchmark_results/plots/confusion_matrix.png")
plt.close()

print("\n" + "="*60)
print("BENCHMARK ERFOLGREICH ABGESCHLOSSEN!")
print("Alle Tabellen (CSV) und Plots liegen im Ordner 'benchmark_results/plots/'.")
print("="*60)