# ======================================================
# main.py — Full HFPSO Feature Selection Pipeline
# Run command:
#   python main.py --dataset "path/to/IIITDMJ_Smoke"
# ======================================================

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time

# ======================================================
# ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(description="Run HFPSO feature selection on ResNet features.")
parser.add_argument("--dataset", required=True, help="Path to dataset folder (e.g., IIITDMJ_Smoke)")
args = parser.parse_args()

DATAROOT = args.dataset
RESIZE = 224
SEED = 42

print(f"📂 Using dataset path: {DATAROOT}")

# ======================================================
# HELPER FUNCTION: Discover Images
# ======================================================
def discover_images(dataroot, exts=(".jpg", ".jpeg", ".png", ".bmp",".tif")):
    root = Path(dataroot)
    pairs = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in cls.rglob("*"):
            if p.suffix.lower() in exts:
                pairs.append((p, cls.name))
    if not pairs:
        raise RuntimeError(f"No images found in {dataroot}")
    return pairs

# ======================================================
# CHECK FOR EXISTING TRAIN/TEST/VAL FOLDERS
# ======================================================
train_dir = os.path.join(DATAROOT, "train")
test_dir = os.path.join(DATAROOT, "test")
val_dir = os.path.join(DATAROOT, "validation")

train_generator = None
val_generator = None
test_generator = None

if all(os.path.exists(d) and len(os.listdir(d)) > 0 for d in [train_dir, test_dir, val_dir]):
    print("✅ Using existing train/test/validation folders")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(RESIZE, RESIZE), batch_size=32, class_mode="categorical", seed=SEED
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(RESIZE, RESIZE), batch_size=32, class_mode="categorical", seed=SEED
    )

    test_generator = val_datagen.flow_from_directory(
        test_dir, target_size=(RESIZE, RESIZE), batch_size=32, class_mode="categorical", seed=SEED
    )

else:
    print("⚙️ No train/test/val folders found — performing custom split (70/20/10)...")

    pairs = discover_images(DATAROOT)
    images, labels = [], []

    for p, c in pairs:
        img = Image.open(p).convert("RGB").resize((RESIZE, RESIZE))
        images.append(np.array(img))
        labels.append(c)

    images = np.array(images)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=SEED)

    print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    num_classes = len(np.unique(labels_encoded))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=SEED)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32, shuffle=False)
    test_generator = val_datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

print("✅ Data generators ready!")

# ======================================================
# RESNET50 FEATURE EXTRACTION + FINE TUNING
# ======================================================
if hasattr(train_generator, "class_indices") and train_generator.class_indices:
    num_classes = len(train_generator.class_indices)
else:
    num_classes = y_train.shape[1]

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(RESIZE, RESIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)

for layer in base_model.layers[-30:]:
    layer.trainable = True

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(train_generator, epochs=20,
                             validation_data=val_generator,
                             callbacks=[reduce_lr, early_stop])

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy after fine-tuning: {test_acc:.4f}")

# ======================================================
# FEATURE EXTRACTION
# ======================================================
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)
X_train_feats = feature_extractor.predict(X_train, batch_size=32, verbose=1)
X_val_feats = feature_extractor.predict(X_val, batch_size=32, verbose=1)
X_test_feats = feature_extractor.predict(X_test, batch_size=32, verbose=1)

print(f"Shape of extracted features: {X_train_feats.shape}")

# ======================================================
# BASELINE CLASSIC ML PERFORMANCE (All Features)
# ======================================================
print("\n🏁 Computing Baseline Accuracy using All Extracted Features...\n")

baseline_models = {
    'svm': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale')),
    'rf': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    'ksvm': make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, C=1.0, gamma='scale')),
    'nn': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42))
}

baseline_results = []

for name, clf in baseline_models.items():
    start_time = time.time()
    clf.fit(X_train_feats, np.argmax(y_train, axis=1))
    duration = time.time() - start_time

    y_val_pred = clf.predict(X_val_feats)
    y_test_pred = clf.predict(X_test_feats)

    val_acc = accuracy_score(np.argmax(y_val, axis=1), y_val_pred)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), y_test_pred, average='weighted', zero_division=0)

    baseline_results.append({
        'model': name,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'train_time_sec': duration
    })

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv("baseline_model_results.csv", index=False)
print("✅ Saved baseline results to 'baseline_model_results.csv'")

best_base = baseline_df.loc[baseline_df['test_accuracy'].idxmax()]
print(f"📊 Best Baseline Model: {best_base['model']}")
print(f"🔹 Validation Accuracy: {best_base['val_accuracy']:.4f}")
print(f"🔹 Test Accuracy: {best_base['test_accuracy']:.4f}\n")

# ======================================================
# HFPSO IMPLEMENTATION
# ======================================================
@dataclass
class HFPSOParams:
    nparticles: int = 20
    maxiter: int = 50
    a1: float = 1.4
    a2: float = 1.4
    wstart: float = 0.9
    wend: float = 0.4
    alpha: float = 0.5
    beta0: float = 0.2
    gamma: float = 0.1
    seed: int = 42
    minfeatures: int = 10

class HFPSOFeatureSelector:
    def __init__(self, params: HFPSOParams):
        self.params = params
        self.rng = np.random.default_rng(self.params.seed)

    def fitness(self, mask):
        if mask.sum() == 0:
            return 1.0
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
        clf.fit(X_train_feats[:, mask], np.argmax(y_train, axis=1))
        y_pred = clf.predict(X_val_feats[:, mask])
        acc = accuracy_score(np.argmax(y_val, axis=1), y_pred)
        return 1 - acc

    def fit(self, X, y):
        self.dim = X.shape[1]
        positions = self.rng.uniform(0, 1, size=(self.params.nparticles, self.dim))
        velocities = self.rng.uniform(-1, 1, size=(self.params.nparticles, self.dim))
        pbest_pos = positions.copy()
        pbest_scores = np.array([self.fitness(pos > 0.5) for pos in positions])

        gbest_idx = np.argmin(pbest_scores)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        for iter_num in range(self.params.maxiter):
            w = self.params.wstart + (self.params.wend - self.params.wstart) * iter_num / self.params.maxiter
            for i in range(self.params.nparticles):
                r1, r2 = self.rng.random(), self.rng.random()
                velocities[i] = (w * velocities[i] +
                                 self.params.a1 * r1 * (pbest_pos[i] - positions[i]) +
                                 self.params.a2 * r2 * (gbest_pos - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                mask = positions[i] > 0.5

                if mask.sum() < self.params.minfeatures:
                    min_feat_idx = np.argsort(positions[i])[-self.params.minfeatures:]
                    mask[min_feat_idx] = True

                score = self.fitness(mask)
                if score < pbest_scores[i]:
                    pbest_pos[i] = positions[i].copy()
                    pbest_scores[i] = score

            gbest_idx = np.argmin(pbest_scores)
            gbest_pos = pbest_pos[gbest_idx].copy()
            gbest_score = pbest_scores[gbest_idx]
            print(f"Iter {iter_num+1}/{self.params.maxiter}, Best fitness: {1-gbest_score:.4f}")

        final_mask = gbest_pos > 0.5
        if final_mask.sum() < self.params.minfeatures:
            min_feat_idx = np.argsort(gbest_pos)[-self.params.minfeatures:]
            final_mask[min_feat_idx] = True

        return final_mask

params = HFPSOParams()
selector = HFPSOFeatureSelector(params)
selected_mask = selector.fit(X_train_feats, y_train)
np.save("fine_tuned_features_mask.npy", selected_mask)
print(f"Selected {selected_mask.sum()} features out of {len(selected_mask)}")

# ======================================================
# MODEL COMPARISON AFTER FEATURE SELECTION
# ======================================================
def make_classifier(name):
    name = name.lower()
    if name == 'svm':
        return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))
    elif name == 'ksvm':
        return make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, C=1.0, gamma='scale'))
    elif name == 'rf':
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    elif name == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    elif name == 'nn':
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42))
    else:
        raise ValueError(f"Unknown model {name}")

X_train_sel = X_train_feats[:, selected_mask]
X_val_sel = X_val_feats[:, selected_mask]
X_test_sel = X_test_feats[:, selected_mask]

model_names = ['svm', 'rf', 'knn', 'ksvm', 'nn']
results = []

for model_name in model_names:
    clf = make_classifier(model_name)
    start = time.time()
    clf.fit(X_train_sel, np.argmax(y_train, axis=1))
    duration = time.time() - start

    y_val_pred = clf.predict(X_val_sel)
    y_test_pred = clf.predict(X_test_sel)

    val_acc = accuracy_score(np.argmax(y_val, axis=1), y_val_pred)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(np.argmax(y_val, axis=1), y_val_pred, average='weighted', zero_division=0)

    test_acc = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(np.argmax(y_test, axis=1), y_test_pred, average='weighted', zero_division=0)
    test_cm = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)

    results.append({
        'model': model_name,
        'val_accuracy': val_acc,
        'val_precision': val_prec,
        'val_recall': val_rec,
        'val_f1': val_f1,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'test_confusion_matrix': test_cm,
        'train_time_sec': duration,
        'selected_features': int(selected_mask.sum())
    })

results_df = pd.DataFrame(results)
results_df.to_csv("fine_tuned_model_results.csv", index=False)
print("Saved all model results to 'fine_tuned_model_results.csv'")

best_row = results_df.loc[results_df['test_accuracy'].idxmax()]
print(f"🏆 Best Model After HFPSO: {best_row['model']}")
print(f"Validation Accuracy: {best_row['val_accuracy']:.4f}")
print(f"Test Accuracy: {best_row['test_accuracy']:.4f}")

# ======================================================
# FINAL COMPARISON: BASELINE vs HFPSO
# ======================================================
print("\n📈 ======= BASELINE vs HFPSO COMPARISON =======")
for model in baseline_df['model']:
    base_acc = baseline_df.loc[baseline_df['model'] == model, 'test_accuracy'].values[0]
    if model in results_df['model'].values:
        hfpso_acc = results_df.loc[results_df['model'] == model, 'test_accuracy'].values[0]
        diff = hfpso_acc - base_acc
        print(f"{model.upper():<6} | Baseline: {base_acc:.4f} | HFPSO: {hfpso_acc:.4f} | Δ = {diff:+.4f}")

print("===============================================")

# Plot Confusion Matrix of Best HFPSO Model
best_cm = best_row['test_confusion_matrix']
plt.figure(figsize=(6, 5))
plt.imshow(best_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - Best Model: {best_row['model']}")
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')

thresh = best_cm.max() / 2.
for i, j in np.ndindex(best_cm.shape):
    plt.text(j, i, format(best_cm[i, j], 'd'),
             ha='center', va='center',
             color='white' if best_cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig("best_confusion_matrix.png")
plt.close()
