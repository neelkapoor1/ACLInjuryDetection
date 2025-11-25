# ACLInjuryDetection

Project: Detect ACL injury status from MRI volume `.pck` files using HOG features and a small CNN.

Repository structure (relevant files)
- `main.ipynb` — Jupyter notebook used for data loading, HOG extraction, sampling/augmentation, and training SVM, Logistic Regression and a small CNN.
- `metadata.csv` — mapping from `volumeFilename` → `aclDiagnosis` (labels).
- `archive/volXX/*.pck` — original MRI volumes (binary pickles).

Summary of reported run (from notebook output)
- Preprocessing: `Preprocessing done — 300 items (failed 0) in 133.3s` (≈0.44s per sample)
- Computed class weights: `{0: 1.0, 1: 1.0, 2: 1.0}`
- SVM accuracy: `0.5833` (60 test samples)
  - Class-level breakdown (precision / recall / f1 / support):
    - Class 0: prec 0.52 / rec 0.70 / f1 0.60 (support 20)
    - Class 1: prec 0.58 / rec 0.35 / f1 0.44 (support 20)
    - Class 2: prec 0.67 / rec 0.70 / f1 0.68 (support 20)
- Logistic Regression accuracy: `0.6333` (60 test samples)
  - Better balanced performance than SVM; Class 1 recall still low.
- Small CNN (trained 5 epochs with class weights) test accuracy: `0.3333` (random baseline for 3 classes)
  - Training accuracy around 0.30—model is not learning (validation stuck at 0.3333).

Quick interpretation
- Preprocessing succeeded for 300 items with no failures. The per-sample time indicates HOG extraction + resizing is reasonably expensive on CPU (133s total).
- Class weights are all `1.0` because the sampling strategy produced (near-)balanced class counts — when classes are balanced by oversampling/augmentation, sklearn's `compute_class_weight('balanced', ...)` will produce approximately equal weights (1.0). That means class imbalance is not the immediate cause of poor model performance here.
- SVM and Logistic Regression perform above chance (0.58 and 0.63 respectively). Logistic Regression shows better macro and weighted metrics, particularly for class 2.
- CNN performance at ~0.333 indicates the CNN is effectively guessing randomly. Training accuracy being ~0.30 and validation stable at 0.333 suggests the model is not learning useful discriminative patterns from the supplied input / labels.

Probable causes for the CNN failing to learn
- Small dataset size: 300 samples (after sampling/augmentation) is tiny for CNN training from scratch.
- Augmentation / oversampling strategy might have created near-duplicates or insufficient diversity.
- Model capacity mismatch: the CNN is small but with too few (or too noisy) signals to learn. Conversely, with tiny dataset, even a small CNN may overfit or fail to generalize.
- Preprocessing / normalization mismatch: ensure input images are in the expected numeric range (float32 in [0,1] is OK for Keras layers). The notebook currently uses float32 images scaled via `resize()` which yields values in [0,1]. Verify this in practice.
- Label issues: label encoding or mapping bugs can cause poor learning. Verify labels are correct and match the expected classes (0..n_classes-1) and that `y_img` values are correct.

Actionable suggestions (ordered)

1) Quick verification steps (fast, high-value)
- Confirm `y_img` values and `n_classes`:
  - Print `np.unique(y_img, return_counts=True)` before training to confirm class counts and encodings.
- Inspect a few pairs (image, label) visually to ensure labels match expected anatomy/diagnosis.
- Verify input ranges: print `Xtr_img.min(), Xtr_img.max(), Xtr_img.dtype`.

2) Faster feature-based improvements (low compute)
- Standardize / scale HOG features before SVM/LogReg: use `StandardScaler()` or `MinMaxScaler()` from scikit-learn. Many ML models (SVM, LogisticRegression) perform better on scaled features.
- Tune SVM hyperparameters (C, gamma with RBF) via a small grid search or randomized search with cross-validation. Example ranges: C ∈ [0.01, 0.1, 1, 10], gamma ∈ [1e-3, 1e-2, 1e-1, 1].
- Try dimensionality reduction (PCA) on HOG features to reduce noise before classification.
- Try oversampling techniques on the HOG feature vectors (SMOTE) if the HOG feature space is more effective for classical ML models.

3) CNN-focused improvements (medium compute)
- Increase dataset diversity and size:
  - Raise `MAX_SAMPLES` and/or produce more diverse augmentations (brightness, contrast, small zooms), but avoid exact duplicates.
  - Cache preprocessed images and HOG descriptors to disk (`np.savez_compressed`) to speed experimentation.
- Use transfer learning:
  - Convert slices to 3-channel (repeat grayscale into 3 channels) and fine-tune a pretrained model (e.g., `EfficientNetB0`, `MobileNetV2`) with the top layers retrained. Pretrained nets need more memory but often provide large accuracy gains on small datasets.
- Improve model training procedure:
  - Use `tf.keras.optimizers.Adam(learning_rate=1e-4)`.
  - Increase epochs to 20–50 with an early stopping callback monitoring validation loss / accuracy.
  - Use a lower batch size (8–16) for better gradient steps with small datasets.
  - Try focal loss (for class imbalance) or maintain `class_weight` if classes are unbalanced.
- Use a proper `tf.data.Dataset` pipeline so augmentations and shuffling are fast and deterministic.

4) Experimentation & model selection
- Use stratified k-fold cross-validation with classical models to get more reliable estimates.
- Evaluate confusion matrices and per-class ROC/AUC where reasonable (one-vs-rest) to better understand which classes are confused.
- Ensemble HOG-based classical model + CNN predictions (voting or stacking) to boost robustness.

5) Long-term / higher-effort options
- If the full 3D volume structure matters, explore 3D CNNs or architectures that accept the full volume rather than a single middle slice.
- Acquire more labeled data, or consider semi-supervised learning and self-supervised pretraining on the available volumes.

Concrete commands and code snippets
- Scale HOG features before SVM / LogisticRegression:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm.fit(X_tr_scaled, y_tr)
```

- Simple SVM grid search (use few folds):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('sc', StandardScaler()), ('svc', SVC(class_weight='balanced'))])
param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': [1e-3, 1e-2, 1e-1]}
gs = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=4)
gs.fit(X, y)
print(gs.best_params_, gs.best_score_)
```

- Transfer learning sketch (repeat grayscale -> 3-channel):

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False
inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp*255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(n_classes, activation='softmax')(x)
model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Recommendations tailored to these results
- Start with quick wins: scale HOG features, run SVM grid search, and compute confusion matrices. The classical models already beat random chance and are cheap to iterate.
- For CNNs, try transfer learning (pretrained 2D models) with repeated grayscale channels, or increase dataset diversity by more realistic augmentations and caching preprocessed images to disk for faster iteration.
- Double-check labels and preprocessing pipeline — CNN training stuck near random is often a data/label problem rather than just a model issue.

Next steps I can take for you
- Run the updated notebook cell and capture the exact printed diagnostics and timing (I can run it if you want). Note: ensure the notebook kernel is using the Python environment where TensorFlow is installed.
- Add HOG feature scaling and an SVM grid-search cell to the notebook and run it.
- Add a `cache_preprocessed/` helper to store and load precomputed HOG and image arrays to speed repeated experiments.

Contact / notes
- The notebook currently uses a balanced oversampling strategy that resulted in `class_weight` values of `1.0` across classes — this is expected when you sample to balance classes. If you want to emphasize the original dataset imbalance instead, sample proportionally to original counts and compute class weights accordingly.

— End of report —
