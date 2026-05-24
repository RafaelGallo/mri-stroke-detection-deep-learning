# MRI-stroke-detection-deep-learning

Deep learning pipeline for automated brain stroke detection and classification from MRI scans.
Implements VGG16 transfer learning with fine-tuning, Keras Tuner hyperparameter optimization,
Grad-CAM explainability, and a Conditional GAN (CGAN) for synthetic MRI generation.
Built with TensorFlow/Keras for clinical triage support.

## Models
- VGG16 Transfer Learning (Phase 1 + Phase 2 Fine-tuning)
- VGG16 + Keras Tuner Hyperband
- Conditional GAN (CGAN)

## Classes
- Haemorrhagic | Ischemic | Normal

## Results
- Test Accuracy: 96%
- AUC Ischemic: 1.000
- Ischemic Recall: 1.00 (zero missed cases)

## Tech Stack
Python | TensorFlow | Keras | Keras Tuner | Scikit-learn | OpenCV | Matplotlib
