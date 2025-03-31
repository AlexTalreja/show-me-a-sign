import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ASLRecognizer:
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = []
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build CNN model architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(26, activation='softmax')  # 26 letters in ASL
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self, dataset_path):
        """Load images from dataset, including flipped folders"""
        images = []
        labels = []
        class_names = sorted([c for c in os.listdir(dataset_path) if c.isalpha() and len(c) == 1])
        self.class_names = class_names  # Store only original classes

        for idx, class_name in enumerate(class_names):
            for suffix in ['', '_flipped']:
                class_path = os.path.join(dataset_path, f"{class_name}{suffix}")
                if not os.path.isdir(class_path):
                    continue

                for img_file in os.listdir(class_path):
                    if not (img_file.lower().endswith(('.png', '.jpg', '.jpeg'))):
                        continue

                    img_path = os.path.join(class_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        img = img / 255.0
                        images.append(img)
                        labels.append(idx)  # Same label whether flipped or not

        return np.array(images), np.array(labels)

    
    def preprocess_data(self, images, labels):
        """Preprocess the data and split into train/test sets"""
        # Reshape for CNN input
        images = images.reshape(images.shape[0], 64, 64, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, dataset_path, epochs=15, batch_size=32, save_path=None):
        """Train the model on the ASL dataset"""
        # Load and preprocess data
        print("Loading data...")
        images, labels = self.load_data(dataset_path)
        print(f"Loaded {len(images)} images across {len(self.class_names)} classes")
        
        X_train, X_test, y_train, y_test = self.preprocess_data(images, labels)
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Generate classification report
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix
        if save_path:
            plt.savefig(os.path.join(os.path.dirname(save_path), 'confusion_matrix.png'))
        
        plt.show()
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the training history
        if save_path:
            plt.savefig(os.path.join(os.path.dirname(save_path), 'training_history.png'))
        
        plt.show()
        
        # Save model if requested
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return history
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Try to load class names if they exist alongside the model
        class_names_path = os.path.join(os.path.dirname(model_path), 'class_names.txt')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} class names")
    
    def predict(self, image):
        """
        Predict the ASL letter from an image
        Returns letter and confidence level (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        if len(self.class_names) == 0:
            raise ValueError("Class names not available")
        
        # Preprocess the image
        processed_img = cv2.resize(image, (64, 64))
        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        processed_img = processed_img / 255.0
        processed_img = processed_img.reshape(1, 64, 64, 1)
        
        # Get prediction
        predictions = self.model.predict(processed_img)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        return self.class_names[predicted_idx], confidence
    
    def predict_top_k(self, image, k=3):
        """
        Predict the top-k most likely ASL letters from an image.
        Returns a list of (class_name, confidence) pairs, sorted by confidence desc.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        if len(self.class_names) == 0:
            raise ValueError("Class names not available")
        
        # Preprocess the image
        processed_img = cv2.resize(image, (64, 64))
        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        processed_img = processed_img / 255.0
        processed_img = processed_img.reshape(1, 64, 64, 1)
        
        # Get prediction (softmax output)
        predictions = self.model.predict(processed_img)[0]  # shape (26,)
        # Sort in ascending order, take top k in descending order
        top_k_indices = predictions.argsort()[-k:][::-1]
        
        top_k_list = [(self.class_names[idx], float(predictions[idx])) for idx in top_k_indices]
        return top_k_list
    
    def save_class_names(self, save_dir):
        """Save class names to a file"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(os.path.join(save_dir, 'class_names.txt'), 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")


def main():
    # Path to dataset
    dataset_path = "../data/aslwireframemodified"
    
    # Create model directory if it doesn't exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, "asl_model.h5")
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        recognizer = ASLRecognizer(model_path)
    else:
        print("Creating and training new model")
        recognizer = ASLRecognizer()
        recognizer.train(dataset_path, epochs=20, save_path=model_path)
        recognizer.save_class_names(model_dir)
    
    # Test with webcam (optional)
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Make a copy for prediction (grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        try:
            letter, confidence = recognizer.predict(gray)
            cv2.putText(img, f"{letter} ({confidence:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Prediction error: {e}")
        
        cv2.imshow("ASL Recognition Test", img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 