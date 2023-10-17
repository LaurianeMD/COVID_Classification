#Importing keras libraries and packages
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt


classifier = Sequential()
classifier.add(Conv2D(32, 3, 3, input_shape =(224, 224, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
classifier.add(Flatten())
classifier.add(Dense(256,activation = 'relu'))
classifier.add(Dense(1,activation = 'sigmoid'))

#Compiling CNN
classifier.compile(
    optimizer = 'adam',
    loss= 'binary_crossentropy',
    metrics = ['accuracy']
)


#Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split=0.2
)

training_set = datagen.flow_from_directory(
    'Xray_data',
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset="training"
)

test_set = datagen.flow_from_directory(
    'Xray_data',
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    class_mode='binary',
    subset="validation"
)
 

history = classifier.fit(
    training_set,
    epochs = 10,
    validation_data = test_set
)


classifier.save("models/Covid_class.h5")


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Importation des bibliothèques pour la matrice de confusion et le traçage
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Après avoir entraîné le modèle, effectuez des prédictions sur l'ensemble de test
y_pred = classifier.predict(test_set)
y_pred = (y_pred > 0.5)  # Convertissez les probabilités en classes (Vrai/Faux)
y_true = test_set.classes

# Calculez la matrice de confusion en utilisant les prédictions et les vraies étiquettes
cm = confusion_matrix(y_true, y_pred)

# Utilisez Seaborn pour créer un tracé de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


