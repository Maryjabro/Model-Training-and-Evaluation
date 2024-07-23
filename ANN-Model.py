import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus

# Read the dataset
col_names = ['gender', 'age', 'goes_to_movies', 'movie_intensity', 'movie_length_hr']
movies = pd.read_csv("movies.csv", header=None, names=col_names)

feature_cols = ['gender', 'age', 'movie_intensity', 'movie_length_hr']  # Including 'gender' as a feature
X = movies[feature_cols]  # Features
y = movies['goes_to_movies']  # Target variable: 'goes_to_movies'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Does not go to movies', 'Goes to movies'])
disp.plot(cmap=plt.cm.Blues, values_format='')
plt.title('Confusion Matrix')
plt.show()

# performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Visualize the decision tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['Does not go to movies', 'Goes to movies'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree.png')
Image(graph.create_png())

# Calculate training and testing MSE
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_mse = ((y_train_pred - y_train) ** 2).mean()
test_mse = ((y_test_pred - y_test) ** 2).mean()

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)

# ANN model training and evaluation
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("ANN Test Accuracy:", test_accuracy)

y_pred_ann = (model.predict(X_test_scaled) > 0.5).astype("int32")

cm_ann = confusion_matrix(y_test, y_pred_ann)

disp_ann = ConfusionMatrixDisplay(confusion_matrix=cm_ann, display_labels=['Does not go to movies', 'Goes to movies'])
disp_ann.plot(cmap=plt.cm.Blues, values_format='')
plt.title('ANN Confusion Matrix')
plt.show()

precision_ann = precision_score(y_test, y_pred_ann)
recall_ann = recall_score(y_test, y_pred_ann)
f1_ann = f1_score(y_test, y_pred_ann)

print("Precision:", precision_ann)
print("Recall:", recall_ann)
print("F1-score:", f1_ann)

# Plot training and validation loss and accuracy
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

max_val_acc = max(val_acc)
print("Max Validation Accuracy:", max_val_acc)
