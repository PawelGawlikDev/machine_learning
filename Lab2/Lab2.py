# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# %%
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
print(mnist.data.shape)
print(mnist.target.shape)

# %%
# Split data
train_img, test_img, train_lbl, test_lbl = train_test_split(


    mnist.data, mnist.target, test_size=0.3, random_state=0, stratify=mnist.target)

print(train_img.shape)

# Valid split
test_img, val_img, test_lbl, val_lbl = train_test_split(
    test_img, test_lbl, test_size=0.5, random_state=0, stratify=test_lbl)

class_counts = np.bincount(val_lbl.astype(int))

for class_id, count in enumerate(class_counts):
    print(f"Klasa {class_id}: {count} wzorców")

# %%
# Plot example data
plt.figure(figsize=(20, 4))


for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):

    plt.subplot(1, 5, index+1)

    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)

    plt.title('Training: %s\n' % label, fontsize=20)

# %%
# Default
clf = MLPClassifier()

clf.fit(train_img, train_lbl)
predictions = clf.predict(test_img)
# Loss function plot
plt.plot(clf.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
plt.show()

# %%
# Predicts
predictions_train = clf.predict(train_img)

predictions_test = clf.predict(test_img)

train_score = accuracy_score(predictions_train, train_lbl)


print('Score on train data: ', train_score)

test_score = accuracy_score(predictions_test, test_lbl)


print('Score on test data: ', test_score)

# %%
# Five error predictions
index = 0

badIndex = 0

misclassifiedIndexes = []


for label, predict in zip(test_lbl, predictions):
    badIndex = badIndex+1
    if label != predict:
        misclassifiedIndexes.append(badIndex)
        print(misclassifiedIndexes[index], label,

              test_lbl[badIndex - 1], predict, predictions[badIndex - 1])
        index += 1

print(test_lbl[4], predictions[4])

# %%
# Plot error predictions
plt.figure(figsize=(20, 4))


for plotIndex, badIndex, in enumerate(misclassifiedIndexes[0:5]):
    print(badIndex, predictions[badIndex - 1], test_lbl[badIndex - 1])
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex-1], (28, 28)), cmap=plt.cm.gray)

    plt.title('Predicted: {}, Actual: {}'.format(

        predictions[badIndex - 1], test_lbl[badIndex-1]), fontsize=20)

# %%
# Confusion matri
ConfusionMatrixDisplay.from_predictions(predictions_test, test_lbl)
plt.show()

# %%
# sgd solver
clf = MLPClassifier(solver='sgd')

clf.fit(train_img, train_lbl)
predictions = clf.predict(test_img)
score = accuracy_score(predictions, test_lbl)
print('sdg score: ', score)
# Loss function plot
plt.plot(clf.loss_curve_)
plt.title('Loss Curve fro sgd solver')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
plt.show()

# %%
# Different neurons neurons
hidden_layer_sizes = [(20,), (300,)]
for h_layer_size in hidden_layer_sizes:
    clf = MLPClassifier(hidden_layer_sizes=h_layer_size)

    clf.fit(train_img, train_lbl)
    predictions = clf.predict(test_img)
    score = accuracy_score(predictions, test_lbl)
    print(f'{h_layer_size} score: ', score)
    # Loss function plot
    plt.plot(clf.loss_curve_)
    plt.title(f'Loss Curve for {h_layer_size}')
    plt.xlabel('Epoch number')
    plt.ylabel('Value of the loss function')
    ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
    plt.show()

# %%
# Disable momentum
clf = MLPClassifier(solver='sgd', momentum=0)

clf.fit(train_img, train_lbl)
predictions = clf.predict(test_img)
score = accuracy_score(predictions, test_lbl)
print('Disable momentum score: ', score)
# Loss function plot
plt.plot(clf.loss_curve_)
plt.title('Loss Curve for disable momentum')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
plt.show()

# %%
# max iteration
iteration = [10, 50, 500]
for max_i in iteration:
    clf = MLPClassifier(max_iter=max_i)

    clf.fit(train_img, train_lbl)
    predictions = clf.predict(test_img)
    score = accuracy_score(predictions, test_lbl)
    print(f'{max_i} score: ', score)
    # Loss function plot
    plt.plot(clf.loss_curve_)
    plt.title(f'Loss Curve for {max_i}')
    plt.xlabel('Epoch number')
    plt.ylabel('Value of the loss function')
    ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
    plt.show()

# %%
# early_stopping
clf = MLPClassifier(early_stopping=True)

clf.fit(train_img, train_lbl)
predictions = clf.predict(test_img)
score = accuracy_score(predictions, test_lbl)
print('Early stopping score: ', score)
# Loss function plot
plt.plot(clf.loss_curve_)
plt.title('Loss Curve for enable early_stopping')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
plt.show()

# %%
# Different activaction
clf = MLPClassifier(activation='logistic')

clf.fit(train_img, train_lbl)
predictions = clf.predict(test_img)
score = accuracy_score(predictions, test_lbl)
print('sigmoidal activation score: ', score)
# Loss function plot
plt.plot(clf.loss_curve_)
plt.title('Loss Curve for logistic activation')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
ConfusionMatrixDisplay.from_predictions(predictions, test_lbl)
plt.show()

# %%
# Hypoparameters to test
hidden_layer_sizes = [(50,), (100,), (300,)]
activation = ['identity', 'relu', 'tanh', 'logistic']
best_score = 0
best_params = None

# %%
for h_layer_size in hidden_layer_sizes:
    for act in activation:
        print(f'Training {h_layer_size} {act} combination')
        clf = MLPClassifier(hidden_layer_sizes=h_layer_size,
                            activation=act,)
        clf.fit(train_img, train_lbl)
        predictions_val = clf.predict(val_img)
        val_score = accuracy_score(predictions_val, val_lbl)
        if val_score > best_score:
            best_score = val_score
            best_params = {'hidden_layer_sizes': h_layer_size,
                           'activation': act}

print("Best scores: ", best_params)

# %%
# Train final model with the best params
final_clf = MLPClassifier(**best_params)
final_clf.fit(np.concatenate((train_img, val_img)),
              np.concatenate((train_lbl, val_lbl)))
final_prediction = final_clf.predict(val_img)
final_score = accuracy_score(final_prediction, val_lbl)
print('Final score: ', final_score)
plt.plot(final_clf.loss_curve_)
plt.title('Loss Curve for enable early_stopping')
plt.xlabel('Epoch number')
plt.ylabel('Value of the loss function')
ConfusionMatrixDisplay.from_predictions(final_prediction, val_lbl)
plt.show()
