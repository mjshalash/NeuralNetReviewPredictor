import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load in data
data = keras.datasets.imdb


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# Loop to find most accurate model
max_acc = 0
best_models = []
for x in range(1):
    # Split into training and testing data
    # Only load in 10,000 most frequent words
    (train_data, train_labels), (test_data,
                                 test_labels) = data.load_data(num_words=88000)

    # Create Word Mappings using already created dataset dictionary
    # get_word_index() returns a tuple
    word_index = data.get_word_index()

    # Tuple is then split into key and value
    # Starting out at 3 as we will have three characters which are special
    word_index = {k: (v+3) for k, v in word_index.items()}

    # Padding to make each review same length
    word_index["<PAD>"] = 0

    # Auto added to beginning
    word_index["<START>"] = 1

    # Represents unknown items
    word_index["<UNK>"] = 2

    word_index["<UNUSED>"] = 3

    # Swap all values and keys (currently keys = words)
    # We want the integer to be the key and it point to words
    reverse_word_index = dict([(value, key)
                               for (key, value) in word_index.items()])

    # Each review in test and train data is a different length thus,
    # going to normalize all data items to arbitrary length of 250
    # by adding padding to the end ("post") if needed
    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data, value=word_index["<PAD>"], padding="post", maxlen=250)

    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

    # Decode data into human readable words
    # Get the index or a question mark if cannot find value

    # Define Model
    model = keras.Sequential()

    # Add layers
    model.add(keras.layers.Embedding(98000, 16))
    # model.add(keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    # "Smooshes" every output value between 0 and 1
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    # Binary Cross entropy works well here as we have two possible output
    # values, 0 and 1 (Good and Bad)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Split training data into two sets

    # Validation Data (to check how well program is performing as we tweak and tune training data)
    # Use first 10000 for train data
    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    # NOTE: We have not touched test_data

    # Fit Model aka train this bad boi
    # Batch Size = How many movie reviews to load in at one time
    fit_model = model.fit(x_train, y_train, epochs=40,
                         batch_size=512, validation_data=(x_val, y_val), verbose=1)

    # Now use test data to see how well ole boy did
    results = model.evaluate(test_data, test_labels)

    # Print out the accuracy of model
    if(results[1] > max_acc):
        best_models.append(results[1])
        print("Max %s: %0.4f%%" %
              (model.metrics_names[1], results[1]))
        max_acc = results[1]
        top_model = fit_model


# Create graph from training of the best model
history_dict = fit_model.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


# Display Loss over Time
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Display Accuracy over Time
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save model to avoid retraining
# .h5 is the file extension for a saved model
# model.save("model.h5")


# SECTION: READING IN CUSTOM TEXT FILE

# Return encoded list per line from text file
# def encode_review(s):
#     # Setting starting point
#     encoded = [1]

#     for word in s:
#         if word.lower() in word_index:  # We use word_index and not reverse word index because we are trying to ENCODE words, not DECODE numbers
#             encoded.append(word_index[word.lower()])
#         else:                   # Add unknown tag if word is not found
#             encoded.append(2)
#     return encoded


#     # Load our saved model
# model = keras.models.load_model("model.h5")

# # Load in text file and load into proper form
# with open("reviews/badRevGodfather.txt", encoding="utf-8") as f:
#     # Preprocess each line as we load into model
#     for line in f.readlines():
#         # Remove puncuation marks because otherwise ("Company," does not map to real word)
#         # Then split up each word by space
#         nline = line.replace(",", "").replace(".", "").replace(
#             "(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")

#         # Look up mapping and return coded list
#         encode = encode_review(nline)

#         # Apply Keras precprocessing such as UNK and PAD tags
#         # Note encode is in brackets as this expects a list of lists
#         encode = keras.preprocessing.sequence.pad_sequences(
#             [encode], value=word_index["<PAD>"], padding="post", maxlen=250)

#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print()
#         print()
#         print(predict[0])

#     # Test Model with outside data

#     # Print out index 0 to see comparison
#     # test_review = test_data[0]
#     # predict = model.predict([test_review])
#     # print("Review: ")
#     # print(decode_review(test_review))
#     # print("Prediction: " + str(predict[0]))
#     # print("Actual: " + str(test_labels[0]))
#     # print(results)
