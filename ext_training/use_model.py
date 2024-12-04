import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('toxic_comment_model.h5')


input_data = np.array([[...]])

predictions = loaded_model.predict(input_data)
print(predictions)


test_data = np.array([...])
test_labels = np.array([...])


loss, accuracy = loaded_model.evaluate(test_data, test_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")
