import tensorflow as tf
K = tf.keras.backend
Lo = tf.keras.losses


y_pred = K.variable([0.7, 0.7, 0.7, 0.7, 0.7])
y_true = K.variable([1, 1, 1, 1, 1])
loss = K.eval(Lo.binary_crossentropy(y_true, y_pred))
print(loss)