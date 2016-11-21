from keras.datasets import mnist

# from keras.models import Sequential
#
# model = Sequential()
#
# model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 60, 60)))

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train[2])
# print(X_test)
# print(y_train)
# print(y_test)
