import tensorflow

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

feature_vector_length = 784
num_classes = 10

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)
print("Quantidade de elementos de treino: {}".format(len(X_train)))
print("Quantidade de elementos de teste: {}".format(len(X_test)))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)
input_shape = (feature_vector_length,)
print(f'Feature shape: {input_shape}')

fig, ax = plt.subplots(2, 5)

ax[0,0].imshow(X_train[1].reshape(28,28), cmap=plt.cm.binary)
ax[0,1].imshow(X_train[6].reshape(28,28), cmap=plt.cm.binary)
ax[0,2].imshow(X_train[5].reshape(28,28), cmap=plt.cm.binary)
ax[0,3].imshow(X_train[7].reshape(28,28), cmap=plt.cm.binary)
ax[0,4].imshow(X_train[2].reshape(28,28), cmap=plt.cm.binary)

ax[1,0].imshow(X_train[0].reshape(28,28), cmap=plt.cm.binary)
ax[1,1].imshow(X_train[13].reshape(28,28), cmap=plt.cm.binary)
ax[1,2].imshow(X_train[15].reshape(28,28), cmap=plt.cm.binary)
ax[1,3].imshow(X_train[17].reshape(28,28), cmap=plt.cm.binary)
ax[1,4].imshow(X_train[4].reshape(28,28), cmap=plt.cm.binary)


modelo = Sequential()
modelo.add(Dense(350, input_shape=input_shape, activation='relu'))
modelo.add(Dense(50, activation='relu'))
modelo.add(Dense(num_classes, activation='softmax'))
modelo.summary()


modelo.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

historico = modelo.fit(X_train, Y_train, epochs=20, batch_size=100, validation_split=0.2)

teste_resultados = modelo.evaluate(X_test, Y_test, verbose=1)
print(f'Resultado nos testes - Erro: {teste_resultados[0]} - Acuracia: {teste_resultados[1]}%')


fig, ax = plt.subplots(1, 2, figsize=(16,8))

ax[0].plot(historico.history['loss'], label="Erro: Treinamento")
ax[0].plot(historico.history['val_loss'], label="Erro: Validação")
legend = ax[0].legend(loc='best')

ax[1].plot(historico.history['accuracy'], label="Acurácia: Treinamento")
ax[1].plot(historico.history['val_accuracy'], label="Acurácia: Validação")
legend = ax[1].legend(loc='best')

plt.show()
