from pathlib import Path
import tensorflow as tf
from tensorflow import keras
MODEL_DIR=Path(__file__).parent/'models'; MODEL_DIR.mkdir(exist_ok=True); MODEL_PATH=MODEL_DIR/'mnist_cnn.keras'
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train.astype('float32')/255.; x_test=x_test.astype('float32')/255.; x_train=x_train[...,None]; x_test=x_test[...,None]
model=keras.Sequential([keras.layers.Input((28,28,1)),keras.layers.Conv2D(32,3,activation='relu'),keras.layers.MaxPooling2D(),keras.layers.Conv2D(64,3,activation='relu'),keras.layers.MaxPooling2D(),keras.layers.Flatten(),keras.layers.Dense(64,activation='relu'),keras.layers.Dropout(.25),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_split=.1,epochs=5,batch_size=128,verbose=1)
loss,acc=model.evaluate(x_test,y_test,verbose=0); print(f'MNIST test accuracy: {acc*100:.2f}%'); model.save(MODEL_PATH); print('Saved',MODEL_PATH)
