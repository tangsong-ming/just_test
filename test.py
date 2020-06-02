import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

f=np.load(r'C:\Users\VULCAN\Desktop\ten\mnist.npz')
x_train,y_train=f['x_train'],f['y_train']
x_test,y_test=f['x_test'],f['y_test']
image_index=100
# plt.imshow(x_train[image_index],cmap='Greys')
# plt.show()
# print(y_train[image_index])
x_train=np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)#常数0
x_train=x_train.astype('float32')
x_train/=255#正则化
x_train=x_train.reshape(x_train.shape[0],32,32,1)#num height width channel
print(x_train.shape)

x_test=np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)#常数0
x_test=x_test.astype('float32')
x_test/=255#正则化
x_test=x_test.reshape(x_test.shape[0],32,32,1)#num height width channel

#################
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='valid',activation=tf.nn.relu,input_shape=(32,32,1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation=tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='same'),
    tf.keras.layers.Flatten(),#展平cheng一维
    #全连接层 输出维数 激活函数 步长
    tf.keras.layers.Dense(units=120,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=84,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)

])
#model.summary()
num_epochs=10
batch_size=64
learning_rate=0.001

adam_optimizer=tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizers=adam_optimizer,loss=tf.keras.losses.sparse_categorical_crossentropy,
		metrics=['accuracy'])
import datetime
start_time=datetime.datetime.now()
model.fit(x=x_train,y=y_train,batch_size=batch_size,
		epochs=num_epochs)
#######

end_time=datetime.datetime.now()
time_cost=end_time-start_time
print('time_cost=',time_cost)
model.save(r'C:\Users\VULCAN\Desktop\ten\lenet_model.h5')
print(model.evaluate(x_test,y_test))
#预测
image_index=4444
plt.imshow(x_test[image_index].reshape(32,32),cmap='Greys')
plt.show()
pred=model.predict(x_test[image_index].reshape(1,32,32,1))
print(pred.argmax())



