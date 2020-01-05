import tensorflow as tf
import tensorflow.keras.backend as kb
import numpy as np

# This is an ultra simple model to learn squares of numbers.
# Do not take the model too seriosuly, it will overfit and is only 
# for deminstration purpose
keras_model=tf.keras.Sequential([ 
    tf.keras.layers.Dense(32,activation=tf.nn.relu,input_shape=[1]),
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
]
)

# Now we define our custom loss function

def custom_loss(y_actual,y_pred): 
    custom_loss=kb.square(y_actual-y_pred)
    return custom_loss

optimizer=tf.keras.optimizers.RMSprop(0.001)
keras_model.compile(loss=custom_loss,optimizer=optimizer)

#Sample data
x=[1,2,3,4,5,6,7,8,9,10]
x=np.asarray(x).reshape((10,1))
y=[1,4,9,16,25,36,49,64,81,100]
y=np.asarray(y).reshape((10,1))
y=y.astype(np.float32)
keras_model.fit(x,y,batch_size=10,epochs=1000)
print(keras_model.predict([11]))