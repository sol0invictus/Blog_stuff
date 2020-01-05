import tensorflow as tf
import numpy as np

#Build the model
class model:
    
    def __init__(self):
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[1])
        self.l2=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(1,kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adagrad(learning_rate=0.1)
        print(self.l1.variables)
    # Running the model
    def run(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return boom2
    #Custom loss fucntion
    def get_loss(self,X,Y):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return tf.math.square(boom2-Y)
    # get gradients
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]])
        return g
    # perform gradient descent
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        # print(self.var)
        self.train_op.apply_gradients(zip(g, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]]))

#Custom training
x=[1,2,3,4,5,6,7,8,9,10]
x=np.asarray(x,dtype=np.float32).reshape((10,1))
y=[1,4,9,16,25,36,49,64,81,100]
y=np.asarray(y,dtype=np.float32).reshape((10,1))
model=model()

for i in range(100):
    model.network_learn(x,y)
# Test Case
x=[11]
x=np.asarray(x,dtype=np.float32).reshape((1,1))
print(model.run(x))