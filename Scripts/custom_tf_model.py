import tensorflow as tf
from keras.layers import Layer, Dropout
from keras import Model
from keras.utils import to_categorical
#from tf.keras.layers import Layer
#from tf.keras import Model
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

class Custom_Dense(Layer):
    def __init__(self, units):
        super(Custom_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class Custom_Model(Model):
    def __init__(self):
        super(Custom_Model, self).__init__()

        self.dense1 = Custom_Dense(32)
        self.dense2 = Custom_Dense(64)
        self.dense3 = Custom_Dense(3)
        self.dropout = Dropout(0.25)

    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        x = tf.nn.relu(x)

        if training:
            x = self.dropout(x, training=training)

        x = self.dense2(x)
        x = tf.nn.relu(x)

        x = self.dense3(x)
        x = tf.nn.softmax(x)

        return x

model = Custom_Model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size = 32, epochs=10)

results = model.evaluate(X_test, y_test)

print(f'test set evaluation: {results}')
