
from flask import Flask, json, jsonify, request, make_response
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys
import io
import tensorflow as tf

app = Flask(__name__)

longitud, altura = 150, 150
modelo = 'modelo.h5'
pesos = 'pesos.h5'



class YourCustomAdamOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        super(YourCustomAdamOptimizer, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.iterations = self.iterations + 1
        lr = self.learning_rate * tf.math.sqrt(1 - tf.math.pow(self.beta_2, self.iterations)) / (1 - tf.math.pow(self.beta_1, self.iterations))
        ms = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        vs = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.weights = [self.iterations] + ms + vs

        updates = [tf.compat.v1.assign(m, self.beta_1 * m + (1 - self.beta_1) * g) for m, g in zip(ms, grads)]
        updates += [tf.compat.v1.assign(v, self.beta_2 * v + (1 - self.beta_2) * tf.math.square(g)) for v, g in zip(vs, grads)]
        updates += [tf.compat.v1.assign_sub(p, lr * m / (tf.math.sqrt(v) + self.epsilon)) for p, m, v in zip(params, ms, vs)]
        return updates

cnn = load_model(modelo, compile=False)
cnn.load_weights(pesos)



@app.route('/')
def Home():  
  return 'API to detect crash in buildings .. '

@app.route('/predict', methods=['POST'] )
def Predict():
  try:
    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file))    
    resizedImage = image.resize((longitud, altura))
    x = np.expand_dims(resizedImage, axis=0)   
    predictions = cnn.predict(x)  
    predicted_class = np.argmax(predictions, axis=1) 
    class_names = ['COLLAPSE', 'HEALTHY', 'PARTIAL']
    predicted_label = class_names[predicted_class[0]]
    res = jsonify(result=predicted_label, status = True)
    return make_response(res, 200)
  except :
    res = jsonify(status = False, message = str(sys.exc_info() ) )
    return make_response(res, 500)
 

if __name__ == '__main__':
    app.run()
