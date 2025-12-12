from HDR import Layer, Activation, Softmax
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np

class App:
    def transfer_network(self, filepath = 'models_data_storage/model_1/model_1_95.28_0.5.npz'):
        #network structure transfer
        self.model = np.load(filepath)
        self.layer1 = Layer(n_inputs=784, n_neurons=128)
        self.layer2 = Layer(n_inputs=128, n_neurons=64)
        self.output_layer = Layer(n_inputs=64, n_neurons=10)
        self.layer1.weights = self.model['w1']
        self.layer1.biases = self.model['b1']
        self.layer2.weights = self.model['w2']
        self.layer2.biases = self.model['b2']
        self.output_layer.weights = self.model['w3']
        self.output_layer.biases = self.model['b3']
        self.activation = Activation()
        self.softmax = Softmax()

    def __init__(self):
        self.transfer_network()
        app_window = Tk()
        app_window.geometry('500x500')
        app_window.title('Digit recognizer')

        self.C = Canvas(app_window, height=300, width=300, bg='white')
        self.C.pack()
        self.C.config()
        self.C.bind("<Button-1>", self.activate_event)       
        self.C.bind("<B1-Motion>", self.draw_line)

        self.image = Image.new(mode='RGB', size=(300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        self.predict = Button(app_window, text='Predict', command=self.predict_digit)
        self.predict.pack()

        self.clear = Button(app_window, text='Clear', command=self.clear_canvas)
        self.clear.pack()

        self.textlabel = Label(app_window, text='Draw your digit')
        self.textlabel.pack()

        #run
        app_window.mainloop()

    def activate_event(self, event):
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        x = event.x
        y = event.y
        self.C.create_line(self.last_x, self.last_y, x, y,fill='black', width=18, capstyle=ROUND)
        self.draw.line(xy=[self.last_x, self.last_y, x, y], fill="black", width=18)
        self.last_x = x
        self.last_y = y

    def clear_canvas(self):
        self.C.delete('all')
        self.image = Image.new(mode='RGB', size=(300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)


    def image_processing(self, raw_image):
        image_resized = raw_image.resize((28,28), resample = Image.LANCZOS)
        image_grayscaled = image_resized.convert('L')
        image_array = np.array(image_grayscaled)
        reversed_array = 255 - image_array
        normalize = reversed_array / 255.0
        vectorized = normalize.reshape(1, 784)
        return vectorized

    def network_pipeline(self, image_vector):
        l1 = self.layer1.fpropagation(input = image_vector)
        a1 = self.activation.forward(l1)
        l2 = self.layer2.fpropagation(input = a1)
        a2 = self.activation.forward(l2)
        ol = self.output_layer.fpropagation(a2)
        softmaxed = self.softmax.forward(ol)
        result = np.argmax(softmaxed)
        return result

    def predict_digit(self):
        vector = self.image_processing(self.image)       
        result = self.network_pipeline(vector)
        self.textlabel.config(text=f'Prediction: {str(result)}')


app1 = App()