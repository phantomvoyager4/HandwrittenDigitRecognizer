from HDR import Layer, Activation, Softmax
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class App:
    def transfer_network(self, filepath = 'models_data_storage/model_1/model_1_95.28_0.5.npz'):
        #network structure transfer
        self.model = np.load(filepath)
        self.layer1 = Layer(n_inputs=784, n_neurons=128)
        self.layer2 = Layer(n_inputs=128, n_neurons=64)
        self.output_layer = Layer(n_inputs=64, n_neurons=10)
        self.activation = Activation()
        self.softmax = Softmax()
        trainable_layers = [self.layer1, self.layer2, self.output_layer]
        for index, layer in enumerate(trainable_layers):
            file_id = index + 1
            layer.weights = self.model[f'w{file_id}']
            layer.biases = self.model[f'b{file_id}']
        self.network = [self.layer1, self.activation, self.layer2, self.activation, self.output_layer, self.softmax]

    def __init__(self):
        self.transfer_network()
        self.app_window = Tk()
        self.app_window.geometry('400x550')
        self.app_window.title('Digit Recognizer AI')
        self.app_window.configure(bg='#f0f0f0')
        self.C = Canvas(self.app_window, height=300, width=300, bg='white', relief='groove', bd=2)
        self.C.pack(pady=20) 
        self.C.bind("<Button-1>", self.activate_event)       
        self.C.bind("<B1-Motion>", self.draw_line)
        self.textlabel = Label(self.app_window, text="Draw a digit...", font=("Helvetica", 20, "bold"), bg='#f0f0f0')
        self.textlabel.pack(pady=10)
        control_frame = Frame(self.app_window, bg='#f0f0f0')
        control_frame.pack(side='bottom', pady=30)
        Button(control_frame, text='Predict', command=self.predict_digit, font=("Helvetica", 11), width=10).grid(row=0, column=0, padx=10)
        Button(control_frame, text='Clear', command=self.clear_canvas, font=("Helvetica", 11), width=10).grid(row=0, column=1, padx=10)
        self.image = Image.new(mode='RGB', size=(300, 300), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        self.app_window.mainloop()

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
        self.image = Image.new(mode='RGB', size=(300, 300), color='white')
        self.draw = ImageDraw.Draw(self.image)


    def image_processing(self, raw_image):
        grayscaled = raw_image.convert('L')
        inverted = ImageOps.invert(grayscaled)

        centralize = inverted.getbbox()
        if centralize == None:
            return np.zeros(shape=(1, 784)) 
        cropped = inverted.crop(centralize)
        get_size = cropped.size
        scaler = max(get_size)
        newimage = Image.new(mode='L', size=[scaler,scaler], color=0)
        width = int((scaler - get_size[0]) / 2)
        height = int((scaler - get_size[1]) / 2)
        newimage.paste(cropped, [width,height])
        newimage = newimage.resize((20,20), resample= Image.LANCZOS)
        final_image = Image.new(mode='L', size=(28,28), color=0)
        final_image.paste(newimage, [4, 4])
        return ((np.array(final_image)) / 255.0).reshape(1, 784)

    def network_pipeline(self, image_vector):
        current_signal = image_vector
        for layer in self.network:
            iteration = layer.forward(current_signal)
            current_signal = iteration
        result_probability = np.max(current_signal)
        result = np.argmax(current_signal)
        return result, result_probability
        
    def predict_digit(self):
        vector = self.image_processing(self.image)       
        result, probability = self.network_pipeline(vector)
        self.textlabel.config(text=f'Prediction: {str(result)}, Certainty: {(probability*100):.2f}')


app1 = App()