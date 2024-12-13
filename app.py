import gradio as gr
import pickle as pkl

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Feature Extracting model
vgg_model = VGG16()
vgg_model.trainable = False

img_model = Model(inputs=vgg_model.input,
                  outputs=vgg_model.layers[-2].output)

# Caption genartion model
model = tf.keras.models.load_model('caption_genaration_model.h5')

# load Tokenizer
with open('tokenizer.pkl','rb') as f:
    tokenizer = pkl.load(f)

# convert index to word from prediction
def index_to_word(word_idx):
    return tokenizer.index_word[word_idx]

# Resize layer
resize_img = tf.keras.layers.Resizing(height=224, width=224)

# Preprocces input Image
def img_preprocces(img): 
    img = tf.expand_dims(img,axis=0)
    resized_image = resize_img(img)
    img = preprocess_input(resized_image)
    feature = vgg_model.predict(img,verbose=False)
    return feature

def genarate_caption(img):
    seq_in = 'startseq'
    feature_img = img_preprocces(img)

    for _ in range(30):
        # Tokenization & Padding
        seq_in_sequence = tokenizer.texts_to_sequences([seq_in])[0]
        seq_in_padded = pad_sequences([seq_in_sequence], padding='post',maxlen=30)

        # Predict next word
        y_hat = model.predict([feature_img,seq_in_padded],verbose=False)
        word_index = y_hat.argmax(axis=1)
        predicted_word = index_to_word(word_index[0])
        if predicted_word == 'endseq':
            break
        seq_in = seq_in + ' ' + predicted_word 
        
    
    return seq_in[9:]

app = gr.Interface(
    fn=genarate_caption,
    inputs=['image'],
    outputs=['text']
)

app.launch()