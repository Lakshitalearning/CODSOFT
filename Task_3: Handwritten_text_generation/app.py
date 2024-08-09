import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1529251333259-d36cccaf22ea?q=80&w=2060&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D.jpg");
    background-size: cover;
    height: 100vh;
    width: 100vw;
}
h1 {
    font-family: 'Pacifico', cursive;
    color: black;  /* Cream color */
    text-align: center;
    font-size: 3em;
}
.big-white-label {
    font-size: 2em; /* Adjust the font size as needed */
    color: white;   /* White color */
    text-align: center;
    display: block;
}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the pre-trained model
model = tf.keras.models.load_model('handwriting_model.h5')

# Character mapping (should be updated based on your actual character mapping)
char_mapping = {0: ' ', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

reverse_char_mapping = {v: k for k, v in char_mapping.items()}

# Global variables
mean = np.array([0.0, 0.0, 0.0])  # Replace with actual mean calculated from your dataset
std = np.array([1.0, 1.0, 1.0])   # Replace with actual std calculated from your dataset
epsilon = 1e-8
input_shape = (326, 3)  # Updated to match the training input shape

def softmax_with_temperature(logits, temperature=1.0):
    logits = np.array(logits) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def generate_sequence(model, start_sequence, length):
    generated_sequence = start_sequence.copy()
    for i in range(length):
        input_seq = pad_sequences([generated_sequence], maxlen=input_shape[0], padding='post', dtype='float32')
        input_seq = (input_seq - mean) / (std + epsilon)
        predicted_prob = model.predict(input_seq, verbose=0)[0, -1, :]
        
        # Use a temperature of 1.0 for sampling
        temperature = 0.1   
        scaled_prob = softmax_with_temperature(predicted_prob, temperature)
        predicted_class = np.argmax(scaled_prob)
        
        # Create a new stroke
        new_stroke = np.array([0, np.random.randn(), np.random.randn()])
        if predicted_class != 0:  # Avoid adding strokes with zero class
            new_stroke[0] = predicted_class
        
        generated_sequence = np.vstack([generated_sequence, new_stroke])
        
        # Optional: Limit the size of the generated sequence
        if generated_sequence.shape[0] > input_shape[0]:
            generated_sequence = generated_sequence[-input_shape[0]:]
    
    return generated_sequence

def preprocess_text_to_sequence(text, char_mapping, input_shape):
    sequence = [char_mapping.get(c, 0) for c in text]
    sequence_array = np.array(sequence)
    required_length = input_shape[0]
    num_features = input_shape[1]
    if len(sequence_array) < required_length:
        sequence_array = np.pad(sequence_array, (0, required_length - len(sequence_array)), mode='constant')
    elif len(sequence_array) > required_length:
        sequence_array = sequence_array[:required_length]
    if sequence_array.size % num_features != 0:
        sequence_array = np.repeat(sequence_array, num_features)[:required_length * num_features]
    reshaped_array = sequence_array.reshape((required_length, num_features))
    return reshaped_array

def draw_handwritten_text(strokes, img_size=(800, 200), scale=10):
  img = Image.new("RGB", img_size, (255, 255, 255))
  draw = ImageDraw.Draw(img)
  
  x, y = img_size[0] // 10, img_size[1] // 2
  pen_down = False
  
  for stroke in strokes:
    if stroke[0] == 0:  # Pen down
      pen_down = True
      dx, dy = stroke[1] * scale, stroke[2] * scale
      new_x, new_y = x + dx, y + dy
      draw.line((x, y, new_x, new_y), fill=0, width=2)
      x, y = new_x, new_y
    elif pen_down:  # Pen up, move without drawing
      x += stroke[1] * scale
      y += stroke[2] * scale
    else:  # Move to a new starting point
      x = stroke[1] * scale
      y = stroke[2] * scale
      pen_down = False
  
  return img

# Streamlit interface
st.markdown("<h1>Handwritten Text Generator</h1>",unsafe_allow_html=True)
st.markdown('<p class="big-white-label">Enter text to generate handwriting:</p>', unsafe_allow_html=True)

user_input = st.text_input(" ")
if user_input:
    input_sequence = preprocess_text_to_sequence(user_input, reverse_char_mapping, input_shape)
    generated_sequence = generate_sequence(model, input_sequence, length=100)
    
    # Debug: Print the generated sequence to verify
    st.write("Generated Sequence:", generated_sequence[:10])
    
    handwritten_img = draw_handwritten_text(generated_sequence)
    
    # Display the generated handwritten text as an image
    st.image(handwritten_img, caption="Generated Handwritten Text")
