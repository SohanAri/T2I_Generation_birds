import os
import sys
import torch
import numpy as np
import random
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
import pickle

# Add the code directory to the path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Code"))
sys.path.insert(0, ROOT_PATH)

from lib.utils import tokenize, truncated_noise, prepare_sample_data, load_netG, get_tokenizer
from lib.perpare import prepare_models
from models.DAMSM import RNN_ENCODER
from models.GAN import NetG

app = Flask(__name__)

# Configuration
class Config:
    """Configuration class for the application"""
    def __init__(self, dataset="bird"):
        # General settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = 100
        self.truncation = True
        self.trunc_rate = 0.8
        
        # Text encoder settings
        self.TEXT = type('', (), {})()
        self.TEXT.EMBEDDING_DIM = 256
        self.TEXT.HIDDEN_DIM = 256
        self.TEXT.WORDS_NUM = 20
        
        # Set paths for bird dataset
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(project_dir, "Code/saved_models/birds")
        self.checkpoint = os.path.join(project_dir, "Code/saved_models/birds/pretrained/state_epoch_1220.pth")
        self.TEXT.DAMSM_NAME = os.path.join(project_dir, "Code/saved_models/birds/text_encoder200.pth")
        self.pickle_path = os.path.join(project_dir, "Code/saved_models/birds/captions_DAMSM.pickle")
        self.dataset = "bird"

# Initialize global variables
config = Config()
wordtoix = None
text_encoder = None
netG = None
models_loaded = False

def check_model_files():
    """Check if all required model files exist"""
    global config
    missing_files = []
    
    print("\nChecking model files:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Config dataset: {config.dataset}")
    
    # Check for word dictionary
    print(f"Checking pickle file: {config.pickle_path}")
    if not os.path.exists(config.pickle_path):
        missing_files.append(f"Word dictionary: {config.pickle_path}")
        print(f"Missing file: {config.pickle_path}")
    else:
        print(f"Found file: {config.pickle_path}")
    
    # Check for text encoder model
    print(f"Checking text encoder: {config.TEXT.DAMSM_NAME}")
    if not os.path.exists(config.TEXT.DAMSM_NAME):
        missing_files.append(f"Text encoder model: {config.TEXT.DAMSM_NAME}")
        print(f"Missing file: {config.TEXT.DAMSM_NAME}")
    else:
        print(f"Found file: {config.TEXT.DAMSM_NAME}")
    
    # Check for generator model
    print(f"Checking generator model: {config.checkpoint}")
    if not os.path.exists(config.checkpoint):
        missing_files.append(f"Generator model: {config.checkpoint}")
        print(f"Missing file: {config.checkpoint}")
    else:
        print(f"Found file: {config.checkpoint}")
    
    return missing_files

def load_models():
    """Load all required models"""
    global wordtoix, text_encoder, netG, config, models_loaded
    missing_files = check_model_files()
    
    if missing_files:
        print("Missing required model files:")
        for file in missing_files:
            print(f"- {file}")
        return False, missing_files
    
    try:
        print("Loading models from paths:")
        print(f"Pickle path: {config.pickle_path}")
        print(f"Text encoder: {config.TEXT.DAMSM_NAME}")
        print(f"Generator: {config.checkpoint}")
        
        # Load word dictionary
        with open(config.pickle_path, 'rb') as f:
            x = pickle.load(f)
            wordtoix = x[3]
            config.vocab_size = len(wordtoix)
            print(f"Loaded word dictionary with {config.vocab_size} words")
            del x
        
        # Load text encoder
        text_encoder = RNN_ENCODER(config.vocab_size, nhidden=config.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(config.TEXT.DAMSM_NAME, map_location="cpu")
        text_encoder.load_state_dict(state_dict)
        text_encoder.to(config.device)
        text_encoder.eval()
        print("Text encoder loaded successfully")
        
        # Load generator
        netG = NetG(32, config.z_dim, 256, 256, 3).to(config.device)
        netG = load_netG(netG, config.checkpoint, False, train=False)
        netG.eval()
        print("Generator loaded successfully")
        
        models_loaded = True
        return True, None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, [str(e)]

def tokenize_text(wordtoix, text_input):
    """Tokenize a text string directly (not from file)"""
    tokenizer = get_tokenizer()
    
    # Process the input text
    sentences = [text_input]  # Just one sentence in this case
    captions = []
    cap_lens = []
    new_sent = []
    
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('Empty tokens for sentence:', sent)
            continue
            
        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        
        if len(rev) == 0:
            print('No valid tokens found in sentence:', sent)
            continue
            
        captions.append(rev)
        cap_lens.append(len(rev))
        new_sent.append(sent)
    
    if len(captions) == 0:
        raise ValueError("Could not tokenize the input text. Please try a different description.")
        
    return captions, cap_lens, new_sent

def generate_image(text):
    """Generate an image from the input text"""
    global wordtoix, text_encoder, netG, config
    
    try:
        print(f"Generating image for text: {text}")
        
        # Tokenize the input text directly
        captions, cap_lens, _ = tokenize_text(wordtoix, text)
        print(f"Tokenized text with length: {cap_lens}")
        
        # Prepare the text embedding
        sent_embs, _ = prepare_sample_data(captions, cap_lens, text_encoder, config.device)
        
        # Generate noise
        if config.truncation:
            noise = truncated_noise(1, config.z_dim, config.trunc_rate)
            noise = torch.tensor(noise, dtype=torch.float).to(config.device)
        else:
            noise = torch.randn(1, config.z_dim).to(config.device)
        
        # Generate the image
        with torch.no_grad():
            sent_emb = sent_embs[0].unsqueeze(0)
            fake_img = netG(noise, sent_emb)
            
        # Convert the tensor to a PIL image
        fake_img = fake_img.detach().cpu()
        fake_img = (fake_img + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        fake_img = fake_img[0].permute(1, 2, 0).numpy()
        fake_img = (fake_img * 255).astype(np.uint8)
        
        print("Image generated successfully")
        return Image.fromarray(fake_img)
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/')
def index():
    """Render the index page"""
    global models_loaded, config
    
    missing_files = []
    instructions = ""
    
    if not models_loaded:
        missing_files = check_model_files()
        
        if missing_files:
            instructions = """
            <h5>Download Instructions for Bird Dataset:</h5>
            <p>Please download the following files and place them in the correct directories:</p>
            <ol>
                <li>Download the text encoder model (<code>text_encoder200.pth</code>) and place it in <code>Code/saved_models/birds/</code></li>
                <li>Download the generator model (<code>state_epoch_1220.pth</code>) and place it in <code>Code/saved_models/birds/pretrained/</code></li>
                <li>Download the word dictionary (<code>captions_DAMSM.pickle</code>) and place it in <code>Code/saved_models/birds/</code></li>
            </ol>
            <p>You can download these files from the <a href="https://github.com/tobran/DF-GAN" target="_blank">DF-GAN GitHub repository</a>.</p>
            """
    
    return render_template('index.html', 
                          models_loaded=models_loaded,
                          missing_files=missing_files,
                          instructions=instructions)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate an image from text"""
    global models_loaded
    
    # Check if models are loaded
    if not models_loaded:
        success, error = load_models()
        if not success:
            return jsonify({'error': f"Failed to load models: {', '.join(error)}"})
    
    # Get text from request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'})
    
    text = data['text']
    if not text:
        return jsonify({'error': 'Text cannot be empty'})
    
    try:
        # Generate image
        image = generate_image(text)
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({'image': img_str})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Try to load models on startup
    load_models()
    
    # Start Flask app
    app.run(debug=True)
