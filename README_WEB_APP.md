# DF-GAN Web Application

This web application provides a user-friendly interface for the DF-GAN text-to-image synthesis model. Users can input text descriptions and generate corresponding images using the DF-GAN model.

## Features

- Simple web interface for text-to-image generation
- Real-time image generation from text descriptions
- Example text prompts for quick testing
- Responsive design that works on desktop and mobile devices

## Requirements

- Python 3.13.0
- PyTorch 2.2.0 or higher
- Flask 2.3.0 or higher
- All dependencies listed in `requirements.txt`

## Setup Instructions

1. Make sure you have Python 3.13.0 installed.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the pretrained model files:
   - Download [DF-GAN for bird](https://drive.google.com/file/d/1rzfcCvGwU8vLCrn5reWxmrAMms6WQGA6/view?usp=sharing) or [DF-GAN for coco](https://drive.google.com/file/d/1e_AwWxbClxipEnasfz_QrhmLlv2-Vpyq/view?usp=sharing)
   - Create the directory structure if it doesn't exist:
     ```
     mkdir -p code/saved_models/coco/pretrained
     ```
   - Save the downloaded model to `code/saved_models/coco/pretrained/state_epoch_290.pth`

4. Download the text encoder model:
   - Create the directory structure:
     ```
     mkdir -p data/coco/DAMSMencoder
     ```
   - Download the text encoder model and save it to `data/coco/DAMSMencoder/text_encoder100.pth`

5. Download the word dictionary:
   - Create the directory structure:
     ```
     mkdir -p data/coco
     ```
   - Download the preprocessed metadata for [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing)
   - Extract it to `data/coco/`

## Running the Web Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter a text description in the input field and click "Generate Image" to create an image based on your description.

## Example Text Prompts

- "A beautiful bird with blue wings perched on a tree branch"
- "A vase with colorful flowers on a wooden table"
- "A pizza with cheese, mushrooms and pepperoni on a plate"
- "A dog running in a green field on a sunny day"

## Troubleshooting

If you encounter any issues:

1. Make sure all required model files are in the correct locations
2. Check that you're using Python 3.13.0 and have installed all dependencies
3. Look for error messages in the terminal where you started the Flask application

## Credits

This web application is built on top of the [DF-GAN](https://github.com/tobran/DF-GAN) model by Ming Tao, Hao Tang, Fei Wu, Xiao-Yuan Jing, Bing-Kun Bao, and Changsheng Xu.

## License

The code is released for academic research use only. For commercial use, please contact [Ming Tao](mingtao2000@126.com).
