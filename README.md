# multimedia-emotion-recognition

## Installation

1. Clone the repository (or download):
    ```
    git clone https://github.com/repository.git
    cd folder repository
    ```

2. Create virtual enviroment on root directory:
    ```
    python3 -m venv .venv
    source .venv/bin/activate                       (#Linux)
    .venv/Scripts/activate.bat                      (#Windows)
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### (Optional) Use app with face landmark detection
This need further installation for the OS before install the dependecies:
 - for Windows:
    ``cmake`` & ``Visual Studio`` for all packages and c++ compiler
 -  for Linux:
    ``cmake``
    ``gcc gcc-c++ make``
    ``boost boost-devel``
    ``python3-devel`` 

Then complete the installation with:
    ``pip install -r requirements.txt``

#### Troubleshooting:
-   If you encounter errors during compilation, ensure that all required development tools and libraries are installed (cmake, gcc, boost, etc.).
-   If you have a specific GPU setup and want to use CUDA, you'll need to configure dlib for GPU acceleration manually.

## Project Structure
```
│
├── app.py             # Main application script
├── services/          # Services application classes for handle the face/speech inference
├── templates/         # HTML templates
├── models/            # Deep learning models
├── preprocessing/     # Preprocessing classes and function for preprocessing data and create datasets
├── utils/             # Utility functions for general use
├── config.py          # Module for define application parameters
├── requirements.txt   # List of dependencies
└── README.md          # Project overview
```

## Datasets
- `FER2013` (Facial Expression Recognition 2013 Dataset). The dataset contains approximately 30,000 facial RGB images of different expressions with size restricted to 48×48, and the main labels of it can be divided into 7 types: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral. The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.
-

## Default Models
- For face recognition, a CNN with architecture like `ResNet50` was trained with the dataset `FER2013`.
- For speech recognition, a CNN with architecture like `VGG` was trained with a sperimental dataset manuel created.
## Setup the models
If you want to run your faces or speechs model, edit or add the path in the `config.py` file:
- ``FACE_MODEL_PATH = "face_model_path"``
- ``SPEECH_MODEL_PATH = "speech_model_path"``

##  Run
```
flask --app app run
```
- Open the application in your browser at `http://localhost:5000`

## Requirements
- Python 3.8+

## Disclaimer
This tool is for research purposes and should not be used for critical decisions requiring professional human evaluation.