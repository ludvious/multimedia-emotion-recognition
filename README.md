# multimedia-emotion-recognition

## Installation

1. Clone the repository (or download):
    ```
    git clone https://github.com/repository.git
    cd folder repository
    ```

2. Create virtual enviroment on root directory and install dependencies:
    ```
    python3 -m venv .venv
    source .venv/bin/activate                       (#Linux)
    .venv/Scripts/activate.bat                      (#Windows)
    ```
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

##  Run
```
flask --app app run
```
- Open the application in your browser at `http://localhost:5000`
