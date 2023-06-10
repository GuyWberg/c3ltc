## Setup

### Requirements
- Python >= 3.9.
- a C++ compiler supporting C++11 standard.
- You also need ``make`` tools. On macOS run:
```brew install autoconf automake libtool```
On ubunto run:
```sudo apt-get install autotools-dev```

After installing the requirements, run the following:
- Run ``sage -pip install -r requirements.txt``. 
- Run ``sh setup_spasm.sh`` in the cloned directory.  

This implementation uses [spasm](https://github.com/cbouilla/spasm) library for sparse finite fields matrix computation to improve the runtime. 