# Neural-network
My first neural network that inputs a 28x28 greyscale image and guesses the hand-drawn number in it.
***
## making ai.py
It is trained on the mnist dataset which has over 40,000+ images, calculates the best fit weights for the two layers containing 64 and 32 neurons respectively.
## mnis_model_weights.npz
contains the weights calculated during training after 90% accuracy.
## using ai.py
the drawing window of 28x28 is created here which calculates the drawn number using __mnis_model_weights.npz__, and displays it.
## quantize.py
since the mnist dataset is created using real photos, the arbitrary pixel values get quantized into the ones used in the drawing window.
***
