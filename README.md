# Generative Adversarial Learning of Sinkhorn Algorithm Initializations

Welcome to the repository of the paper 'Generative Adversarial Learning of Sinkhorn Algorithm Initializations'.

The paper aims at warm-starting the Sinkhorn algorithm with initializations computed by a neural network, which is trained in an adversarial fashion similar to a GAN using a second, generating neural network.
It is based on the Master's thesis 'A Sinkhorn-NN Hybrid Algorithm' by Jonathan Geuter, as well the follow up Master's Thesis 'Learning Optimal Transport Solutions with Deep neural networks' by Ingimar Tomasson.


# Reproduce Results

In order **to** reproduce the results of the paper it is as simple as executing the
`experiment.py` file. The required packages and their versions are detailed in the `requirements.txt` file.
The test data can be generated (scraped from online) by executing the `make_data.py` file or can be found at [this Google Drive folder](https://drive.google.com/drive/folders/1o5pz9-Zhr1-7s1QvPnOiTCVGWULmfC9f?usp=drive_link).

NB: This package is CUDA compatible 
