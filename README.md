Final Project for Dr. Bruce Donald's "Computational Biology Seminar" at Duke University 2019. This was my first foray into studying proteins including modern challenges in folding and design. Final presentation here: https://docs.google.com/presentation/d/1Czgu30rfdBjQuM1SJKkvwG54b05p5bEazRBvFeRRrJI/edit?usp=sharing and write up: 

Simultaneously reading through Ian Goodfellow's Deep Learning textbook, I wanted to apply what I was learning to try and do protein design. 

I learnt a huge amount about Deep Learning, the state of the art of protein design, and Tensorflow/PyTorch. Ultimately, I never got the project working and to its full potential for three reasons. 1. It turns out the theoretical developments I was most excited about had already been done, almost exactly, through https://arxiv.org/pdf/1610.02415.pdf for small molecules. 2. I lacked the GPU hardware needed to test enough hyperparameters with large enough models to get the network to start learning. 3. I started working in Dr. Debora Marks's lab on other projects.  

Taking some of this code may be of interest to you, reader, if you are looking for PyTorch dataloaders for ProteinNet https://github.com/aqlaboratory/proteinnet or a really cool interface to see how your model is learning (largely taken from https://github.com/biolib/openprotein) 
