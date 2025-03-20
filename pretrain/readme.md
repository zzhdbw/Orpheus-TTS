# Pretraining
## Overview
We find that trying to keep good semantic understanding of text boosts the models ability when speaking naturally and empathetically. We propose training the model on batches of speech and text. If you want the model to retain a large part of its text ability - i.e. function as an end-to-end speech model you should keep the ratio of text batch :speech batch as 2:1 to start and then gradually decrease to 1:1 throughout training. If your model is just trained for TTS start with 1:1 and gradually decrease to 0:1.


### Disclaimer

This code was copy and pasted into this repo quickly so there maybe bugs. The general outline should be pretty straightforward. It's also set up for multinode training.

Depending on how good the models reasoning abilities to be (and what specifically you want to retain), you can choose with text-based dataset you use. Using simple datasets with QA pairs (for finetuning like ) works pretty well. You can also try using wikipedia - to boost the 
