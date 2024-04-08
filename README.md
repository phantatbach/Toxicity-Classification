# Toxicity-Classification

This is the finetuning code for toxicity classification of text from university domain.
I finetuned 3 model:
- ViSoBERT
- PhoBERT (without segmenting the data)
- PhoBERT (with the data segmented)

Note that using VNCoreNLP to segment the data will yield better results with PhoBERT since this is how the model was pre-trained.
