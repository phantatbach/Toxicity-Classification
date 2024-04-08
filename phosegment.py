# Pho save the result for test set
# no segment = 0.0496 (CPU), 0.0084 (GPU)
# segment = 
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
import time
# import torch.nn.functional as F
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
model = RobertaForSequenceClassification.from_pretrained('/home4/bachpt/text_classification/Phobert_segment/saved_checkpoints/checkpoint-50170')

model.cuda()
model.eval()

data = pd.read_csv('/home4/bachpt/text_classification/mixed_test_lowered_shuffled_segmented.csv')

data_text = data['text']
total_time = 0
predictions = []
for datum in data_text:
    inputs = tokenizer(datum, return_tensors="pt", truncation=True, max_length=256).to('cuda')
    with torch.no_grad():
        start_time = time.time()
        logits = model(**inputs).logits
        end_time = time.time()

    predicted_class_id = logits.argmax().item()
    # softmax_values = F.softmax(logits, dim=1)
    
    # if predicted_class_id == 1:
        # if softmax_values[0][1] < 0.7:
            # predicted_class_id = 0

    # Save the predicted_class_id to the column result
    predictions.append(predicted_class_id)
    inference_time = end_time - start_time
    total_time += inference_time
data['predicted label'] = predictions
average_time = total_time / len(data_text)
print("Average Inference Time per Sample:", average_time)
data.to_csv('Pho_mixed_segmented_test_result_lowered_shuffled.csv', index = False)

# To get GPU usage
# Uncomment the following lines if you are using GPU
# import torch.cuda as cuda
# gpu_usage = cuda.max_memory_allocated()
# print("GPU Memory Usage:", gpu_usage)