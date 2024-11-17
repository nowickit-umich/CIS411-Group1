import numpy as np
import random

# Load the .npy file
file_path = 'WP-train.npy'
data = np.load(file_path, allow_pickle=True)

# Baseline
count = 0
correct = 0
for item in data:
    count += 1
    guess = item["choice_list"][random.randint(0, 2)]
    if guess == item["answer"]:
        correct += 1
    
print(correct / count)
