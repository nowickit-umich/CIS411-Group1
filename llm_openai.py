from openai import OpenAI
import numpy as np
import time

# API key
file = open("openai.secret")
key = file.readline().strip()
client = OpenAI(api_key=key)
file.close()

# Load the .npy training file
file_path = 'WP-train.npy'
data = np.load(file_path, allow_pickle=True)

# 
count = 0
correct = 0
for item in data:
    count += 1
    # API is rate limited - batch mode could be used  
    time.sleep(0.1)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the multiple choice question by returning only the correct answer by itself. The answer is almost never None of the above."},
            {
                "role": "user",
                "content": item["question"] + " Select One: " + str(item["choice_list"])
            }
        ]
    )
    guess = response.choices[0].message.content
    print("GUESS:",guess,"ANSWER:",item["answer"])
    if guess == item["answer"]:
        correct += 1

print(correct/count)

