#Using transformers library to obtain GPT2 model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy

#Using pretrained data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")

#Getting file
file_path = 'data/WP-train.npy'
data = numpy.load(file_path, allow_pickle=True)

count = 0
correct = 0
for item in data:
    count += 1

    #Getting question and answer list from file
    question = item["question"]
    choices = str(item["choice_list"])

    #Creating a prompt using question and answer list
    prompt = f"Question: {question} Provide the letter of the correct choice from the following Choices: {choices}"

    #Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    #Generate a response
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  #Ensures proper attention
    )

    #Decode the response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    #Extract the answer part (Model responds with prompt before answer)
    answer = generated_text[len(prompt):].strip()

    #Note that the model sometimes starts stating the question at the end of its output, it doesn't seem to interfere with anything

    #Going to be slightly generous since the model is very inconsistent with the format of its output
    #Will search entire output for correct answer
    if item["answer"] in answer:
        correct += 1
        print("Correct")

print(correct/count)