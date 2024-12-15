import requests
import json
import numpy as np

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to query the Ollama model
def query_ollama_model(question, choices):
    try:
        prompt = "<|start_header_id|>system<|end_header_id|>Select the correct answer to the question from the choice list. Respond with only the correct answer.<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id|>Question: {question}\nChoice List: {choices}<|eot_id|>"

        # Payload for the API request
        payload = {
            "model": "gemma2",
            "prompt": prompt,
            "stream": False
        }

        # Send the request to the API
        response = requests.post(
            OLLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        # Handle the response
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response field in API output.")
        else:
            return f"Error: Received status code {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"

def main():
    # load data
    file_path = 'train.npy'
    data = np.load(file_path, allow_pickle=True)

    count = 0
    correct = 0
    for item in data:
        response = query_ollama_model(item["question"], item["choice_list"])
        print(f"RESPONSE: {response}")
        count += 1
        if response.strip().strip("'") == item["answer"].strip():
            correct += 1
            print("CORRECT!")
        else:
            print("WRONG!")

    print(f"ACCURACY: {correct / count}")

    return 0

if __name__ == "__main__":
    main()
