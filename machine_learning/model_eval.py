import torch
import numpy as np

# Evaluate the model with 5 random test cases
with torch.no_grad():
    for i in range(5):
        # Get a random test case
        index = np.random.randint(0, len(test_inputs))
        input = test_inputs[index]
        output = test_outputs[index]

        # Make a prediction
        prediction = model(input)

        # Print the prediction and actual output
        print(f"Input: {input}")
        print(f"Output: {output}")
        print(f"Prediction: {prediction}")
        print(f"Accuracy: {1 - torch.mean(torch.abs(prediction - output) / torch.abs(output))}")
        print()