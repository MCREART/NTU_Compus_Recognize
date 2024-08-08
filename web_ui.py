import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import gradio as gr
from PIL import Image
import requests
import json
import gradio as gr

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
model = efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 9)  # Adjust the number of output classes as per your dataset
model.load_state_dict(torch.load('model.pth'))  # Load your trained model weights
model.eval()

# Define the class labels
class_labels = ["gate", "hive", "memorial", "华裔馆", "商学院", "教学楼一角", "校牌", "计算机与数据科学学院"]

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Apply the transform and add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_name = class_labels[class_idx]  # Map the index to the class name
    # return class_name  # Return the predicted class name
    return class_name, get_response(class_name)


api_url = "http://localhost:11434/api/chat"
def get_response(user_input):
    payload = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": f"介绍南洋理工{user_input}"
            }
        ],
        "stream": False
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        response_data = response.json()
        message_content = response_data.get("message", {}).get("content", "")
        return message_content
    else:
        return f"Failed to get response from API. Status code: {response.status_code}"
# Define the Gradio interface


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", "markdown"],
    live=True,
    title="Image Classification with EfficientNet"
)

# Launch the Gradio interface
interface.launch(share=True)
