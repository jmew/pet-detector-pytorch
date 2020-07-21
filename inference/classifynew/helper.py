import os
import tempfile
import torch

from azure.storage.blob import BlobServiceClient
from torchvision import transforms
from PIL import Image

def get_model_from_az_storage():
    model_path = 'checkpoint.pth'

    # Get environment variable for Az Storage connection string to reference model
    if 'connect_str' in os.environ:
        connect_str = os.environ['connect_str']
    else:
        raise Exception('msg', 'connection string not found')

    # Get the model from Az Storage
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container='petdetector', blob='checkpoint.pth')

    with open(os.path.join(tempfile.gettempdir(), model_path), "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

    model = torch.load(os.path.join(tempfile.gettempdir(), model_path), map_location=torch.device('cpu'))
    model.eval()

    return model

# Get the classification labels based on the .txt file
def get_class_labels():
    try:
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, 'labels.txt'), 'r') as f:
            classes = f.read().splitlines() 
    except FileNotFoundError:
        raise

    return classes

def convert_image_to_tensor(model, image):
    input_image = Image.open(image).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    return input_batch

def get_model_prediction(model, input_batch):
    class_dict = get_class_labels()
    
    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it
    softmax = (torch.nn.functional.softmax(output[0], dim=0))
    out = class_dict[softmax.argmax().item()]

    return out