import torch
from PIL import Image
from torchvision import transforms
from model import CatDogCNN

def predict_image(image_path, model_path = './model/cat_dog_cnn.pth'):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)            
        ]
    )

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    model = CatDogCNN()
    


    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        label = 'dog 🐶' if prob > 0.5 else 'cat 🐱'
        return label, prob