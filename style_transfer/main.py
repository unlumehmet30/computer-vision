import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert("RGB")
    if shape is not None:
        size = shape
    else:
        size = max(image.size)
        if size > max_size:
            size = max_size

    in_transform = transforms.Compose([
        transforms.Resize((size, size) if isinstance(size, int) else size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = in_transform(image).unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0,1)
    return image.permute(1,2,0).numpy()

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

class vggFeatures(nn.Module):
    def __init__(self):
        super(vggFeatures, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
        }

        self.model = nn.Sequential()
        for idx, layer in enumerate(vgg):
            self.model.add_module(str(idx), layer)
            if str(idx) in self.layers:
                pass

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def run_style_transfer(content_img, style_img, steps=2000, style_weight=1e6, content_weight=1):
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=1e-3)
    model = vggFeatures()

    for step in tqdm(range(steps)):
        target_features = model(target)
        content_features = model(content_img)
        style_features = model(style_img)

        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

        style_loss = 0
        for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
            target_feat = target_features[layer]
            style_feat = style_features[layer]

            target_gram = gram_matrix(target_feat)
            style_gram = gram_matrix(style_feat)

            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_loss

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Loss {total_loss.item():.2f}")

    return target

content = load_image("cat.jpeg")
style = load_image("style.jpg", shape=tuple(content.shape[-2:]))

output = run_style_transfer(content, style)

plt.figure(figsize=(6,6))
plt.imshow(im_convert(output))
plt.title("Result of Style Transfer")
plt.axis("off")
plt.show()







    
