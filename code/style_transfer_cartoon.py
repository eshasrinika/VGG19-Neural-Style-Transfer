import torch
from torch import nn, optim
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Smaller size so it runs faster in the UI
imsize = 256

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


def save_image(tensor: torch.Tensor, path: str):
    img = tensor.detach().cpu().clone().squeeze(0)
    img = unloader(img)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


def get_vgg19():
    try:
        from torchvision.models import vgg19, VGG19_Weights
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    except Exception:
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
    return cnn


def build_model(cnn, style_img, content_img):
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.mean = mean.view(-1, 1, 1)
            self.std = std.view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4"]

    model = nn.Sequential(Normalization(norm_mean, norm_std))
    style_losses = []
    content_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target_feature = model(style_img).detach()
            sl = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)

    # trim after last loss layer
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            model = model[: j + 1]
            break

    return model, style_losses, content_losses


def run_style_transfer(
    content_path: str,
    style_path: str,
    output_path: str,
    num_steps: int = 100,
    style_weight: float = 8e4,
    content_weight: float = 20.0,
    progress=None,
):
    """
    content_path, style_path, output_path: file paths
    progress: optional gr.Progress object for UI
    """
    print("Loading images...")
    content_img = load_image(content_path)
    style_img = load_image(style_path)

    assert content_img.size() == style_img.size(), "Resize images to same size."

    input_img = content_img.clone()
    cnn = get_vgg19()
    model, style_losses, content_losses = build_model(cnn, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1

            # update progress bar every few steps
            if progress is not None and run[0] % 10 == 0:
                progress(run[0] / num_steps, desc="Applying style...")

            if run[0] % 20 == 0:
                print(
                    f"Step {run[0]}/{num_steps} | "
                    f"Style: {style_score.item():.4f} | "
                    f"Content: {content_score.item():.4f}"
                )
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    save_image(input_img, output_path)
    print(f"Saved stylized image to {output_path}")
