[Dataset]()
```python
from torch.utils.data import Dataset, DataLoader
```
```python
from torchvision.utils import save_image, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, ),  # 3 for RGB channels
                         std=(0.5, ))])
   
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

fake_images = generate_image.view(generate_image.size(0), 1, 28, 28)
save_image(denorm(fake_images.data), 'samples/test.png')
```