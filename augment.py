from torchvision import  transforms
import random
import numpy as np

class RandAug:
    """Randomly chosen image augmentations."""

    def __init__(self, img_size, choice=None):
        # Augmentation options
        self.trans = ['identity', 'rotate', 'color', 'sharpness', 'blur', 'padding' ,'perspective']
        self.img_size = img_size
        self.choice = choice

    def __call__(self, img):
        if self.choice == None:
            # Weights set 40% probability for the 'identity' augmentation choice
            self.choice = random.choices(self.trans, weights=(40, 10, 10, 10, 10, 10, 10))[0]

        if self.choice == 'identity':
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor()
                        ])
            img = trans(img)

        elif self.choice == 'rotate':
            degrees = random.uniform(0, 180)
            rand_fill = random.choice([0,1])
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.RandomRotation(degrees, expand=True, fill=rand_fill),
                            transforms.Resize((self.img_size,self.img_size))
                        ])
            img = trans(img)

        elif self.choice == 'color':
            rand_brightness = random.uniform(0, 0.3)
            rand_hue = random.uniform(0, 0.5)
            rand_contrast = random.uniform(0, 0.5)
            rand_saturation = random.uniform(0, 0.5)
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.ColorJitter(brightness=rand_brightness, contrast=rand_contrast, saturation=rand_saturation, hue=rand_hue)
                        ])
            img = trans(img)

        elif self.choice=='sharpness':
            sharpness = 1+(np.random.exponential()/2)
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.RandomAdjustSharpness(sharpness, p=1)
                        ])
            img = trans(img)

        elif self.choice=='blur':
            kernel = random.choice([1,3,5])
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.GaussianBlur(kernel, sigma=(0.1, 2.0))
                        ])
            img = trans(img)

        elif self.choice=='padding':
            pad = random.choice([3,10,25])
            rand_fill = random.choice([0,1])
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.Pad(pad, fill=rand_fill, padding_mode='constant'),
                            transforms.Resize((self.img_size,self.img_size))
                        ])
            img = trans(img)

        elif self.choice=='perspective':
            scale = random.uniform(0.1, 0.5)
            rand_fill = random.choice([0,1])
            trans = transforms.Compose([
                            transforms.Resize((self.img_size,self.img_size)),
                            transforms.ToTensor(),
                            transforms.RandomPerspective(distortion_scale=scale, p=1.0, fill=rand_fill),
                            transforms.Resize((self.img_size,self.img_size))
                        ])
            img = trans(img)
            
        return img
