import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image as im
import torchvision


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

z_dim = 64
image_dimension = 28*28

# Instantiate model and optimizer
model = Generator(z_dim,image_dimension)

# Load the trained model
model = torch.load("mnist_gan_generator8input.pt",map_location=torch.device('cpu') )


pp1=st.slider("p1",0.01,10.0)
pp2=st.slider("p2",0.01,10.0)
pp3=st.slider("p3",0.01,10.0)
pp4=st.slider("p4",0.01,10.0)
pp5=st.slider("p5",0.01,10.0)
pp6=st.slider("p6",0.01,10.0)
pp7=st.slider("p7",0.01,10.0)
pp8=st.slider("p8",0.01,10.0)

fixed_noise = torch.tensor([pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8])
model.eval()
fake = model(fixed_noise).reshape(-1, 1, 28, 28)
# print(fixed_noise)
data = fake.reshape(-1, 1, 28, 28)
# print(data)
# img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

# white_torch = torchvision.io.read_image('white_horse.jpg')

# plt.imshow(data.detach().squeeze().cpu())
# plt.show()
# imagee = im.fromarray(data.detach().squeeze().cpu().numpy())
# st.image(data.detach().squeeze().cpu(), caption='上传了核磁共振成像。', use_column_width=True)
# plt.show()

fig1 = plt.figure(figsize=(14,8))

fig1.suptitle("randomly generated digits")

plt.imshow(data.detach().squeeze().cpu(), 'gray')        

st.pyplot(fig1)