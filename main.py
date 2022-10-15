import torch
# discriminator와 generator의 architecture를 정의하기 위해 nn library 불러옴
import torch.nn as nn
# dataset인 MNIST를 불러오기 위해 torchvision library 불러옴
from torchvision import datasets
# 의도한대로 변형하여 전처리하기 위해 transforms library 불러옴
import torchvision.transforms as transforms
# 학습 과정에서 반복적으로 생성된 이미지를 출력하기 위해 save_image library 불러옴
from torchvision.utils import save_image

from multiprocessing import freeze_support


# image를 출력하기 위해 불러옴
from IPython.display import Image


# noise distribution의 dimension
latent_dim = 100


# 생성자(Generator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 하나의 블록(block) 정의
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                # 배치 정규화(batch normalization) 수행(차원 동일)
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 생성자 모델은 연속적인 여러 개의 블록을 가짐
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # batch size, number of channels, height, width를 이용해 G(z) 반환
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # sigmoid function은 확률값
            nn.Sigmoid(),
        )

    # 이미지에 대한 판별 결과를 반환
    def forward(self, img):
        # 하나의 vector로 flatten
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output


if __name__=='__main__': # 이미 실행된 함수가 다른 객체에 할당되어 실행될 때, 이전의 내용과 중복되어 실행되는 것을 막아 자원의 중복 사용을 막아주는 기능
    freeze_support() # python multiprocessing이 윈도우에서 실행될 경우, 자원이 부족할 경우를 대비해 파일 실행을 위한 자원을 추가해주는 역할
    transforms_train = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)



    # 생성자(generator)와 판별자(discriminator) 초기화
    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()

    # 손실 함수(loss function)
    adversarial_loss = nn.BCELoss()
    adversarial_loss.cuda()

    # 학습률(learning rate) 설정
    lr = 0.0002

    # 생성자와 판별자를 위한 최적화 함수
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))




    import time

    n_epochs = 200 # 학습의 횟수(epoch) 설정
    sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
    start_time = time.time()

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
            real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0) # 진짜(real): 1
            fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0) # 가짜(fake): 0

            real_imgs = imgs.cuda()

            """ 생성자(generator)를 학습합니다. """
            optimizer_G.zero_grad()

            # 랜덤 노이즈(noise) 샘플링
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cuda()

            # 이미지 생성
            generated_imgs = generator(z)

            # 생성자(generator)의 손실(loss) 값 계산
            g_loss = adversarial_loss(discriminator(generated_imgs), real)

            # 생성자(generator) 업데이트
            g_loss.backward()
            optimizer_G.step()

            """ 판별자(discriminator)를 학습합니다. """
            optimizer_D.zero_grad()

            # 판별자(discriminator)의 손실(loss) 값 계산
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # 판별자(discriminator) 업데이트
            d_loss.backward()
            optimizer_D.step()

            done = epoch * len(dataloader) + i
            if done % sample_interval == 0:
                # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 출력
                save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

        # 하나의 epoch이 끝날 때마다 로그(log) 출력
        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapse time: {time.time() - start_time:.2f}s]")




    Image('2000.png')
