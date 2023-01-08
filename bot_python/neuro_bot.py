from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
import logging
from aiogram.utils import executor
import os
from PIL import Image
import json
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from translate import Translator


translator= Translator(to_lang="Russian")

token = '5488268409:AAGv58AEXlQo9K7qjZ3Jcef_RhiZVZZYJRM'

bot = Bot(token=token)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)



def get_idx_to_label():
    with open("imagenet_idx_to_label.json") as f:
        return json.load(f)


def get_image_transform():
    transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def predict(image):
    model = models.resnet18(pretrained=True)
    model.eval()

    out = model(image)


    _, pred = torch.max(out, 1)
    idx_to_label = get_idx_to_label()
    cls = idx_to_label[str(int(pred))]
    translation = translator.translate(cls)
    return translation


def load_image():
    image = Image.open('test.jpg')
    transform = get_image_transform()
    image = transform(image)[None]
    return image


@dp.message_handler(commands=['start'])
async def start(msg: types.Message):
    await msg.answer('Отправьте фото с животным')


@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
    await message.photo[-1].download('test.jpg')
    path = os.getcwd()
    x = load_image()
    await message.answer(f'Мне кажется это: {predict(x)}')
    os.remove(path=path+'/test.jpg')


@dp.message_handler(content_types=['text'])
async def get_photo(message: types.Message):
    await message.answer('Это не фото')


if __name__ == '__main__':
    executor.start_polling(dp)
