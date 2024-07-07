import re

import mobi
from bs4 import BeautifulSoup
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager


def get_chapters(filename: str) -> list[str]:
    tempdir, filepath = mobi.extract(filename)
    with open(filepath, 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    tags = soup.find_all(['p', 'br'])
    text = ''
    for tag in tags:
        if tag.name == 'p':
            print(len(tag.text))
            text += f'{tag.text} '
        elif tag.name == 'br':
            text += '\n '
    r = re.compile(r'Table of Contents', re.IGNORECASE)
    text = r.split(text)
    chapter_regex = re.compile(r'Chapter \d+', re.IGNORECASE)
    chapters = chapter_regex.split(text[0])[1:]
    return chapters


def init_model():
    manager = ModelManager()
    print('Loading model:')
    model_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
    xtts_path, _, _ = manager.download_model(model_name)
    config = XttsConfig()
    config.load_json(xtts_path + '/config.json')
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config,
                          checkpoint_dir=xtts_path,
                          use_deepspeed=True if torch.cuda.is_available() else False)
    if torch.cuda.is_available():
        model.cuda()
    return model


def synthesize_chapter(text, chapter_num, gpt_cond_latent, speaker_embedding, model) -> None:

    out = model.inference(
        text,
        'en',
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7
    )
    torchaudio.save(f'Chapter {chapter_num}.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)
    print(f'Finished chapter {chapter_num}')


def main() -> None:
    chapters = get_chapters("Harry Potter and the Sorcerer's Stone.mobi")
    model = init_model()
    print('Computing speaker latents...')
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=['audio/yegor_en.wav']
    )

    for index in range(len(chapters)):
        synthesize_chapter(chapters[index],
                           index,
                           gpt_cond_latent,
                           speaker_embedding,
                           model)

    print('DONE!')


if __name__ == '__main__':
    main()

# manager = ModelManager()
# print('Loading model...')
# model_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
# xtts_path, _, _ = manager.download_model(model_name)
# config = XttsConfig()
# config.load_json(xtts_path + '/config.json')
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config,
#                       checkpoint_dir=xtts_path,
#                       use_deepspeed=True if torch.cuda.is_available() else False)
# if torch.cuda.is_available():
#     model.cuda()
#
# print('Computing speaker latents...')
# gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=['audio/yegor_en.wav'])
#
# print('Inference...')
# print('First')
# out = model.inference(
#     'It took me quite a long time to develop a voice and now that I have it I am not going to be silent.',
#     'en',
#     gpt_cond_latent,
#     speaker_embedding,
#     temperature=0.7
# )
# torchaudio.save('output.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)
#
# print('Second')
# out = model.inference(
#     'It doesn\'t bother me! But what\'s your name?',
#     'en',
#     gpt_cond_latent,
#     speaker_embedding,
#     temperature=0.7
# )
# torchaudio.save('output2.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)
#
# print('DONE!')
