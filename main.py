import re
import mobi
import torch
import torchaudio
from pysbd.segmenter import Segmenter
from bs4 import BeautifulSoup
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from pathlib import Path

segmenter = Segmenter(clean=True)


def get_chapters(filename: str) -> list[list[str]]:
    tempdir, filepath = mobi.extract(filename)
    with open(filepath, 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    tags = soup.find_all(['p', 'br'])
    text = ''
    for tag in tags:
        if tag.name == 'p':
            text += f'{tag.text} '
        elif tag.name == 'br':
            text += '\n '
    r = re.compile(r'Table of Contents', re.IGNORECASE)
    text = r.split(text)
    chapter_regex = re.compile(r'Chapter \d+', re.IGNORECASE)
    chapters = chapter_regex.split(text[0])[1:]
    chapters = [segmenter.segment(chapter) for chapter in chapters]
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


def merge_wav_files(wav_files, output_path):
    waveforms = []
    sample_rate = 0
    for wav_file in wav_files:
        waveform, sample_rate = torchaudio.load(wav_file)
        waveforms.append(waveform)
    combined_waveform = torch.cat(waveforms, dim=1)
    torchaudio.save(output_path, combined_waveform, sample_rate)


def synthesize_chapter(chapter, chapter_num, gpt_cond_latent, speaker_embedding, model) -> None:
    paragraphs = []
    for i, sentence in enumerate(chapter):
        print(len(sentence))
        out = model.inference(
            sentence,
            'en',
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,
            top_k=75,
            top_p=0.9
        )
        wav_filepath = f'audio/temp/c{chapter_num + 1}_p{i + 1}.wav'
        Path(wav_filepath).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(wav_filepath,
                        torch.tensor(out['wav']).unsqueeze(0),
                        24000)
        paragraphs.append(wav_filepath)
    res_filepath = f'audio/result/chapter_{chapter_num + 1}.wav'
    merge_wav_files(paragraphs, res_filepath)
    print(f'Finished chapter {chapter_num}')


def main(book_file: str, ref_files: list[str]) -> None:
    chapters = get_chapters(book_file)
    model = init_model()
    print('Computing speaker latents...')
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=ref_files
    )

    for index, chapter in enumerate(chapters):
        synthesize_chapter(chapter,
                           index,
                           gpt_cond_latent,
                           speaker_embedding,
                           model)
    print('DONE!')


if __name__ == '__main__':
    main("Harry Potter and the Sorcerer's Stone.mobi",
         ['audio/reference/yegor_en.wav'])
