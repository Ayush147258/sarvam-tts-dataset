from pathlib import Path
import soundfile as sf
from src.segmenter import segment_audio

processed_dir = Path('processed')
segments_dir = Path('segments')
segments_dir.mkdir(exist_ok=True)

passed_files = [
    ('en_src001.wav', 'en-IN'), ('en_src002.wav', 'en-IN'),
    ('en_src003.wav', 'en-IN'), ('en_src004.wav', 'en-IN'),
    ('en_src005.wav', 'en-IN'), ('en_src008.wav', 'en-IN'),
    ('en_src009.wav', 'en-IN'), ('en_src010.wav', 'en-IN'),
    ('en_src011.wav', 'en-IN'), ('hi_src001.wav', 'hi-IN'),
    ('hi_src002.wav', 'hi-IN'), ('hi_src004.wav', 'hi-IN'),
    ('hi_src005.wav', 'hi-IN'), ('hi_src006.wav', 'hi-IN'),
    ('hi_src008.wav', 'hi-IN'), ('hi_src009.wav', 'hi-IN'),
]

total = 0
for filename, lang in passed_files:
    wav_path = processed_dir / filename
    stem = wav_path.stem
    print(f'\nSegmenting {filename}...')
    
    segments = segment_audio(wav_path)
    if not segments:
        print(f'  No segments found')
        continue

    audio, sr = sf.read(str(wav_path))
    
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        clip = audio[start_sample:end_sample]
        
        duration = end - start
        clip_id = f'{stem}_seg{i+1:03d}'
        out_path = segments_dir / f'{clip_id}.wav'
        
        sf.write(str(out_path), clip, sr)
        total += 1
        print(f'  ✅ {clip_id} ({duration:.1f}s) -> {out_path}')

print(f'\nTotal segments saved: {total}')
