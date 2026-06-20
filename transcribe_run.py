from pathlib import Path
from src.transcriber import transcribe_clip

processed_dir = Path('processed')

passed_files = [
    'en_src001.wav', 'en_src002.wav', 'en_src003.wav', 'en_src004.wav',
    'en_src005.wav', 'en_src008.wav', 'en_src009.wav', 'en_src010.wav',
    'en_src011.wav', 'hi_src001.wav', 'hi_src002.wav', 'hi_src004.wav',
    'hi_src005.wav', 'hi_src006.wav', 'hi_src008.wav', 'hi_src009.wav'
]

for filename in passed_files:
    wav = processed_dir / filename
    lang = 'hi-IN' if filename.startswith('hi') else 'en-IN'
    print(f'\nTranscribing {filename} ({lang})...')
    result = transcribe_clip(wav, language_code=lang)
    if result and result.get('transcript'):
        transcript = result['transcript']
        print('  OK: ' + transcript[:100])
    else:
        print('  FAILED')

print('\nDone.')
