from pathlib import Path
import soundfile as sf

segments_dir = Path('segments')
all_segs = list(segments_dir.glob('*.wav'))

short = []
good = []
long = []

for seg in sorted(all_segs):
    info = sf.info(str(seg))
    dur = info.duration
    if dur < 15:
        short.append((seg.name, dur))
    elif dur <= 30:
        good.append((seg.name, dur))
    else:
        long.append((seg.name, dur))

print(f'Too short (<15s): {len(short)}')
print(f'Good (15-30s):    {len(good)}')
print(f'Too long (>30s):  {len(long)}')
print(f'Total:            {len(all_segs)}')
print(f'\nGood segments that will be transcribed:')
for name, dur in good:
    print(f'  {name} ({dur:.1f}s)')
