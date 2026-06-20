from pathlib import Path
from src.quality_checker import check_quality

processed_dir = Path('processed')
wav_files = list(processed_dir.glob('*.wav'))
print(f'Found {len(wav_files)} files to quality check')

passed = 0
failed = 0
for wav in sorted(wav_files):
    result = check_quality(
        audio_path=wav,
        snr_threshold_db=18.0,
        silence_ratio_max=0.30,
        clipping_max_pct=0.1
    )
    status = 'PASS' if result['passed'] else 'FAIL'
    reasons = result['reject_reasons']
    snr = round(result['snr_db'], 1)
    silence = round(result['silence_ratio'], 2)
    quality = round(result['quality_score'], 2)
    print(f"{status} | {wav.name} | SNR={snr}dB | silence={silence} | quality={quality} | {reasons}")
    if result['passed']:
        passed += 1
    else:
        failed += 1

print(f'Passed: {passed} | Failed: {failed}')
