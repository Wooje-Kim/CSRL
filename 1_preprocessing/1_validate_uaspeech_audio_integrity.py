import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

PATTERN = re.compile(r'^(?P<speaker>C?[FM]\d+)_B(?P<block>[123])_(?P<word>[A-Z]+\d*)_M(?P<mic>\d+)$', re.I)
SPEAKERS = 'F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16'.split()


def save(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def inspect(path):
    result = dict(valid_audio=False, sample_rate='', channels='', frames='', duration_sec='', error='', silent=False)
    try:
        total = 0
        peak = 0.0
        with sf.SoundFile(path) as f:
            declared = len(f)
            result.update(sample_rate=f.samplerate, channels=f.channels)
            for block in f.blocks(blocksize=65536, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError('nonfinite_samples')
                total += len(block)
                if block.size:
                    peak = max(peak, float(np.abs(block).max()))
            if total == 0 or total != declared:
                raise ValueError(f'frame_count: decoded={total}, declared={declared}')
            result.update(valid_audio=True, frames=total, duration_sec=total/f.samplerate, silent=peak == 0)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', type=Path, default=Path('/home/wooje/data/UASpeech/audio'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--speakers', nargs='+', default=SPEAKERS)
    args = parser.parse_args()
    root = args.audio_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = args.output_dir or Path.cwd() / ('audio_integrity_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    versions = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    selected = set(s.upper() for s in args.speakers)
    words = [f'D{i}' for i in range(10)] + ['L'+chr(i) for i in range(65,91)] + [f'{prefix}{i}' for prefix, count in [('C',19),('CW',100),('UW',100)] for i in range(1,count+1)]
    expected = {(s, b, w) for s in selected for b in (1,2,3) for w in words}
    records = []
    groups = defaultdict(list)
    microphones = set()
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.wav' and not p.name.startswith('._'))
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root)
        version = relative.parts[0] if len(relative.parts)>1 else '__ROOT__'
        match = PATTERN.fullmatch(path.stem)
        row = dict(path=str(path), version=version, speaker='', block='', word='', microphone='', filename_valid=bool(match), study_speaker=False)
        if match:
            speaker = match['speaker'].upper()
            row.update(speaker=speaker, block=int(match['block']), word=match['word'].upper(), microphone='M'+match['mic'], study_speaker=speaker in selected)
            microphones.add(row['microphone'])
        row.update(inspect(path))
        records.append(row)
        if match:
            groups[(version,row['microphone'])].append(row)
        if index % 100 == 0 or index == len(files):
            print(f'{index}/{len(files)}', flush=True)
    summaries, differences, duplicates = [], [], []
    sets = {}
    for version in versions:
        malformed = sum(r['version']==version and not r['filename_valid'] for r in records)
        for mic in sorted(microphones):
            all_rows = groups[(version,mic)]
            rows = [r for r in all_rows if r['study_speaker']]
            counts = Counter((r['speaker'],r['block'],r['word']) for r in rows)
            present = set(counts)
            sets[(version,mic)] = present
            missing, extra = expected-present, present-expected
            for kind, keys in [('missing',missing),('unexpected',extra)]:
                for s,b,w in sorted(keys):
                    differences.append(dict(version=version,microphone=mic,kind=kind,speaker=s,block=b,word=w))
            for key,count in counts.items():
                if count>1:
                    for r in rows:
                        if (r['speaker'],r['block'],r['word'])==key:
                            duplicates.append(dict(version=version,microphone=mic,speaker=key[0],block=key[1],word=key[2],path=r['path']))
            invalid = sum(not r['valid_audio'] for r in rows)
            format_errors = sum(r['valid_audio'] and (r['sample_rate']!=16000 or r['channels']!=1) for r in rows)
            silent = sum(r['silent'] for r in rows)
            duplicate_count = sum(n-1 for n in counts.values())
            passed = bool(rows) and not any([len(missing),len(extra),duplicate_count,invalid,format_errors,silent,malformed])
            summaries.append(dict(version=version,microphone=mic,all_speaker_files=len(all_rows),study_files=len(rows),expected_files=len(expected),missing=len(missing),unexpected=len(extra),duplicates=duplicate_count,invalid_audio=invalid,format_errors=format_errors,silent_files=silent,unparsed_files_in_version=malformed,duration_over_20s=sum(r['valid_audio'] and r['duration_sec']>20 for r in rows),passed=passed))
    correspondence = []
    for mic in sorted(microphones):
        for i, first in enumerate(versions):
            for second in versions[i+1:]:
                a,b = sets[(first,mic)],sets[(second,mic)]
                correspondence.append(dict(microphone=mic,version_a=first,version_b=second,shared=len(a&b),only_a=len(a-b),only_b=len(b-a),same_inventory=bool(a) and a==b,both_complete=a==expected and b==expected))
    save(out/'audio_inventory.csv',records,list(records[0]) if records else ['path'])
    save(out/'combination_summary.csv',summaries,list(summaries[0]) if summaries else ['version','microphone','passed'])
    save(out/'missing_unexpected.csv',differences,['version','microphone','kind','speaker','block','word'])
    save(out/'duplicates.csv',duplicates,['version','microphone','speaker','block','word','path'])
    save(out/'version_correspondence.csv',correspondence,['microphone','version_a','version_b','shared','only_a','only_b','same_inventory','both_complete'])
    report = dict(audio_root=str(root),versions=versions,scanned_files=len(files),study_speakers=sorted(selected),expected_per_combination=len(expected),expected_word_codes_per_block=255,expectation='D0-9,LA-LZ,C1-19,CW1-100,UW1-100; B1-3',passed_combinations=[r for r in summaries if r['passed']],output_dir=str(out.resolve()),scope='All WAV files are decoded; completeness and candidate selection use study_speakers. Identity correspondence does not verify spoken content or transcript correctness.')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not records:
        raise RuntimeError('No WAV files found')


if __name__ == '__main__':
    main()
