import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SPEAKERS = {
    'F02': (29.0, 'medium', 2), 'F03': (6.0, 'high', 3),
    'F04': (62.0, 'low', 1), 'F05': (95.0, 'very_low', 0),
    'M01': (15.0, 'high', 3), 'M04': (2.0, 'high', 3),
    'M05': (58.0, 'low', 1), 'M07': (28.0, 'medium', 2),
    'M08': (93.0, 'very_low', 0), 'M09': (86.0, 'very_low', 0),
    'M10': (93.0, 'very_low', 0), 'M11': (62.0, 'low', 1),
    'M12': (7.4, 'high', 3), 'M14': (90.4, 'very_low', 0),
    'M16': (43.0, 'medium', 2),
}
COMMON = {f'D{i}' for i in range(10)} | {'L' + chr(i) for i in range(65, 91)} | {f'C{i}' for i in range(1, 20)} | {f'CW{i}' for i in range(1, 101)}
UNCOMMON = {f'UW{i}' for i in range(1, 101)}
PATTERN = re.compile(r'^(C?[FM]\d+)_B([123])_([A-Z]+\d*)_M(\d+)$', re.I)
FIELDS = ['utterance_id', 'path', 'relative_path', 'filename', 'speaker', 'gender', 'block', 'word_code_raw', 'word_code', 'word_key', 'word_type', 'is_common', 'is_uncommon', 'intelligibility', 'severity_name', 'severity_label', 'audio_version', 'microphone']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(audio_root):
    source = audio_root / 'original'
    require(source.is_dir(), f'원본 음성 폴더가 없습니다: {source}')
    rows = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() != '.wav' or path.name.startswith('._'):
            continue
        match = PATTERN.fullmatch(path.stem)
        if not match:
            require(not path.stem.upper().endswith('_M5'), f'M5 파일 이름 해석 실패: {path}')
            continue
        speaker, block, word, mic = match.groups()
        speaker, word = speaker.upper(), word.upper()
        if speaker not in SPEAKERS or mic != '5':
            continue
        require(word in COMMON | UNCOMMON, f'예상 외 단어 코드: {path}')
        common = word in COMMON
        key = word if common else f'B{block}_{word}'
        score, severity, label = SPEAKERS[speaker]
        kind = 'uncommon_word' if not common else ('common_word' if word.startswith('CW') else 'digit' if word.startswith('D') else 'letter' if word.startswith('L') else 'command')
        rows.append(dict(utterance_id=f'{speaker}_B{block}_{word}_M5', path=str(path.resolve()), relative_path=path.relative_to(audio_root).as_posix(), filename=path.name, speaker=speaker, gender=speaker[0], block=f'B{block}', word_code_raw=word, word_code=key, word_key=key, word_type=kind, is_common=int(common), is_uncommon=int(not common), intelligibility=score, severity_name=severity, severity_label=label, audio_version='original', microphone='M5'))
    expected = {(s, f'B{b}', w) for s in SPEAKERS for b in (1, 2, 3) for w in COMMON | UNCOMMON}
    counts = Counter((r['speaker'], r['block'], r['word_code_raw']) for r in rows)
    require(not any(n != 1 for n in counts.values()), '중복 화자·블록·단어 ID가 있습니다.')
    missing = sorted(expected - set(counts))
    require(not missing, f'누락 발화 {len(missing)}개: {missing[:20]}')
    require(len(rows) == 11475, f'전체 발화 수 오류: {len(rows)}')
    require(len({r['path'] for r in rows}) == len(rows), '음성 경로가 중복됩니다.')
    return rows


def verify(train, test, heldout=None):
    n = 14 if heldout else 15
    require(len(train) == n * 465, '학습 발화 수 오류')
    require(len(test) == (300 if heldout else 4500), '평가 발화 수 오류')
    train_sp = {r['speaker'] for r in train}
    test_sp = {r['speaker'] for r in test}
    require(train_sp == set(SPEAKERS) - ({heldout} if heldout else set()), '학습 화자 구성 오류')
    require(test_sp == ({heldout} if heldout else set(SPEAKERS)), '평가 화자 구성 오류')
    require(all(r['is_common'] for r in train), '학습에 uncommon 포함')
    require(all(r['is_uncommon'] for r in test), '평가에 common 포함')
    require(len({r['word_key'] for r in train}) == 155, '학습 단어 수 오류')
    require(len({r['word_key'] for r in test}) == 300, '평가 단어 수 오류')
    require(not ({r['utterance_id'] for r in train} & {r['utterance_id'] for r in test}), '학습·평가 발화 중복')
    require(not ({r['word_key'] for r in train} & {r['word_key'] for r in test}), '학습·평가 단어 ID 중복')
    require(not ({r['path'] for r in train} & {r['path'] for r in test}), '학습·평가 경로 중복')
    require(set(r['severity_label'] for r in train) == {0,1,2,3}, '학습 중증도 클래스 누락')
    return dict(protocol='LOSO' if heldout else 'SD', fold=f'fold_{heldout}' if heldout else 'SD', heldout_speaker=heldout or '', train_rows=len(train), test_rows=len(test), train_speakers=len(train_sp), test_speakers=len(test_sp), train_words=155, test_words=300, passed=True)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', type=Path, default=Path('/home/wooje/data/UASpeech/audio'))
    parser.add_argument('--output-dir', type=Path, default=Path('/home/wooje/바탕화면/code/1_preprocessing/splits_train'))
    args = parser.parse_args()
    root = args.audio_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    require(not out.exists(), f'저장 폴더가 이미 존재합니다. 별도 --output-dir을 지정하세요: {out}')
    rows = collect(root)
    common = [r for r in rows if r['is_common']]
    uncommon = [r for r in rows if r['is_uncommon']]
    splits = [('SD', common, uncommon, None)]
    for s in sorted(SPEAKERS):
        splits.append((f'LOSO/fold_{s}', [r for r in common if r['speaker'] != s], [r for r in uncommon if r['speaker'] == s], s))
    summaries = [verify(train, test, heldout) for _, train, test, heldout in splits]
    hashes = {}
    for folder, train, test, _ in splits:
        for name, data in [('train.csv', train), ('test.csv', test)]:
            relative = f'{folder}/{name}'
            hashes[relative] = write_csv(out / relative, data, FIELDS)
    write_csv(out / 'split_summary.csv', summaries, list(summaries[0]))
    report = dict(audio_root=str(root), audio_version='original', microphone='M5', source_rows=len(rows), sd_train=6975, sd_test=4500, loso_folds=15, loso_train_per_fold=6510, loso_test_per_fold=300, validation_split_created=False, duration_filter_applied=False, speaker_labels=SPEAKERS, split_sha256=hashes, ordering='speaker/path lexical order', word_identity='common: raw code; uncommon: block + raw code', output_dir=str(out))
    (out / 'split_info.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'SD: TRAIN=6975, TEST=4500\nLOSO: 15 folds, TRAIN=6510, TEST=300/fold\n저장: {out}')


if __name__ == '__main__':
    main()
