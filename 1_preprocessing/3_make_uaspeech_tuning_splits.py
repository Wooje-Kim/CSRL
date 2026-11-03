import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SPEAKERS = {'F02': 2, 'F03': 3, 'F04': 1, 'F05': 0, 'M01': 3, 'M04': 3, 'M05': 1, 'M07': 2, 'M08': 0, 'M09': 0, 'M10': 0, 'M11': 1, 'M12': 3, 'M14': 0, 'M16': 2}
LABELS = ['very_low', 'low', 'medium', 'high']
GROUPS = {
    'digit': ([f'D{i}' for i in range(10)], 2),
    'letter': (['L'+chr(i) for i in range(65,91)], 5),
    'command': ([f'C{i}' for i in range(1,20)], 4),
    'common_word': ([f'CW{i}' for i in range(1,101)], 20),
}
WORDS = set().union(*(set(words) for words, _ in GROUPS.values()))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(path, speakers):
    require(path.is_file(), f'입력 파일이 없습니다: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    required = {'utterance_id','path','speaker','block','word_key','word_code_raw','word_type','is_common','is_uncommon','severity_label','severity_name','audio_version','microphone'}
    require(fields is not None and required <= set(fields), f'필수 열 누락: {path}')
    require(len(rows) == len(speakers)*465, f'발화 수 오류: {path}')
    expected = {(s, f'B{b}', w) for s in speakers for b in (1,2,3) for w in WORDS}
    counts = Counter((r['speaker'],r['block'],r['word_key']) for r in rows)
    require(set(counts)==expected and all(n==1 for n in counts.values()), f'화자·단어·블록 구성 오류: {path}')
    require(len({r['utterance_id'] for r in rows})==len(rows), f'중복 발화: {path}')
    require(len({r['path'] for r in rows})==len(rows), f'중복 경로: {path}')
    types = {w:t for t,(words,_) in GROUPS.items() for w in words}
    for r in rows:
        require(r['is_common']=='1' and r['is_uncommon']=='0', f'common 외 발화: {path}')
        require(r['audio_version']=='original' and r['microphone']=='M5', f'음성 버전 오류: {path}')
        require(r['word_key']==r['word_code_raw'] and r['word_type']==types[r['word_key']], f'단어 식별 정보 오류: {path}')
        label = SPEAKERS[r['speaker']]
        require(int(r['severity_label'])==label and r['severity_name']==LABELS[label], f'레이블 오류: {path}')
    return fields,rows


def verify(train, val, source, speakers, train_words, val_words):
    require(len(train)==len(speakers)*124*3 and len(val)==len(speakers)*31*3, '내부 분할 발화 수 오류')
    for data, words in [(train,train_words),(val,val_words)]:
        require({r['speaker'] for r in data}==speakers, '분할 화자 누락')
        require({r['word_key'] for r in data}==words, '분할 단어 목록 오류')
        counts=Counter((r['speaker'],r['word_key']) for r in data)
        require(len(counts)==len(speakers)*len(words) and all(n==3 for n in counts.values()), '단어 반복 발화 구성 오류')
        for w in words:
            for label in range(4):
                group={r['speaker'] for r in data if r['word_key']==w and int(r['severity_label'])==label}
                require(len(group)>=2, f'CSRL 교차 화자 구성 불가: {w}, label={label}')
    for field in ['utterance_id','path','word_key']:
        require(not ({r[field] for r in train}&{r[field] for r in val}), f'학습·검증 중복: {field}')
    require({r['utterance_id'] for r in train+val}=={r['utterance_id'] for r in source}, '원본 학습 발화 보존 실패')


def split_loso_utterances(rows, speakers, seed):
    """화자별 비율을 유지해 전체 발화의 10%를 무작위 검증 분리한다.

    14명 x 465발화에서는 총 651개이며, 화자별 46개 또는 47개다.
    SD RNG와 독립적이며, 단어 기준 분리나 episode 기반 재추첨은 하지 않는다.
    """
    rng = random.Random(seed)
    groups = {s: [i for i, r in enumerate(rows) if r['speaker'] == s]
              for s in sorted(speakers)}
    quotas = {s: len(indices) // 10 for s, indices in groups.items()}
    target = len(rows) // 10
    extra = target - sum(quotas.values())
    # 각 화자의 발화 수가 동일하므로 0.5개 반올림의 동률을 무작위 처리한다.
    for speaker in rng.sample(sorted(groups), extra):
        quotas[speaker] += 1
    val_indices = set()
    for speaker, indices in groups.items():
        val_indices.update(rng.sample(indices, quotas[speaker]))
    train = [r for i, r in enumerate(rows) if i not in val_indices]
    val = [r for i, r in enumerate(rows) if i in val_indices]
    return train, val


def loso_episode_gaps(train):
    gaps = []
    for word in sorted(WORDS):
        for label in range(4):
            count = len({r['speaker'] for r in train
                         if r['word_key'] == word and int(r['severity_label']) == label})
            if count < 2:
                gaps.append(dict(word_key=word, severity_label=label, speakers=count))
    return gaps


def verify_loso(train, val, source, speakers):
    require(len(train) == 5859 and len(val) == 651, 'LOSO 90%/10% 발화 수 오류')
    require(len(speakers) == 14, 'LOSO 학습 화자는 14명이어야 합니다.')
    for data in (train, val):
        require({r['speaker'] for r in data} == speakers, 'LOSO 튜닝 화자 누락/유입')
        for field in ('utterance_id', 'path'):
            require(len({r[field] for r in data}) == len(data), f'중복 {field}')
    val_counts = Counter(r['speaker'] for r in val)
    train_counts = Counter(r['speaker'] for r in train)
    require(sorted(val_counts.values()) == [46]*7 + [47]*7, '화자별 검증 비율 오류')
    require(all(train_counts[s] + val_counts[s] == 465 for s in speakers), '화자별 발화 보존 실패')
    for field in ('utterance_id', 'path'):
        require(not ({r[field] for r in train} & {r[field] for r in val}), f'학습·검증 중복: {field}')
    original = {r['utterance_id']: r for r in source}
    require({r['utterance_id'] for r in train+val} == set(original), '원본 발화 보존 실패')
    require(all(r == original[r['utterance_id']] for r in train+val), '원본 발화 정보 변경')
    # Validation은 분류 평가만 수행하므로 episode 구성 가능성을 요구하지 않는다.
    # 학습 episode 부족은 기록하고 튜닝 입력 검사에서 명시적으로 중단한다.
    return loso_episode_gaps(train)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return digest(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-splits-root',type=Path,default=Path('/home/wooje/바탕화면/code/1_preprocessing/splits_train'))
    parser.add_argument('--output-dir',type=Path,default=Path('/home/wooje/바탕화면/code/1_preprocessing/splits_tuning'))
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    root=args.training_splits_root.expanduser().resolve()
    out=args.output_dir.expanduser().resolve()
    require(not out.exists(), f'저장 폴더가 이미 있습니다. 다른 --output-dir을 지정하세요: {out}')
    rng=random.Random(args.seed)
    # 전체 155단어의 무작위 순열에서 앞의 31단어를 선택한다. 종류별 할당 없음.
    ordered_words=sorted(WORDS)
    rng.shuffle(ordered_words)
    val_words=set(ordered_words[:31])
    train_words=WORDS-val_words
    require(len(val_words)==31 and len(train_words)==124, '단어 분할 수 오류')
    jobs=[('SD',None)]+[(f'LOSO/fold_{s}',s) for s in sorted(SPEAKERS)]
    prepared=[]
    source_hashes={}
    sd_map=None
    loso_split_details={}
    for folder,heldout in jobs:
        speakers=set(SPEAKERS)-({heldout} if heldout else set())
        path=root/folder/'train.csv'
        fields,rows=read_source(path,speakers)
        if heldout is None:
            sd_map={r['utterance_id']:r for r in rows}
        else:
            require(all(r==sd_map.get(r['utterance_id']) for r in rows), f'SD와 LOSO의 원본 발화 정보 불일치: {folder}')
        if heldout is None:
            train=[r for r in rows if r['word_key'] in train_words]
            val=[r for r in rows if r['word_key'] in val_words]
            verify(train,val,rows,speakers,train_words,val_words)
        else:
            train,val=split_loso_utterances(rows,speakers,args.seed)
            gaps=verify_loso(train,val,rows,speakers)
            loso_split_details[folder]=dict(
                validation_fraction=0.1, group_unit='utterance, stratified by speaker',
                train_rows=len(train), val_rows=len(val),
                val_rows_per_speaker=dict(sorted(Counter(r['speaker'] for r in val).items())),
                csrl_train_feasible=not gaps, episode_ineligible_groups=gaps)
        source_hashes[f'{folder}/train.csv']=digest(path)
        prepared.append((folder,heldout,fields,train,val))
    summaries=[]
    hashes={}
    for folder,heldout,fields,train,val in prepared:
        for name,data in [('train.csv',train),('val.csv',val)]:
            relative=f'{folder}/{name}'
            hashes[relative]=write_csv(out/relative,data,fields)
        summaries.append(dict(protocol='LOSO' if heldout else 'SD',fold=folder,excluded_test_speaker=heldout or '',train_rows=len(train),val_rows=len(val),train_speakers=len({r['speaker'] for r in train}),val_speakers=len({r['speaker'] for r in val}),train_words=len({r['word_key'] for r in train}),val_words=len({r['word_key'] for r in val}),word_overlap=len({r['word_key'] for r in train}&{r['word_key'] for r in val}),utterance_overlap=0,csrl_train_feasible=loso_split_details[folder]['csrl_train_feasible'] if heldout else True,csrl_val_feasible=None if heldout else True,passed=True))
    write_csv(out/'split_summary.csv',summaries,list(summaries[0]))
    word_rows=[dict(word_type=t,word_key=w,split='val' if w in val_words else 'train') for t,(words,_) in GROUPS.items() for w in words]
    write_csv(out/'word_assignment.csv',word_rows,['word_type','word_key','split'])
    info=dict(seed=args.seed,training_splits_root=str(root),output_dir=str(out),protocols={
        'SD':dict(validation_fraction=0.2,train_words=sorted(train_words),val_words=sorted(val_words),validation_type_counts={t:sum(w in val_words for w in words) for t,(words,_) in GROUPS.items()},group_unit='word_key across all speakers and B1/B2/B3',word_assignment_file='word_assignment.csv'),
        'LOSO':dict(validation_fraction=0.1,group_unit='utterance, stratified by speaker',folds=loso_split_details)},shared_word_assignment_across_protocols=False,test_csv_read=False,test_csv_written=False,source_train_sha256=source_hashes,output_sha256=hashes)
    (out/'split_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'SD: TRAIN=5580, VAL=1395 (기존 124/31단어 분할 유지)\nLOSO: 15 folds, TRAIN=5859, VAL=651/fold (화자별 46/47발화 검증)\n시드: {args.seed}\n저장: {out}')
    for folder, detail in loso_split_details.items():
        if not detail['csrl_train_feasible']:
            print(f"주의: {folder} 학습 episode 구성 불가 그룹: {detail['episode_ineligible_groups']}. 자동 재추첨/단어 제외는 하지 않았습니다.")


if __name__=='__main__':
    main()
