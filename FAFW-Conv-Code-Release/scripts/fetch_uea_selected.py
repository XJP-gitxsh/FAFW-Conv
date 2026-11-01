"""Range-download only selected .ts members from the official UEA/TSML ZIP."""
import hashlib
import argparse
import struct
import zlib
from pathlib import Path

import requests

URL = 'https://zenodo.org/records/11206331/files/TSML%20MV%20Archive%202018.zip?download=1'
DATASETS = ['FaceDetection', 'JapaneseVowels', 'SelfRegulationSCP1',
            'SelfRegulationSCP2', 'SpokenArabicDigits', 'UWaveGestureLibrary']
ROOT = Path(__file__).resolve().parents[1] / 'uea_data'

def get_range(session, start, end, total):
    response = session.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
    response.raise_for_status()
    expected = f'bytes {start}-{end}/{total}'
    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
        raise RuntimeError('Server did not return requested ZIP byte range: ' + str(response.headers))
    return response.content

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--list',action='store_true',help='list selected archive member sizes without downloading')
    args=parser.parse_args()
    session = requests.Session()
    head = session.head(URL, allow_redirects=True, timeout=60); head.raise_for_status()
    total = int(head.headers['Content-Length'])
    tail_start = max(0, total - 65536)
    tail = get_range(session, tail_start, total - 1, total)
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0: raise RuntimeError('ZIP end-of-central-directory record not found')
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment_len = struct.unpack_from('<IHHHHIIH', tail, eocd)
    if disk or cd_disk or n_disk != n_total or cd_size == 0xffffffff:
        raise RuntimeError('Multi-disk or ZIP64 archives need a different parser')
    central = get_range(session, cd_offset, cd_offset + cd_size - 1, total)
    wanted = {f'{name}_{split}.ts': name for name in DATASETS for split in ('TRAIN', 'TEST')}
    members = {}
    pos = 0
    while pos + 46 <= len(central) and central[pos:pos+4] == b'PK\x01\x02':
        v = struct.unpack_from('<IHHHHHHIIIHHHHHII', central, pos)
        comp, uncomp, name_len, extra_len, comment_len, local_offset = v[8],v[9],v[10],v[11],v[12],v[16]
        raw_name = central[pos+46:pos+46+name_len]
        name = raw_name.decode('utf-8' if (v[3] & 0x800) else 'cp437')
        base = name.replace('\\','/').rsplit('/',1)[-1]
        if base in wanted: members[base] = (name, v[4], comp, uncomp, local_offset)
        pos += 46 + name_len + extra_len + comment_len
    missing = sorted(set(wanted)-set(members))
    if missing: raise FileNotFoundError('Selected UEA members missing from archive: '+', '.join(missing))
    if args.list:
        for base, (arc_name, method, comp, uncomp, local_offset) in sorted(members.items()):
            print(base, 'compressed=',comp,'uncompressed=',uncomp)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    for base, (arc_name, method, comp, uncomp, local_offset) in members.items():
        out = ROOT / wanted[base] / base
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == uncomp:
            print('exists',out,uncomp); continue
        local = get_range(session,local_offset,local_offset+29,total)
        if local[:4] != b'PK\x03\x04': raise RuntimeError('Invalid local ZIP header: '+arc_name)
        fname_len, extra_len = struct.unpack_from('<HH',local,26)
        data_start=local_offset+30+fname_len+extra_len
        payload=get_range(session,data_start,data_start+comp-1,total)
        if method == 8: data=zlib.decompress(payload,-15)
        elif method == 0: data=payload
        else: raise RuntimeError('Unsupported ZIP compression method {} for {}'.format(method,arc_name))
        if len(data)!=uncomp: raise RuntimeError('Length mismatch for '+arc_name)
        out.write_bytes(data)
        print('saved',out,'bytes',len(data),'sha256',hashlib.sha256(data).hexdigest())

if __name__ == '__main__': main()
