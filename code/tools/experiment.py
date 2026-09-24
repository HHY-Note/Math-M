"""实验文件的通用哈希、结果摘要和表格写出。"""

import csv
import hashlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_summary(result):
    return dict(makespan=result['makespan'], num_cores=result['num_cores'],
                data_movement_bytes=result['data_movement_bytes'],
                memory_peak_by_core=result['memory_peak_by_core'],
                per_core_tasks=[dict(core_id=c['core_id'], tasks=c['tasks'])
                                for c in result['per_core_timeline']])


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
