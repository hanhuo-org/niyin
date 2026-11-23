import Qieyun from 'qieyun';
const fin = new Set(), ini = new Set(), finTone = new Set(); let chars = 0;
for (const 地位 of Qieyun.資料.iter音韻地位()) {
  const d = 地位;
  ini.add(d.母);
  const key = `${d.韻}${d.呼 ?? ''}${d.等}${d.類 ?? ''}`;
  fin.add(key); finTone.add(key + d.聲);
}
console.log('音韻地位 3810 | 母', ini.size, '| 韻×呼×等×類 (韵母类)', fin.size, '| 含声调', finTone.size);
