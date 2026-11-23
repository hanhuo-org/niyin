// 导出统一中古词表（CC0：qieyun-js 广韵数据 + qieyun-examples 推导方案）
import Qieyun from 'qieyun';
import { from字頭 } from 'qieyun-examples';
import fs from 'fs';
const rows = [['字', '序', '描述', '母', '呼', '等', '類', '韻', '聲', '清濁', 'tupa', 'unt', 'mid_tang', 'zhongyuan', 'putonghua', '反切', '釋義']];
const seen = new Set();
for (const 地位 of Qieyun.資料.iter音韻地位()) {
  for (const e of Qieyun.資料.query音韻地位(地位)) {
    const ch = e.字頭;
    if (seen.has(ch)) continue;
    seen.add(ch);
    const rs = from字頭(['tupa', 'unt', 'mid_tang', 'zhongyuan', 'putonghua'], ch);
    rs.forEach((r, i) => {
      const d = r.音韻地位;
      rows.push([ch, i, d.描述, d.母, d.呼 ?? '', d.等, d.重紐 ?? '', d.韻, d.聲, d.清濁, ...r.擬音.map(x => String(x).replace(/[\t\n\r]+/g, '/')), r.反切 ?? '', (r.釋義 ?? '').replace(/[\t\n]/g, ' ').slice(0, 40)]);
    });
  }
}
fs.writeFileSync('mc_qieyun_unified.tsv', rows.map(r => r.join('\t')).join('\n'));
console.log('chars', seen.size, 'rows', rows.length - 1);
