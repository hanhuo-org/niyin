import { from字頭 } from 'qieyun-examples';
for (const ch of '千山鳥飛絕萬徑人蹤滅孤舟蓑笠翁獨釣寒江雪語爾') {
  const r = from字頭(['unt', 'tupa', 'mid_tang', 'zhongyuan'], ch);
  console.log(ch, r.map(x => `${x.音韻地位.描述}:${x.擬音.join(' | ')}`).join(' ;; '));
}
