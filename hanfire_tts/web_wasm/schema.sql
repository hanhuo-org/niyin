-- 汉火古音 TTS 听审打分与众包纠错数据库 Schema
-- 同时兼容 Cloudflare D1 与本地 SQLite (serve_web.py)

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  nickname TEXT NOT NULL,
  email TEXT NOT NULL DEFAULT '',
  rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  ipa_correction TEXT NOT NULL DEFAULT '',
  comment TEXT NOT NULL DEFAULT '',
  era TEXT NOT NULL,
  source_type TEXT NOT NULL DEFAULT 'synth',
  rhyme_id TEXT NOT NULL DEFAULT '',
  input_text TEXT NOT NULL,
  voice_key TEXT NOT NULL DEFAULT 'male_128Hz',
  speed REAL NOT NULL DEFAULT 1.0,
  num_step INTEGER NOT NULL DEFAULT 16,
  compiled_gloss_json TEXT NOT NULL DEFAULT '[]',
  user_adjustments_json TEXT NOT NULL DEFAULT '[]',
  model_build_id TEXT NOT NULL DEFAULT '',
  is_public INTEGER NOT NULL DEFAULT 0 CHECK (is_public IN (0, 1)),
  status TEXT NOT NULL DEFAULT 'pending',
  admin_note TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_public ON feedback(is_public, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_era_status ON feedback(era, status);
