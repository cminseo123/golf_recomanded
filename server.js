const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const ROOT_DIR = __dirname;
const PORT = Number(process.env.PORT || 4173);
const DEFAULT_OLLAMA_MODELS = [
  'fitted-golf',
  'qwen2.5:3b-instruct',
  'qwen2.5:3b',
  'qwen2.5:1.5b-instruct',
  'qwen2.5:1.5b',
];
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://127.0.0.1:11434/api/chat';
const PRIMARY_OLLAMA_MODEL =
  typeof process.env.OLLAMA_MODEL === 'string' && process.env.OLLAMA_MODEL.trim()
    ? process.env.OLLAMA_MODEL.trim()
    : DEFAULT_OLLAMA_MODELS[0];
const OLLAMA_MODEL_CANDIDATES = [
  ...new Set(
    [
      PRIMARY_OLLAMA_MODEL,
      ...(process.env.OLLAMA_MODEL_CANDIDATES || '')
        .split(',')
        .map(item => item.trim())
        .filter(Boolean),
      ...DEFAULT_OLLAMA_MODELS,
    ].filter(Boolean),
  ),
];

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function withCors(headers = {}) {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    ...headers,
  };
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, withCors({ 'Content-Type': 'application/json; charset=utf-8' }));
  res.end(JSON.stringify(payload, null, 2));
}

function sendText(res, statusCode, body, contentType = 'text/plain; charset=utf-8') {
  res.writeHead(statusCode, withCors({ 'Content-Type': contentType }));
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', chunk => {
      raw += chunk;
      if (raw.length > 1_000_000) {
        reject(new Error('Request body too large.'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(raw));
    req.on('error', reject);
  });
}

async function readJson(req) {
  const raw = await readBody(req);
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error('Invalid JSON body.');
  }
}

function unique(items) {
  return [...new Set((items || []).filter(Boolean))];
}

function extractLastUserMessage(messages) {
  const reversed = Array.isArray(messages) ? [...messages].reverse() : [];
  const last = reversed.find(message => message && message.role === 'user');
  return last?.content || '';
}

function safeString(value, fallback = '') {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback;
}

function safeNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function summarizePayload(payload = {}) {
  const profile = payload.profile || {};
  const swing = payload.swing || {};
  const results = payload.results || {};
  const driver = results.driver || {};

  return [
    `성별/나이: ${safeString(profile.gender, '미상')} / ${safeNumber(profile.age, 0)}세`,
    `핸디캡: ${safeNumber(profile.handicap, 0)}`,
    `헤드스피드: ${safeNumber(swing.headSpeed, 0)}mph`,
    `볼스피드: ${safeNumber(swing.ballSpeed, 0)}mph`,
    `구질: ${safeString(swing.shape, '미상')}`,
    `패스: ${safeString(swing.path, '미상')}`,
    `드라이버 로프트: ${safeString(driver.loft, '미상')}`,
    `드라이버 플렉스: ${safeString(driver.flex, '미상')}`,
  ].join('\n');
}

function normalizeExplainOutput(payload, raw = {}, mode = 'fallback') {
  const fallback = buildExplainFallback(payload);
  return {
    mode,
    summary: safeString(raw.summary, fallback.summary),
    coachComment: safeString(raw.coachComment, fallback.coachComment),
    priorities: Array.isArray(raw.priorities) && raw.priorities.length ? raw.priorities : fallback.priorities,
    nextSteps: Array.isArray(raw.nextSteps) && raw.nextSteps.length ? raw.nextSteps : fallback.nextSteps,
  };
}

function buildExplainFallback(payload = {}) {
  const profile = payload.profile || {};
  const swing = payload.swing || {};
  const results = payload.results || {};
  const driver = results.driver || {};
  const irons = results.irons || {};
  const wedges = results.wedges || {};
  const putter = results.putter || {};

  const headSpeed = safeNumber(swing.headSpeed, 90);
  const handicap = safeNumber(profile.handicap, 18);
  const smashFactor = safeNumber(swing.smashFactor, 1.42);
  const attackAngle = safeNumber(swing.attackAngle, 0);

  const priorities = [];
  const nextSteps = [];

  if (/슬라이스|fade|right/i.test(safeString(swing.shape)) || /아웃/i.test(safeString(swing.path))) {
    priorities.push('드라이버는 슬라이스 보정과 시작 방향 안정화가 1순위입니다.');
    nextSteps.push('드로우 바이어스 성향 헤드와 현재 로프트 조합을 먼저 시타해 보세요.');
  }

  if (smashFactor < 1.45) {
    priorities.push('스펙 변경보다 정타율과 임팩트 품질 개선이 거리 증가에 더 큰 영향을 줄 수 있습니다.');
    nextSteps.push('드라이버 페이스 중앙 타점 여부를 먼저 체크해 보세요.');
  }

  if (attackAngle < 0 && headSpeed < 95) {
    priorities.push('현재 데이터에서는 어택 앵글과 로프트를 함께 봐야 탄도 손실을 줄이기 쉽습니다.');
    nextSteps.push('티 높이와 볼 포지션을 조정하며 출발각 변화를 확인해 보세요.');
  }

  if (handicap >= 18) {
    priorities.push('관용성과 반복 재현성을 우선한 세팅이 현재 단계에서 더 유리합니다.');
  } else {
    priorities.push('안정성은 유지하면서도 컨트롤 손실이 없는 범위의 미세 조정이 우선입니다.');
  }

  if (!nextSteps.length) {
    nextSteps.push('현재 추천 스펙으로 10~15구 정도 시타하며 볼 출발 방향과 탄도를 확인해 보세요.');
  }

  nextSteps.push('필드 전에는 라이브러리 수치보다 실제 탄도와 좌우 분산을 함께 확인하세요.');
  nextSteps.push('최종 구매 전에는 드라이버와 아이언 중 체감 변화가 큰 카테고리부터 우선 테스트하세요.');

  return {
    mode: 'fallback',
    summary:
      `${safeString(swing.shape, '현재 구질')} 경향과 ${headSpeed}mph 헤드스피드 기준으로, ` +
      `관용성과 탄도 확보를 우선한 거친 MVP 추천입니다.`,
    coachComment:
      `드라이버는 ${safeString(driver.loft, '기본 로프트')} / ${safeString(driver.flex, '기본 플렉스')} 조합을 기준으로 시작하고, ` +
      `아이언은 ${safeString(irons.type, '현재 추천 타입')} 쪽이 무난합니다. ` +
      `웨지는 ${safeString(wedges.bounce, '기본 바운스')}처럼 범용성 있는 세팅이 안전하고, ` +
      `퍼터는 ${safeString(putter.type, '안정형 헤드')} 쪽이 현재 미스 패턴을 줄이는 출발점이 됩니다.`,
    priorities: unique(priorities).slice(0, 3),
    nextSteps: unique(nextSteps).slice(0, 3),
  };
}

function buildChatFallback(payload = {}, question = '') {
  const profile = payload.profile || {};
  const swing = payload.swing || {};
  const results = payload.results || {};
  const driver = results.driver || {};
  const irons = results.irons || {};
  const wedges = results.wedges || {};
  const putter = results.putter || {};

  if (/샤프트|shaft|flex/i.test(question)) {
    return (
      `현재 헤드스피드 ${safeNumber(swing.headSpeed, 90)}mph 기준으로 보면 드라이버는 ${safeString(driver.flex, '기본')} 플렉스가 출발점입니다. ` +
      `지금 MVP 추천에서는 ${safeString(driver.shaftWeight, '중간 무게')} 구간을 보고 있어서, 너무 무거운 쪽으로 가기보다 템포와 타점 안정성을 먼저 맞추는 편이 좋습니다. ` +
      `아이언도 ${safeString(irons.shaftType, '기본 샤프트')} 쪽이 현재 핸디캡과 재현성 기준으로 더 안전합니다.`
    );
  }

  if (/슬라이스|slice|오른쪽/i.test(question)) {
    return (
      `지금 데이터에서는 구질이 ${safeString(swing.shape, '미상')}이고 패스가 ${safeString(swing.path, '미상')}라서, 장비 쪽에서는 시작 방향 안정화가 먼저입니다. ` +
      `드라이버 로프트 ${safeString(driver.loft, '기본값')}와 드로우 바이어스 성향 헤드를 우선 체크하고, 동시에 볼 포지션과 티 높이도 같이 봐야 효과가 큽니다. ` +
      `장비만으로 완전히 해결하려 하지 말고, 시타 때 오른쪽 미스 폭이 실제로 줄어드는지 확인하세요.`
    );
  }

  if (/예산|budget|먼저|우선/i.test(question)) {
    return (
      `예산이 제한적이면 먼저 드라이버부터 건드리는 쪽이 체감 변화가 클 가능성이 큽니다. ` +
      `현재 결과에서도 드라이버는 로프트, 플렉스, 샤프트 무게가 구질과 거리 둘 다에 영향을 주고 있습니다. ` +
      `짧은 클럽은 기존 세팅 유지 후, 드라이버 시타 결과가 확실히 좋아질 때 다음 카테고리로 넘어가는 편이 효율적입니다.`
    );
  }

  if (/로프트|탄도|launch/i.test(question)) {
    return (
      `로프트는 헤드스피드 ${safeNumber(swing.headSpeed, 90)}mph와 어택 앵글 ${safeNumber(swing.attackAngle, 0)}도를 같이 봐야 합니다. ` +
      `지금 추천 로프트는 ${safeString(driver.loft, '기본값')}인데, 이 값은 탄도 확보와 스핀 안정화를 함께 노린 세팅입니다. ` +
      `실제 시타에서는 출발각과 최고점을 같이 보고, 뜨기만 하고 밀리는지 아니면 캐리가 늘어나는지 확인해야 합니다.`
    );
  }

  if (/퍼터/i.test(question)) {
    return (
      `퍼터는 현재 ${safeString(putter.type, '기본 타입')}과 ${safeString(putter.length, '기본 길이')} 조합으로 잡혀 있습니다. ` +
      `이건 스트로크 재현성과 미스 방향을 줄이는 쪽에 초점을 둔 MVP 추천입니다. ` +
      `실전에서는 거리감보다 시작 방향이 더 안정되는지 먼저 확인하는 게 좋습니다.`
    );
  }

  return (
    `현재 피팅 결과를 기준으로 보면 헤드스피드 ${safeNumber(swing.headSpeed, 90)}mph, 핸디캡 ${safeNumber(profile.handicap, 18)}, 구질 ${safeString(swing.shape, '미상')} 조합에서는 안정적인 탄도와 분산 관리가 우선입니다. ` +
    `드라이버는 ${safeString(driver.loft, '기본 로프트')} / ${safeString(driver.flex, '기본 플렉스')} 조합으로 시작하고, 아이언은 ${safeString(irons.type, '기본 타입')} 계열이 무난합니다. ` +
    `원하시면 드라이버, 아이언, 웨지, 퍼터 중 하나만 골라서 더 자세히 풀어드릴게요.`
  );
}

function buildExplainMessages(payload) {
  return [
    {
      role: 'system',
      content:
        'You are FITTED, a Korean golf fitting assistant. ' +
        'Return JSON only with keys summary, coachComment, priorities, nextSteps. ' +
        'Keep it concise, practical, and grounded in the provided data. ' +
        'Do not invent exact product claims.',
    },
    {
      role: 'user',
      content:
        'Here is the structured fitting payload.\n' +
        JSON.stringify(payload, null, 2),
    },
  ];
}

function buildChatMessages(payload, messages) {
  if (Array.isArray(messages) && messages.length) {
    if (messages[0]?.role === 'system') return messages;
    return [
      {
        role: 'system',
        content:
          'You are FITTED, a Korean golf fitting assistant. ' +
          'Use the fitting snapshot below as context and answer briefly but concretely.\n\n' +
          summarizePayload(payload),
      },
      ...messages,
    ];
  }

  return [
    {
      role: 'system',
      content:
        'You are FITTED, a Korean golf fitting assistant. ' +
        'Use the fitting snapshot below as context and answer briefly but concretely.\n\n' +
        summarizePayload(payload),
    },
    {
      role: 'user',
      content: '현재 피팅 결과를 짧게 설명해줘.',
    },
  ];
}

function extractResponseText(data, isOpenAiCompat) {
  if (isOpenAiCompat) {
    return data?.choices?.[0]?.message?.content || '';
  }
  return data?.message?.content || '';
}

function maybeParseJson(text) {
  if (!text) return null;

  const trimmed = text.trim()
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim();

  try {
    return JSON.parse(trimmed);
  } catch {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      return JSON.parse(match[0]);
    } catch {
      return null;
    }
  }
}

async function callOllama(messages, { jsonMode = false, temperature = 0.4, model = PRIMARY_OLLAMA_MODEL } = {}) {
  const openAiCompat = OLLAMA_URL.includes('/v1/');

  const body = openAiCompat
    ? {
        model,
        messages,
        stream: false,
        temperature,
        ...(jsonMode ? { response_format: { type: 'json_object' } } : {}),
      }
    : {
        model,
        messages,
        stream: false,
        options: { temperature },
        ...(jsonMode ? { format: 'json' } : {}),
      };

  const res = await fetch(OLLAMA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Ollama ${res.status}: ${detail.slice(0, 200)}`);
  }

  const data = await res.json();
  return extractResponseText(data, openAiCompat);
}

function shouldTryNextOllamaModel(error) {
  const message = String(error?.message || '').toLowerCase();
  return (
    message.includes('404') ||
    message.includes('not found') ||
    message.includes('model') ||
    message.includes('pull')
  );
}

async function callOllamaWithFallbackModels(messages, options = {}) {
  let lastError = null;

  for (const model of OLLAMA_MODEL_CANDIDATES) {
    try {
      const text = await callOllama(messages, {
        ...options,
        model,
      });
      return { text, model };
    } catch (error) {
      lastError = error;
      if (!shouldTryNextOllamaModel(error)) break;
    }
  }

  throw lastError || new Error('No Ollama model available.');
}

async function generateExplanation(payload) {
  try {
    const { text: raw, model } = await callOllamaWithFallbackModels(buildExplainMessages(payload), {
      jsonMode: true,
      temperature: 0.25,
    });
    const parsed = maybeParseJson(raw);
    if (!parsed) throw new Error('Model did not return valid JSON.');
    return {
      ...normalizeExplainOutput(payload, parsed, 'ollama'),
      model,
    };
  } catch (error) {
    return {
      ...buildExplainFallback(payload),
      debug: error.message,
    };
  }
}

async function generateChatAnswer(payload, messages) {
  const question = extractLastUserMessage(messages);

  try {
    const { text: answer, model } = await callOllamaWithFallbackModels(buildChatMessages(payload, messages), {
      jsonMode: false,
      temperature: 0.5,
    });

    const trimmed = safeString(answer);
    if (!trimmed) throw new Error('Model returned empty content.');

    return {
      mode: 'ollama',
      answer: trimmed,
      model,
    };
  } catch (error) {
    return {
      mode: 'fallback',
      answer: buildChatFallback(payload, question),
      debug: error.message,
    };
  }
}

function safeResolvePath(pathname) {
  const decoded = decodeURIComponent(pathname);
  const requestedPath = decoded === '/' ? '/index.html' : decoded;
  const filePath = path.normalize(path.join(ROOT_DIR, requestedPath));

  if (!filePath.startsWith(ROOT_DIR)) return null;
  return filePath;
}

function serveStatic(res, pathname, method) {
  const filePath = safeResolvePath(pathname);
  if (!filePath) {
    sendText(res, 403, 'Forbidden');
    return;
  }

  fs.stat(filePath, (error, stat) => {
    if (error || !stat.isFile()) {
      sendText(res, 404, 'Not found');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    res.writeHead(200, withCors({ 'Content-Type': contentType }));

    if (method === 'HEAD') {
      res.end();
      return;
    }

    fs.createReadStream(filePath).pipe(res);
  });
}

function createServer() {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || '127.0.0.1'}`);

    if (req.method === 'OPTIONS') {
      res.writeHead(204, withCors());
      res.end();
      return;
    }

    if (url.pathname === '/api/health' && req.method === 'GET') {
      sendJson(res, 200, {
        ok: true,
        ollamaUrl: OLLAMA_URL,
        ollamaModel: PRIMARY_OLLAMA_MODEL,
        ollamaModelCandidates: OLLAMA_MODEL_CANDIDATES,
      });
      return;
    }

    if (url.pathname === '/api/fitting/explain' && req.method === 'POST') {
      try {
        const body = await readJson(req);
        const result = await generateExplanation(body);
        sendJson(res, 200, result);
      } catch (error) {
        sendJson(res, 400, { error: error.message });
      }
      return;
    }

    if (url.pathname === '/api/fitting/chat' && req.method === 'POST') {
      try {
        const body = await readJson(req);
        const result = await generateChatAnswer(body.payload || {}, body.messages || []);
        sendJson(res, 200, result);
      } catch (error) {
        sendJson(res, 400, { error: error.message });
      }
      return;
    }

    if (req.method === 'GET' || req.method === 'HEAD') {
      serveStatic(res, url.pathname, req.method);
      return;
    }

    sendJson(res, 404, { error: 'Not found.' });
  });
}

if (require.main === module) {
  const server = createServer();
  server.listen(PORT, () => {
    console.log(`FITTED local server listening on http://127.0.0.1:${PORT}`);
    console.log(`Ollama endpoint: ${OLLAMA_URL}`);
    console.log(`Primary Ollama model: ${PRIMARY_OLLAMA_MODEL}`);
    console.log(`Ollama model candidates: ${OLLAMA_MODEL_CANDIDATES.join(', ')}`);
  });
}

module.exports = {
  createServer,
  summarizePayload,
  buildExplainFallback,
  buildChatFallback,
  generateExplanation,
  generateChatAnswer,
};
