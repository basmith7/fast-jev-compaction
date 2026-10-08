import { describe, expect, it } from 'vitest';
import {
  FORK_SKIP,
  register,
  archiveWriter,
  compactSession,
  looksSecret,
  decisionLog,
  decisionLogLines,
  recordDecisions,
  recordPath,
  resolveHookConfig,
  summarize,
  summaryCut,
  summaryMessage,
  summaryPrompt,
  SUMMARY_HEADER,
  toSessionMessages,
  transcriptTokens,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      compactAtTokens: 250_000,
      summaryAboveTokens: 40_000,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      archiveResults: true,
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      compactAtTokens: 250_000,
      summaryAboveTokens: 40_000,
      minReductionRatio: 0.25,
      archiveResults: true,
    });
    expect(resolveHookConfig({ archiveResults: false }).archiveResults).toBe(false);
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, preserveRecentPrompts: 0 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, preserveRecentPrompts: 0 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1, preserveRecentPrompts: 0 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('second tier', () => {
  const convo = (): SessionMessage[] => [
    message('user', 'first ask', { handle: 'p0' }),
    message('assistant', 'long answer '.repeat(50), { handle: 'a0' }),
    message('user', 'second ask', { handle: 'p1' }),
    message('assistant', 'ok', { handle: 'a1' }),
    message('user', 'third ask', { handle: 'p2' }),
  ];

  it('cuts at the Nth-newest prompt and never at the first message', () => {
    expect(summaryCut(convo(), 2)).toBe(2);
    expect(summaryCut(convo(), 1)).toBe(4);
    expect(summaryCut(convo(), 0)).toBe(4);
    expect(summaryCut(convo(), 3)).toBe(-1);
    expect(summaryCut([message('assistant', 'x')], 1)).toBe(-1);
  });

  it('asks for a summary of only what precedes the cut', () => {
    const prompt = summaryPrompt('second   ask\nwith detail');
    expect(prompt).toContain('> second ask with detail');
    expect(prompt).toMatch(/BEFORE/);
    expect(summaryMessage('  notes ').text).toBe(`${SUMMARY_HEADER}\n\nnotes`);
    expect(summaryMessage('x').handle).toBeUndefined();
  });

  it('estimates transcript tokens across text, inputs and outputs', () => {
    const base = transcriptTokens([message('user', 'hello world')]);
    expect(base).toBeGreaterThan(0);
    expect(transcriptTokens(transcript())).toBeGreaterThan(base);
  });
});

describe('decision record', () => {
  it('pairs each decision with its call input and writes under a safe file name', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, preserveRecentPrompts: 0 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const decisions = recordDecisions(transcript(), output);
    expect(decisions.map((d) => [d.id, d.tool, d.input])).toEqual([
      ['t1', 'Read', '{"file_path":"src/a.ts"}'],
      ['t2', 'Bash', '{"command":"npm test"}'],
    ]);
    expect(recordPath('/x/logs/', '2026-09-22T10:00:00.000Z', 'auto')).toBe('/x/logs/2026-09-22T10-00-00-000Z-auto.json');
    expect(recordPath('/x/logs', '2026-09-22T10:00:00.000Z', 'auto', 'a1/../b')).toBe(
      '/x/logs/2026-09-22T10-00-00-000Z-auto-a1____b.json',
    );
  });

  it('reads the new options', () => {
    const config = resolveHookConfig({
      toolThresholds: 'Read=0.7',
      compactAtTokens: 100_000,
      summaryAboveTokens: 0,
      preserveRecentPrompts: 2,
      logDir: '/tmp/x',
    });
    expect(config).toMatchObject({
      toolThresholds: { Read: 0.7 },
      compactAtTokens: 100_000,
      summaryAboveTokens: 0,
      preserveRecentPrompts: 2,
      logDir: '/tmp/x',
    });
  });
});

describe('archiveWriter', () => {
  const fsStub = () => {
    const files = new Map<string, string>();
    return {
      files,
      fs: {
        exists: async (path: string) => files.has(path),
        write: async (path: string, text: string) => void files.set(path, text),
      },
    };
  };
  const output = (text: string, input: Record<string, unknown> = { command: 'cat server.log' }) => ({
    tool_use_id: 'toolu_1',
    tool: 'Bash',
    input,
    text,
  });

  it('names a file per output and writes them on flush, with a .gitignore', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub);
    const path = writer.cite(output('a'.repeat(5000)));
    expect(path).toBe('.claude/fast-jev-compaction/result-toolu_1.txt');
    expect(stub.files.size).toBe(0);
    await writer.flush();
    expect(stub.files.get('.claude/fast-jev-compaction/.gitignore')).toBe('*\n');
    expect(stub.files.get(path!)?.length).toBe(5000);
  });

  it('keeps a tool_use_id from escaping the archive folder', () => {
    const writer = archiveWriter(fsStub());
    expect(writer.cite({ ...output('a'), tool_use_id: '../../x' })).toBe(
      '.claude/fast-jev-compaction/result-______x.txt',
    );
  });

  it('never writes output that looks like credentials, wherever it shows', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub);
    expect(writer.cite(output(`${'ok\n'.repeat(10_000)}AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG`))).toBeUndefined();
    expect(writer.cite(output('PORT=3000', { file_path: '/srv/app/.env' }))).toBeUndefined();
    expect(writer.cite(output('token: ghp_abcdefghijklmnopqrstuvwxyz0123456789'))).toBeUndefined();
    await writer.flush();
    expect(stub.files.size).toBe(0);
  });

  it('writes nothing when no output was cited', async () => {
    const stub = fsStub();
    await archiveWriter(stub).flush();
    expect(stub.files.size).toBe(0);
  });

  it('flags secrets in input or output only', () => {
    expect(looksSecret({ command: 'npm test' }, 'PASS')).toBe(false);
    expect(looksSecret({ file_path: 'src/tokenizer.ts' }, 'export {}')).toBe(false);
    expect(looksSecret({ command: 'printenv' }, '')).toBe(true);
    expect(looksSecret({}, 'postgres://app:hunter2@db/app')).toBe(true);
  });
});

describe('compactSession archive', () => {
  it('writes the output of a dropped call and cites it', async () => {
    const files = new Map<string, string>();
    const writer = archiveWriter({
      fs: { exists: async (p) => files.has(p), write: async (p, t) => void files.set(p, t) },
    });
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, preserveRecentPrompts: 0 }), apiKey: 'k' };
    const { messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name.endsWith('t1') ? 0.1 : 0.9)),
      writer,
    );
    expect(files.get('.claude/fast-jev-compaction/result-tool-1.txt')).toBe(fileA);
    expect(messages[1]?.text).toContain('full output: .claude/fast-jev-compaction/result-tool-1.txt');
  });
});

describe('session.compact under the summary fork', () => {
  function engine(onFork: (compact: Handler, $: unknown) => Promise<unknown>) {
    const handlers = new Map<string, Handler>();
    const files = new Map<string, string>();
    const forkResults: unknown[] = [];
    const $ = {
      env: { get: async (name: string) => (name === 'TYPESAFE_API_KEY' ? 'k' : undefined) },
      settings: { read: async () => ({}) },
      http: { fetch: jevFetch((name) => (name.endsWith('t1') ? 0.1 : 0.9)) },
      ui: { log: () => {}, toast: () => {} },
      fs: { exists: async (p: string) => files.has(p), write: async (p: string, t: string) => void files.set(p, t) },
      model: {
        fork: async () => {
          forkResults.push(await onFork(handlers.get('session.compact')!, $));
          return { isAnswered: true, text: 'SUMMARY', usage: {} };
        },
      },
    };
    const on = (name: string, handler: Handler) => void handlers.set(name, handler);
    register(on as never, {
      preserveRecentMessages: 1,
      preserveRecentPrompts: 1,
      summaryAboveTokens: 1,
      logDir: 'logs',
      archiveResults: false,
    });
    return { $, files, forkResults, compact: handlers.get('session.compact')! };
  }
  type Handler = (...args: any[]) => Promise<any>;
  const main = () => [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'Read it.', { handle: 'h-3' }),
    message('user', 'now run the tests', { handle: 'h-4' }),
    message('assistant', 'Running.', { handle: 'h-5' }),
  ];
  const core = async () => ({ messages: [message('user', 'core summary')] });

  it('skips the fork\'s own compaction and records why', async () => {
    const { $, files, forkResults, compact } = engine((handler, engine$) =>
      handler(engine$, { trigger: 'auto', agentId: 'fork-1', messages: main().slice(0, -1) }, core),
    );
    const out = await compact($, { trigger: 'auto', messages: main() }, core);
    expect(forkResults).toEqual([{ skip: FORK_SKIP }]);
    expect(out.messages[0].text).toContain('SUMMARY');
    const records = [...files.entries()].filter(([path]) => path.startsWith('logs/')).map(([, t]) => JSON.parse(t));
    expect(records.map((r) => [r.outcome, r.agentId ?? null])).toEqual([
      ['skipped', 'fork-1'],
      ['jev+summary', null],
    ]);
  });

  it('still compacts a subagent that is not the fork, during the fork and after it', async () => {
    const subagent = () => [message('user', 'Survey the repo.'), ...main().slice(1)];
    const { $, forkResults, compact } = engine((handler, engine$) =>
      handler(engine$, { trigger: 'auto', agentId: 'sub-1', messages: subagent() }, core),
    );
    await compact($, { trigger: 'auto', messages: main() }, core);
    expect(forkResults[0]).not.toEqual({ skip: FORK_SKIP });
    const after = await compact($, { trigger: 'auto', agentId: 'fork-2', messages: main() }, core);
    expect(after).not.toEqual({ skip: FORK_SKIP });
  });
});
