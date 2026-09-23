import { describe, expect, it } from 'vitest';
import {
  collectToolCalls,
  isPrompt,
  compact,
  decideCall,
  isMutatingTool,
  parseToolThresholds,
  pinStartIndex,
  ruleDecisions,
  type JevAsker,
  type JevQuestions,
  type Message,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const big = 'line of output\n'.repeat(100);

function askAll(answer: number, seen: string[][] = []): JevAsker {
  return {
    async ask(_state, questions: JevQuestions) {
      seen.push(Object.keys(questions));
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer }]),
        ),
      };
    },
  };
}

describe('rule pre-pass', () => {
  const messages = (): Message[] => [
    message('user', 'refactor a.ts'),
    call('r1', 'Read', { file_path: 'src/a.ts' }),
    result('r1', big),
    call('b1', 'Bash', { command: 'npm test' }),
    result('b1', big, true),
    call('e1', 'Edit', { file_path: 'src/a.ts', old_string: 'x', new_string: 'y' }),
    result('e1', 'ok'),
    call('r2', 'Read', { file_path: 'src/b.ts' }),
    result('r2', big),
    call('r3', 'Read', { file_path: 'src/b.ts' }),
    result('r3', big),
    call('b2', 'Bash', { command: 'npm test' }),
    result('b2', big),
    message('assistant', 'done'),
    message('user', 'thanks'),
  ];

  it('marks reads superseded by a later edit or identical read, and repeated commands', () => {
    const calls = collectToolCalls(messages(), { preserveRecentMessages: 0 });
    const rules = ruleDecisions(calls);
    const byTool = Object.fromEntries(calls.map((c) => [c.tool_use_id, rules.get(c.id)]));
    expect(byTool).toEqual({
      r1: 'superseded',
      b1: 'repeated',
      e1: undefined,
      r2: 'repeated',
      r3: undefined,
      b2: undefined,
    });
  });

  it('does not treat reads of different ranges as repeats', () => {
    const calls = collectToolCalls(
      [
        message('user', 'x'),
        call('r1', 'Read', { file_path: 'a', offset: 1, limit: 10 }),
        result('r1', big),
        call('r2', 'Read', { file_path: 'a', offset: 50, limit: 10 }),
        result('r2', big),
      ],
      { preserveRecentMessages: 0 },
    );
    expect(ruleDecisions(calls).size).toBe(0);
  });

  it('never touches pinned calls', () => {
    const calls = collectToolCalls(messages(), { preserveRecentMessages: 20 });
    expect(ruleDecisions(calls).size).toBe(0);
  });

  it('does not ask Jev about rule-decided calls and truncates their results', async () => {
    const seen: string[][] = [];
    const out = await compact(messages(), askAll(0.9, seen), { preserveRecentMessages: 0, preserveRecentPrompts: 0 });
    const asked = seen.flat();
    const ids = Object.fromEntries(out.decisions.map((d) => [d.id, d]));
    const idOf = (toolUseId: string) =>
      collectToolCalls(messages(), { preserveRecentMessages: 0 }).find((c) => c.tool_use_id === toolUseId)!.id;
    expect(asked).not.toContain(`result_${idOf('r1')}`);
    expect(ids[idOf('r1')]).toMatchObject({ action: 'drop_result', reason: 'superseded' });
    expect(ids[idOf('b1')]).toMatchObject({ action: 'drop_result', reason: 'repeated' });
    expect(out.stats.ruleDecided).toBe(3);
  });
});

describe('mutating tools', () => {
  it('are recognised by name', () => {
    expect(isMutatingTool('Edit')).toBe(true);
    expect(isMutatingTool('Write')).toBe(true);
    expect(isMutatingTool('NotebookEdit')).toBe(true);
    expect(isMutatingTool('Read')).toBe(false);
  });

  it('keep at least a stub of the call even when Jev says drop it', () => {
    const decision = decideCall(
      { id: 't1', tool: 'Edit', pinned: false },
      { keepCall: 0.1, keepResult: 0.1 },
      { keepThreshold: 0.5, toolThresholds: {} },
    );
    expect(decision.action).toBe('drop_result');
  });
});

describe('per-tool thresholds', () => {
  it('parses "Tool=0.7, Other=0.3" and ignores junk', () => {
    expect(parseToolThresholds('Read=0.7, Grep=0.3, bad, X=nope')).toEqual({ Read: 0.7, Grep: 0.3 });
    expect(parseToolThresholds(undefined)).toEqual({});
  });

  it('override the global threshold for that tool', () => {
    const options = { keepThreshold: 0.5, toolThresholds: { Read: 0.8 } };
    expect(decideCall({ id: 't', tool: 'Read', pinned: false }, { keepCall: 0.9, keepResult: 0.7 }, options).action).toBe(
      'drop_result',
    );
    expect(decideCall({ id: 't', tool: 'Bash', pinned: false }, { keepCall: 0.9, keepResult: 0.7 }, options).action).toBe(
      'keep',
    );
  });
});

describe('pinning by prompts', () => {
  const convo = (): Message[] => [
    message('user', 'first'),
    call('a', 'Bash', { command: 'a' }),
    result('a', 'x'),
    message('user', 'second'),
    call('b', 'Bash', { command: 'b' }),
    result('b', 'x'),
    message('assistant', 'ok'),
    message('user', 'third'),
    call('c', 'Bash', { command: 'c' }),
    result('c', 'x'),
    call('d', 'Bash', { command: 'd' }),
    result('d', 'x'),
  ];

  it('pins from the Nth-last real user prompt, capped by maxPinnedMessages', () => {
    expect(pinStartIndex(convo(), { preserveRecentMessages: 0, preserveRecentPrompts: 2, maxPinnedMessages: 100 })).toBe(3);
    expect(pinStartIndex(convo(), { preserveRecentMessages: 0, preserveRecentPrompts: 1, maxPinnedMessages: 100 })).toBe(7);
    expect(pinStartIndex(convo(), { preserveRecentMessages: 0, preserveRecentPrompts: 2, maxPinnedMessages: 4 })).toBe(8);
  });

  it('always pins at least preserveRecentMessages', () => {
    expect(pinStartIndex(convo(), { preserveRecentMessages: 6, preserveRecentPrompts: 0, maxPinnedMessages: 100 })).toBe(6);
  });

  it('never counts tool results as prompts', () => {
    const calls = collectToolCalls(convo(), { preserveRecentMessages: 0, preserveRecentPrompts: 1, maxPinnedMessages: 100 });
    expect(calls.map((c) => c.pinned)).toEqual([false, false, true, true]);
  });
});

describe('prompt detection', () => {
  it('ignores engine-written user text', () => {
    for (const text of [
      '<command-name>/clear</command-name>\n<command-message>clear</command-message>',
      '<local-command-stdout>Compacted</local-command-stdout>',
      '<system-reminder>x</system-reminder>',
      '<task-notification>done</task-notification>',
      '[fast-jev-compaction: the earlier part of this conversation was summarised]\n\nnotes',
      'This session is being continued from a previous conversation that ran out of context.',
    ]) {
      expect(isPrompt(message('user', text))).toBe(false);
    }
    expect(isPrompt(message('user', 'fix the <div> layout'))).toBe(true);
    expect(isPrompt(message('assistant', 'hi'))).toBe(false);
  });
});
