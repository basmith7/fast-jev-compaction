import type {
  On,
  PluginOptions,
  Register,
  SessionCompactInput,
  SessionCompactResult,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, messageChars, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import { parseToolThresholds } from '../src/rules.js';
import { estimateTokens, isPrompt, truncate } from '../src/state.js';
import type {
  ArchiveResult,
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  archiveResults: true,
  compactAtPercent: 60,
  compactAtTokens: 250_000,
  summaryAboveTokens: 40_000,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  /** Context tokens at which turn.complete requests compaction; 0 leaves it to the percentage. */
  compactAtTokens: number;
  /** Above this many tokens after pruning, older text is summarised too; 0 turns that off. */
  summaryAboveTokens: number;
  minReductionRatio: number;
  model: string;
  /** Where each compaction's decision record is written; '' for the default under $HOME. */
  logDir?: string;
  /** Save the output of dropped calls and truncated results under ARCHIVE_DIR. */
  archiveResults: boolean;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'preserveRecentPrompts',
    'maxPinnedMessages',
    'resultPreviewChars',
    'retries',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    compactAtTokens: optionNumber(options, 'compactAtTokens', HOOK_DEFAULTS.compactAtTokens),
    summaryAboveTokens: optionNumber(options, 'summaryAboveTokens', HOOK_DEFAULTS.summaryAboveTokens),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    archiveResults:
      typeof options['archiveResults'] === 'boolean'
        ? options['archiveResults']
        : HOOK_DEFAULTS.archiveResults,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  const thresholds = parseToolThresholds(optionString(options, 'toolThresholds'));
  if (Object.keys(thresholds).length > 0) config.toolThresholds = thresholds;
  const logDir = optionString(options, 'logDir');
  if (logDir) config.logDir = logDir;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/**
 * Runs the library over a session transcript; throws when the key is missing or
 * Jev fails. With `archive`, the files it named are written before returning.
 */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  archive?: ArchiveWriter,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), {
    ...config,
    ...(archive ? { archive: archive.cite } : {}),
  });
  if (archive) await archive.flush();
  return { result, messages: toSessionMessages(messages, result.messages) };
}

/** Project-relative, so the agent can Read or grep a saved output without leaving the project. */
export const ARCHIVE_DIR = '.claude/fast-jev-compaction';

const SECRET_INPUT =
  /(^|[|;&\s"'])(printenv|env)\b|\.env\b|\b(secret|secrets|credential|credentials|password|passwd|token|keychain|netrc|id_rsa|id_ed25519|private[_-]?key)\b|\.(pem|key|p12|pfx)\b/i;
const SECRET_OUTPUT =
  /-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(aws_secret_access_key|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret[_-]?key|password)\s*["']?\s*[=:]\s*\S|:\/\/[^\s:@/]+:[^\s:@/]+@/i;
/** Well-known token formats (AWS, GitHub, OpenAI-style, Slack), matched case-sensitively. */
const SECRET_TOKEN =
  /\b(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,})/;

/** True when a tool call's input or output looks like it carries credentials. */
export function looksSecret(input: Record<string, unknown>, output: string): boolean {
  let inputText = '';
  try {
    inputText = JSON.stringify(input) ?? '';
  } catch {
    inputText = '';
  }
  return SECRET_INPUT.test(inputText) || SECRET_OUTPUT.test(output) || SECRET_TOKEN.test(output);
}

export type ArchiveWriter = {
  /** Names the file an output will be written to, for its marker; undefined skips it. */
  cite: ArchiveResult;
  /** Writes the files named so far. */
  flush: () => Promise<void>;
};

/**
 * Saves the output compaction removes, so a detail buried in it can be read
 * back later instead of being lost. Nothing that looks like a credential is
 * written; nothing at all is written when no output was cited.
 */
export function archiveWriter($: {
  fs: { exists: (path: string) => Promise<boolean>; write: (path: string, text: string) => Promise<void> };
}): ArchiveWriter {
  const pending = new Map<string, string>();
  return {
    cite({ tool_use_id, input, text }) {
      if (!text || looksSecret(input, text)) return undefined;
      const path = `${ARCHIVE_DIR}/result-${tool_use_id.replace(/[^\w-]/g, '_')}.txt`;
      pending.set(path, text);
      return path;
    },
    async flush() {
      if (pending.size === 0) return;
      const ignorePath = `${ARCHIVE_DIR}/.gitignore`;
      if (!(await $.fs.exists(ignorePath))) await $.fs.write(ignorePath, '*\n');
      for (const [path, text] of pending) await $.fs.write(path, text);
      pending.clear();
    },
  };
}

/** Estimated tokens of a transcript: text, tool inputs and tool outputs. */
export function transcriptTokens(messages: readonly Message[]): number {
  let tokens = 0;
  for (const message of messages) {
    tokens += estimateTokens(message.text);
    for (const tool of message.toolUses) {
      tokens += estimateTokens(JSON.stringify(tool.input) ?? '') + estimateTokens(tool.text ?? '');
    }
    for (const result of message.toolResults ?? []) tokens += estimateTokens(result.text);
  }
  return tokens;
}

/**
 * Where the second tier cuts: the Nth-newest prompt of the pruned transcript
 * (at least the newest one). Everything before it is summarised; it and what
 * follows stay verbatim. -1 when there is nothing before it to summarise.
 */
export function summaryCut(messages: readonly Message[], recentPrompts: number): number {
  const prompts: number[] = [];
  messages.forEach((message, index) => {
    if (isPrompt(message)) prompts.push(index);
  });
  const cut = prompts[Math.max(0, prompts.length - Math.max(1, recentPrompts))];
  return cut === undefined || cut <= 0 ? -1 : cut;
}

export const SUMMARY_HEADER =
  '[fast-jev-compaction: the earlier part of this conversation was summarised; what follows the summary is verbatim]';

/** The question the fork answers: a summary of everything before the cut. */
export function summaryPrompt(cutText: string): string {
  return [
    'This conversation is being compacted. Summarise ONLY the part that comes BEFORE the user message that begins:',
    '',
    `> ${truncate(cutText.replace(/\s+/g, ' '), 300)}`,
    '',
    'That message and everything after it stay verbatim, so do not summarise them. Your summary replaces the earlier part entirely, so keep what the work still depends on:',
    "- the user's goals, requests, preferences and constraints, quoting short ones exactly",
    '- decisions made and why, including approaches that were tried and rejected',
    '- files created, modified or deleted, and their current state',
    '- results of commands or tests that still matter, and errors still unresolved',
    "- corrections the user made to the assistant's work",
    '- open tasks and agreed next steps',
    'Write dense, factual notes under short headings. No preamble. Output only the summary.',
  ].join('\n');
}

export function summaryMessage(text: string): SessionMessage {
  return { role: 'user', text: `${SUMMARY_HEADER}\n\n${text.trim()}`, toolUses: [] };
}

export type CompactionRecord = {
  time: string;
  trigger: string;
  agentId?: string;
  outcome: 'jev' | 'jev+summary' | 'fallback';
  reason?: string;
  messagesBefore: number;
  /** One `role:textChars:toolUses:toolResults[:P]` per message (P marks a prompt), for diagnosing pinning. */
  shape?: string[];
  messagesAfter?: number;
  tokensBefore: number;
  tokensAfterPrune?: number;
  tokensAfter?: number;
  stats?: CompactResult['stats'];
  summary?: { chars: number; usage?: unknown } | { skipped: string };
  decisions?: Array<{
    id: string;
    tool: string;
    action: string;
    reason: string;
    keepCall: number;
    keepResult: number;
    input: string;
  }>;
};

/** The decisions as the record keeps them: each call's input, cut short, beside the verdict. */
export function recordDecisions(
  messages: readonly Message[],
  result: CompactResult,
): NonNullable<CompactionRecord['decisions']> {
  const inputs = new Map<string, string>();
  let n = 0;
  const results = new Set<string>();
  for (const message of messages) for (const r of message.toolResults ?? []) results.add(r.tool_use_id);
  for (const message of messages) {
    for (const tool of message.toolUses) {
      if (!results.has(tool.tool_use_id)) continue;
      n += 1;
      inputs.set(`t${n}`, truncate(JSON.stringify(tool.input) ?? '', 200));
    }
  }
  return result.decisions.map((d) => ({
    id: d.id,
    tool: d.tool,
    action: d.action,
    reason: d.reason,
    keepCall: Math.round(d.keepCall * 1000) / 1000,
    keepResult: Math.round(d.keepResult * 1000) / 1000,
    input: inputs.get(d.id) ?? '',
  }));
}

export function messageShape(messages: readonly Message[]): string[] {
  return messages.map(
    (m) =>
      `${m.role[0]}:${m.text.length}:${m.toolUses.length}:${m.toolResults?.length ?? 0}${isPrompt(m) ? ':P' : ''}`,
  );
}

export function recordPath(dir: string, time: string, trigger: string): string {
  return `${dir.replace(/\/+$/, '')}/${time.replace(/[:.]/g, '-')}-${trigger}.json`;
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

type Engine = Parameters<Parameters<On>[2]>[0];

async function writeRecord($: Engine, config: HookConfig, record: CompactionRecord): Promise<void> {
  try {
    const home = await $.env.get('HOME');
    const dir = config.logDir || (home ? `${home}/.local/state/fast-jev-compaction` : '');
    if (!dir) return;
    await $.fs.write(recordPath(dir, record.time, record.trigger), JSON.stringify(record, null, 1));
  } catch (error) {
    $.ui.log(`could not write the compaction record (${error instanceof Error ? error.message : String(error)})`);
  }
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.compact', async ($, event: SessionCompactInput, next) => {
    const record: CompactionRecord = {
      time: new Date().toISOString(),
      trigger: event.trigger,
      outcome: 'fallback',
      messagesBefore: event.messages.length,
      shape: messageShape(event.messages),
      tokensBefore: transcriptTokens(event.messages),
    };
    if (event.agentId) record.agentId = event.agentId;
    const fallback = async (reason: string) => {
      notify($, `fallback to built-in summary (${reason})`);
      record.reason = reason;
      await writeRecord($, configured, record);
      return next(event);
    };
    let pruned: SessionCompaction;
    const config = { ...configured, apiKey: await getApiKey($, configured).catch(() => undefined) };
    try {
      pruned = await compactSession(
        event.messages,
        config,
        async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text };
        },
        config.archiveResults ? archiveWriter($) : undefined,
      );
    } catch (error) {
      return fallback(error instanceof Error ? error.message : String(error));
    }
    const { result } = pruned;
    let messages = pruned.messages;
    record.stats = result.stats;
    record.decisions = recordDecisions(event.messages, result);
    record.tokensAfterPrune = transcriptTokens(messages);
    for (const line of decisionLogLines(result)) $.ui.log(line);

    const summarising =
      config.summaryAboveTokens > 0 &&
      record.tokensAfterPrune > config.summaryAboveTokens &&
      event.trigger !== 'precompute' &&
      !event.agentId;
    if (summarising) {
      const cut = summaryCut(messages, config.preserveRecentPrompts ?? resolveOptions().preserveRecentPrompts);
      if (cut < 0) {
        record.summary = { skipped: 'no earlier prompt to cut at' };
      } else {
        const reply = await $.model.fork({ prompt: summaryPrompt(messages[cut]!.text) });
        if (reply.isAnswered && reply.text.trim()) {
          messages = [summaryMessage(reply.text), ...messages.slice(cut)];
          record.summary = { chars: reply.text.length, usage: reply.usage };
          record.outcome = 'jev+summary';
        } else {
          // No fork (a session resumed straight into /compact has no reply to
          // fork from): the engine's own summarizer over the older part alone.
          const forkReason = reply.isAnswered ? 'empty reply' : reply.reason;
          const core: SessionCompactResult = await next({ ...event, messages: messages.slice(0, cut) }).catch(
            (error: unknown) => ({ skip: error instanceof Error ? error.message : String(error) }),
          );
          if (core.messages && core.messages.length > 0) {
            const summaryChars = core.messages.reduce((sum, m) => sum + messageChars(m), 0);
            messages = [...core.messages, ...messages.slice(cut)];
            record.summary = { chars: summaryChars, usage: `core summarizer (fork: ${forkReason})` };
            record.outcome = 'jev+summary';
          } else {
            record.summary = { skipped: `fork: ${forkReason}; core: ${core.skip ?? 'no messages'}` };
          }
        }
      }
    }

    record.messagesAfter = messages.length;
    record.tokensAfter = transcriptTokens(messages);
    const reduction = record.tokensBefore === 0 ? 0 : 1 - record.tokensAfter / record.tokensBefore;
    if (record.outcome !== 'jev+summary' && reductionRatio(result) < config.minReductionRatio) {
      return fallback(`below ${percent(config.minReductionRatio)} minimum: ${summarize(result)}`);
    }
    if (record.outcome === 'fallback') record.outcome = 'jev';
    await writeRecord($, configured, record);
    notify(
      $,
      `kept ${messages.length}/${event.messages.length} messages, ${
        record.outcome === 'jev+summary' ? 'older text summarised' : 'no summary'
      }, ~${percent(reduction)} smaller (${summarize(result)})`,
    );
    return { messages };
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      const byTokens = configured.compactAtTokens > 0 && (context.tokens ?? 0) >= configured.compactAtTokens;
      const byPercent = (context.percent ?? 0) >= configured.compactAtPercent;
      if (!byTokens && !byPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
