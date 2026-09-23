import type { ToolCall } from './types.js';

/** Tools that change files: dropping the call would erase the record that a change happened. */
const MUTATING_TOOLS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit']);

/** Tools whose output is a file's contents at the time of the call. */
const READ_TOOLS = new Set(['Read']);

export function isMutatingTool(tool: string): boolean {
  return MUTATING_TOOLS.has(tool);
}

function filePath(call: ToolCall): string | undefined {
  const { file_path, notebook_path, path } = call.input;
  for (const value of [file_path, notebook_path, path]) {
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
}

function inputKey(call: ToolCall): string {
  let json = '';
  try {
    json = JSON.stringify(call.input);
  } catch {
    return `${call.tool}#${call.tool_use_id}`;
  }
  return `${call.tool} ${json}`;
}

export type RuleReason = 'superseded' | 'repeated';

/**
 * Decides, without asking Jev, calls whose output is stale by construction: a
 * read of a file that a later call edits or writes (`superseded`), and any
 * non-mutating call that a later call repeats with identical input
 * (`repeated`). The call stays; only its output goes. Pinned calls are never
 * decided here. Keys are the calls' short ids.
 */
export function ruleDecisions(calls: readonly ToolCall[]): Map<string, RuleReason> {
  const decided = new Map<string, RuleReason>();
  const laterInputs = new Set<string>();
  const laterWrites = new Set<string>();
  for (let index = calls.length - 1; index >= 0; index -= 1) {
    const call = calls[index]!;
    const key = inputKey(call);
    const path = filePath(call);
    if (!call.pinned) {
      if (READ_TOOLS.has(call.tool) && path !== undefined && laterWrites.has(path)) {
        decided.set(call.id, 'superseded');
      } else if (!isMutatingTool(call.tool) && laterInputs.has(key)) {
        decided.set(call.id, 'repeated');
      }
    }
    laterInputs.add(key);
    if (isMutatingTool(call.tool) && path !== undefined) laterWrites.add(path);
  }
  return decided;
}

/** Parses `"Read=0.7, Grep=0.3"` into per-tool thresholds; malformed entries are skipped. */
export function parseToolThresholds(text: string | undefined): Record<string, number> {
  const thresholds: Record<string, number> = {};
  for (const entry of (text ?? '').split(',')) {
    const [name, value] = entry.split('=').map((part) => part.trim());
    if (!name || value === undefined || value === '') continue;
    const number = Number(value);
    if (Number.isFinite(number)) thresholds[name] = number;
  }
  return thresholds;
}
