export type RoundRow = {
  round?: number | string | null;
  date: string;
  kickoff_utc?: string | null;
};

export function groupByRound<T extends RoundRow>(rows: T[]): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const row of rows) {
    const key = String(row.round || "Later");
    groups.set(key, [...(groups.get(key) || []), row]);
  }

  return [...groups.entries()].sort((a, b) => {
    const dateDifference = firstKickoff(a[1]) - firstKickoff(b[1]);
    return dateDifference || roundNumber(a[0]) - roundNumber(b[0]);
  });
}

export function selectFocusRound<T extends RoundRow>(
  upcoming: T[],
  played: T[],
  generatedAt: string,
): [string, T[]] {
  const upcomingRounds = groupByRound(upcoming);
  if (!upcomingRounds.length) return ["volgt", []];

  const allRounds = groupByRound([...played, ...upcoming]);
  const regularRoundSize = Math.max(0, ...allRounds.map(([, rows]) => rows.length));
  const minimumRoundSize = Math.min(5, regularRoundSize);
  const reference = new Date(generatedAt).getTime();
  const recentCutoff = Number.isFinite(reference)
    ? reference - 4 * 24 * 60 * 60 * 1000
    : Number.NEGATIVE_INFINITY;

  for (const [round, rows] of upcomingRounds) {
    const completeRound = allRounds.find(([candidate]) => candidate === round)?.[1] || rows;
    if (completeRound.length >= minimumRoundSize && medianKickoff(completeRound) >= recentCutoff) {
      return [round, rows];
    }
  }

  const largestRemainingRound = Math.max(...upcomingRounds.map(([, rows]) => rows.length));
  return upcomingRounds.find(([, rows]) => rows.length === largestRemainingRound) || upcomingRounds[0];
}

function timestamp(row: RoundRow): number {
  const parsed = new Date(row.kickoff_utc || row.date).getTime();
  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
}

function firstKickoff(rows: RoundRow[]): number {
  return Math.min(...rows.map(timestamp));
}

function medianKickoff(rows: RoundRow[]): number {
  const values = rows.map(timestamp).sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)] ?? Number.MAX_SAFE_INTEGER;
}

function roundNumber(round: string): number {
  const parsed = Number(round);
  return Number.isFinite(parsed) ? parsed : 999;
}
