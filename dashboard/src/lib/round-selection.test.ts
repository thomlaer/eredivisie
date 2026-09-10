import assert from "node:assert/strict";
import test from "node:test";
import { selectFocusRound, type RoundRow } from "./round-selection.ts";

const match = (round: number, date: string): RoundRow => ({ round, date });
const matches = (round: number, date: string, count = 9): RoundRow[] =>
  Array.from({ length: count }, () => match(round, date));

test("old catch-up matches do not replace the next full round", () => {
  const upcoming = [
    ...matches(3, "2026-09-09", 2),
    ...matches(5, "2026-09-08", 1),
    ...matches(6, "2026-09-12"),
    ...matches(7, "2026-09-19"),
  ];
  const played = [...matches(3, "2026-08-22", 7), ...matches(5, "2026-08-29", 8)];

  assert.equal(selectFocusRound(upcoming, played, "2026-09-09T08:00:00Z")[0], "6");
});

test("a round stays active while some of its matches remain", () => {
  const upcoming = [...matches(6, "2026-09-13", 4), ...matches(7, "2026-09-19")];
  const played = matches(6, "2026-09-12", 5);

  const [round, rows] = selectFocusRound(upcoming, played, "2026-09-12T20:00:00Z");
  assert.equal(round, "6");
  assert.equal(rows.length, 4);
});

test("the next round becomes active after the previous round is complete", () => {
  const upcoming = matches(7, "2026-09-19");
  const played = matches(6, "2026-09-12");

  assert.equal(selectFocusRound(upcoming, played, "2026-09-14T08:00:00Z")[0], "7");
});
