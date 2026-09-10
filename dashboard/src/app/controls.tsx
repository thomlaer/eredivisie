"use client";

import { FormEvent, useMemo, useState } from "react";
import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";

type ExcelPick = {
  homeTeam: string;
  awayTeam: string;
  outcome: string;
};

type UpdateResult = {
  ok?: boolean;
  message?: string;
  actionsUrl?: string;
};

const TEAM_ALIASES: Record<string, string[]> = {
  "ado den haag": ["ado den haag", "ado"],
  "fc groningen": ["fc groningen", "groningen"],
  "fc twente": ["fc twente", "twente"],
  "fc utrecht": ["fc utrecht", "utrecht"],
  "fortuna sittard": ["fortuna sittard", "fortuna"],
  "go ahead eagles": ["go ahead eagles", "go ahead"],
  "heerenveen": ["sc heerenveen", "heerenveen"],
  "pec zwolle": ["pec zwolle", "pec"],
  "sc cambuur": ["sc cambuur", "cambuur"],
  "sparta rotterdam": ["sparta rotterdam", "sparta"],
  "willem ii": ["willem ii", "willem 2"],
};

function normalize(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function aliases(team: string): string[] {
  const key = normalize(team);
  return (TEAM_ALIASES[key] || [key]).map(normalize);
}

function includesAlias(rowText: string, team: string): boolean {
  const padded = ` ${rowText} `;
  return aliases(team).some((alias) => padded.includes(` ${alias} `));
}

function scoreForOutcome(outcome: string): number {
  if (outcome === "home_win") return 1;
  if (outcome === "away_win") return 2;
  return 3;
}

function relationshipPath(target: string): string {
  const clean = target.replace(/^\//, "");
  if (clean.startsWith("xl/")) return clean;
  return `xl/${clean.replace(/^\.\//, "")}`;
}

function cellText(cell: Element, sharedStrings: string[]): string {
  const type = cell.getAttribute("t");
  if (type === "inlineStr") {
    return [...cell.getElementsByTagName("t")].map((node) => node.textContent || "").join("");
  }
  const raw = cell.getElementsByTagName("v")[0]?.textContent || "";
  if (type === "s") return sharedStrings[Number(raw)] || "";
  return raw;
}

function columnNumber(reference: string): number {
  const letters = reference.match(/^[A-Z]+/i)?.[0].toUpperCase() || "A";
  return [...letters].reduce((total, letter) => total * 26 + letter.charCodeAt(0) - 64, 0);
}

function setNumericCell(document: XMLDocument, row: Element, reference: string, value: number): void {
  const cells = [...row.getElementsByTagName("c")];
  let cell = cells.find((candidate) => candidate.getAttribute("r") === reference);
  if (!cell) {
    cell = document.createElementNS(document.documentElement.namespaceURI, "c");
    cell.setAttribute("r", reference);
    const previousCell = [...cells].reverse().find((candidate) => columnNumber(candidate.getAttribute("r") || "A1") < columnNumber(reference));
    const style = previousCell?.getAttribute("s");
    if (style) cell.setAttribute("s", style);
    const nextCell = cells.find((candidate) => columnNumber(candidate.getAttribute("r") || "A1") > columnNumber(reference));
    row.insertBefore(cell, nextCell || null);
  }
  while (cell.firstChild) cell.removeChild(cell.firstChild);
  cell.setAttribute("t", "n");
  const valueNode = document.createElementNS(document.documentElement.namespaceURI, "v");
  valueNode.textContent = String(value);
  cell.appendChild(valueNode);
}

function serializeXml(document: XMLDocument): string {
  const xml = new XMLSerializer().serializeToString(document);
  return xml.startsWith("<?xml") ? xml : `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>${xml}`;
}

async function fillWorkbook(file: File, picks: ExcelPick[]): Promise<{ blob: Blob; matched: number }> {
  if (file.size > 12 * 1024 * 1024) throw new Error("Het Excel-bestand is groter dan 12 MB.");
  const files = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const parser = new DOMParser();
  const workbookFile = files["xl/workbook.xml"];
  const relationsFile = files["xl/_rels/workbook.xml.rels"];
  if (!workbookFile || !relationsFile) throw new Error("Dit is geen geldig Excel-bestand.");

  const sharedStrings = files["xl/sharedStrings.xml"]
    ? [...parser.parseFromString(strFromU8(files["xl/sharedStrings.xml"]), "application/xml").getElementsByTagName("si")]
        .map((item) => [...item.getElementsByTagName("t")].map((node) => node.textContent || "").join(""))
    : [];
  const workbook = parser.parseFromString(strFromU8(workbookFile), "application/xml");
  const relations = parser.parseFromString(strFromU8(relationsFile), "application/xml");
  const targets = new Map(
    [...relations.getElementsByTagName("Relationship")].map((relation) => [
      relation.getAttribute("Id") || "",
      relationshipPath(relation.getAttribute("Target") || ""),
    ]),
  );

  let matched = 0;
  let foundScoreColumn = false;
  for (const sheet of [...workbook.getElementsByTagName("sheet")]) {
    const relationId = sheet.getAttribute("r:id") || sheet.getAttributeNS("http://schemas.openxmlformats.org/officeDocument/2006/relationships", "id") || "";
    const sheetPath = targets.get(relationId);
    if (!sheetPath || !files[sheetPath]) continue;
    const document = parser.parseFromString(strFromU8(files[sheetPath]), "application/xml");
    const rows = [...document.getElementsByTagName("row")];
    let headerRowIndex = -1;
    let scoreColumn = "";
    rows.some((row, index) => {
      const header = [...row.getElementsByTagName("c")].find((cell) => normalize(cellText(cell, sharedStrings)) === "totoscore");
      if (!header) return false;
      headerRowIndex = index;
      scoreColumn = header.getAttribute("r")?.match(/^[A-Z]+/i)?.[0].toUpperCase() || "";
      return true;
    });
    if (headerRowIndex < 0 || !scoreColumn) continue;
    foundScoreColumn = true;

    for (const row of rows.slice(headerRowIndex + 1)) {
      const rowText = normalize([...row.getElementsByTagName("c")].map((cell) => cellText(cell, sharedStrings)).join(" "));
      const pick = picks.find((candidate) => includesAlias(rowText, candidate.homeTeam) && includesAlias(rowText, candidate.awayTeam));
      if (!pick) continue;
      const rowNumber = row.getAttribute("r");
      if (!rowNumber) continue;
      setNumericCell(document, row, `${scoreColumn}${rowNumber}`, scoreForOutcome(pick.outcome));
      matched += 1;
    }
    files[sheetPath] = strToU8(serializeXml(document));
  }

  if (!foundScoreColumn) throw new Error("Kolom TOTOSCORE is niet gevonden.");
  if (!matched) throw new Error("Geen wedstrijden uit de huidige speelronde gevonden.");
  return {
    blob: new Blob([zipSync(files, { level: 6 })], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }),
    matched,
  };
}

export function ExcelFiller({ picks, round }: { picks: ExcelPick[]; round: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const expected = useMemo(() => picks.length, [picks]);

  async function processFile() {
    if (!file) return;
    setBusy(true);
    setStatus("");
    try {
      const result = await fillWorkbook(file, picks);
      const url = URL.createObjectURL(result.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = file.name;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`${result.matched} van ${expected} wedstrijden ingevuld. Het bestand is gedownload.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Het Excel-bestand kon niet worden ingevuld.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section id="excel" className="section">
      <div className="section-heading">
        <div>
          <h2>Excel invullen</h2>
          <p>Kies het ontvangen bestand. De site vult speelronde {round} in met 1, 2 of 3 en downloadt het onder dezelfde naam.</p>
        </div>
      </div>
      <div className="tool-panel excel-tool">
        <label className="file-picker">
          <span>Excel-bestand</span>
          <input
            accept=".xlsx"
            type="file"
            onChange={(event) => {
              setFile(event.target.files?.[0] || null);
              setStatus("");
            }}
          />
          <strong>{file?.name || "Kies een .xlsx-bestand"}</strong>
        </label>
        <button className="primary-button" disabled={!file || busy} onClick={processFile} type="button">
          {busy ? "Bezig…" : "Vul Excel in"}
        </button>
        {status && <p className="tool-status" role="status">{status}</p>}
      </div>
    </section>
  );
}

export function UpdateControl() {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<UpdateResult | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setResult(null);
    try {
      const response = await fetch("/api/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const body = (await response.json()) as UpdateResult;
      setResult(body);
    } catch {
      setResult({ ok: false, message: "Bijwerken kon niet worden gestart." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section id="bijwerken" className="section update-section">
      <div className="section-heading"><div><h2>Handmatig bijwerken</h2><p>Gebruik dit alleen wanneer je niet tot woensdag wilt wachten.</p></div></div>
      <form className="tool-panel update-tool" onSubmit={submit}>
        <label><span>Code</span><input autoComplete="off" value={code} onChange={(event) => setCode(event.target.value)} type="password" /></label>
        <button className="primary-button" disabled={!code || busy} type="submit">{busy ? "Wordt gestart…" : "Start bijwerken"}</button>
        {result?.message && <p className={`tool-status ${result.ok ? "success" : "error"}`} role="status">{result.message}</p>}
        {result?.actionsUrl && <a className="actions-link" href={result.actionsUrl} rel="noreferrer" target="_blank">Bekijk voortgang op GitHub</a>}
      </form>
    </section>
  );
}
