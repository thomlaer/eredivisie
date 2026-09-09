import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";

const artifactPath = "file:///C:/Users/thoml/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { SpreadsheetFile, Workbook } = await import(artifactPath);

const root = new URL("../", import.meta.url);
const dashboardPath = new URL("dashboard/public/data/dashboard.json", root);
const outputDir = new URL("outputs/019e1b53-e36c-7c60-b50f-e3af5765acce/", root);
const previewDir = new URL("previews/", outputDir);
const data = JSON.parse(await fs.readFile(dashboardPath, "utf8"));

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const colors = {
  ink: "#0B1F24",
  green: "#0F766E",
  greenSoft: "#DDF4EF",
  mint: "#EEF8F5",
  line: "#D8E2E0",
  white: "#FFFFFF",
  muted: "#5D6B6F",
  redSoft: "#FDE8E7",
  amberSoft: "#FFF2CC",
};

function value(value, fallback = "") {
  return value === null || value === undefined || Number.isNaN(value) ? fallback : value;
}

function pct(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function outcomeLabel(value) {
  if (value === "home_win") return "Thuis";
  if (value === "away_win") return "Uit";
  return "Gelijk";
}

function baseSheet(name, title, lastColumn) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: colors.ink,
    font: { bold: true, color: colors.white, size: 16 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 32;
  return sheet;
}

function styleHeader(range) {
  range.format = {
    fill: colors.green,
    font: { bold: true, color: colors.white },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: colors.green },
  };
  range.format.rowHeight = 26;
}

function styleBody(range) {
  range.format = {
    font: { color: colors.ink, size: 10 },
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: colors.line } },
  };
}

const overview = baseSheet("Overzicht", "Eredivisie voorspeller 2026/27", "H");
overview.getRange("A3:B8").values = [
  ["Modelcontrole", "Waarde"],
  ["Testaccuracy 2025/26", pct(data.metadata.model_accuracy)],
  ["XGBoost zonder odds", pct(data.benchmarks.xgboost_without_odds.accuracy)],
  ["Accuracy 2026/27 tot nu", pct(data.metadata.current_season_accuracy)],
  ["Historische wedstrijden", Number(data.metadata.historical_matches)],
  ["Laatste uitslag", data.metadata.latest_result],
];
styleHeader(overview.getRange("A3:B3"));
styleBody(overview.getRange("A4:B8"));
overview.getRange("B4:B6").format.numberFormat = "0.0%";
overview.getRange("A10:C20").values = [
  ["#", "Team", "Kampioenskans"],
  ...data.champions.slice(0, 10).map((row) => [Number(row.champion_rank), row.team, pct(row.champion_prob)]),
];
styleHeader(overview.getRange("A10:C10"));
styleBody(overview.getRange("A11:C20"));
overview.getRange("C11:C20").format.numberFormat = "0.0%";
overview.getRange("C11:C20").conditionalFormats.add("dataBar", { color: colors.green, gradient: true });
overview.getRange("A22:H23").merge();
overview.getRange("A22").values = [[
  `Chronologische validatie: training t/m 2022/23, validatie 2023/24-2024/25, test 2025/26. Gegenereerd ${data.metadata.generated_at_utc}.`,
]];
overview.getRange("A22:H23").format = {
  fill: colors.mint,
  font: { color: colors.muted, italic: true },
  wrapText: true,
  verticalAlignment: "center",
};
overview.getRange("A:A").format.columnWidth = 26;
overview.getRange("B:B").format.columnWidth = 20;
overview.getRange("C:C").format.columnWidth = 16;
const championChart = overview.charts.add("bar", overview.getRange("B10:C20"));
championChart.title = "Kampioenskans top 10";
championChart.hasLegend = false;
championChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
championChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
championChart.setPosition("E3", "L20");

const predictions = baseSheet("Voorspellingen", "Resterende wedstrijden", "K");
const predictionHeaders = [
  "Ronde", "Datum", "Aftrap UTC", "Thuis", "Uit", "Keuze", "Geselecteerd", "Thuis %", "Gelijk %", "Uit %", "Bron",
];
const predictionRows = data.upcoming.map((row) => [
  Number(row.round), row.date, value(row.kickoff_utc), row.home_team, row.away_team, outcomeLabel(row.predicted_outcome), row.predicted_winner,
  pct(row.prob_home_win), pct(row.prob_draw), pct(row.prob_away_win), row.probability_source,
]);
predictions.getRange("A3:K3").values = [predictionHeaders];
predictions.getRangeByIndexes(3, 0, predictionRows.length, predictionHeaders.length).values = predictionRows;
styleHeader(predictions.getRange("A3:K3"));
styleBody(predictions.getRangeByIndexes(3, 0, predictionRows.length, predictionHeaders.length));
predictions.getRange(`H4:J${predictionRows.length + 3}`).format.numberFormat = "0.0%";
predictions.getRange(`A3:K${predictionRows.length + 3}`).conditionalFormats.addCustom("=MOD(ROW(),2)=0", { fill: colors.mint });
predictions.tables.add(`A3:K${predictionRows.length + 3}`, true, "PredictionsTable");
predictions.freezePanes.freezeRows(3);
for (const [column, width] of Object.entries({ A: 9, B: 12, C: 24, D: 20, E: 20, F: 11, G: 20, H: 11, I: 11, J: 11, K: 16 })) {
  predictions.getRange(`${column}:${column}`).format.columnWidth = width;
}

const played = baseSheet("Gespeeld", "Eerlijke voorspellingen 2026/27", "L");
played.getRange("A3:L3").values = [["Datum", "Thuis", "Uit", "Keuze", "Geselecteerd", "Werkelijk", "Uitslag", "Goed", "Thuis %", "Gelijk %", "Uit %", "Bron"]];
const playedRows = data.played.map((row) => [
  row.date, row.home_team, row.away_team, outcomeLabel(row.predicted_outcome), row.predicted_winner, outcomeLabel(row.actual_outcome), row.actual_score,
  Number(Boolean(row.winner_correct)), pct(row.prob_home_win), pct(row.prob_draw), pct(row.prob_away_win), row.probability_source,
]);
played.getRangeByIndexes(3, 0, playedRows.length, 12).values = playedRows;
styleHeader(played.getRange("A3:L3"));
styleBody(played.getRangeByIndexes(3, 0, playedRows.length, 12));
played.getRange(`I4:K${playedRows.length + 3}`).format.numberFormat = "0.0%";
played.getRange(`H4:H${playedRows.length + 3}`).format.numberFormat = `[=1]"Ja";[=0]"Nee"`;
played.getRange("N3:O4").values = [["Controle", "Uitkomst"], ["Winnaar goed", null]];
played.getRange("O4").formulas = [[`=AVERAGE(H4:H${playedRows.length + 3})`]];
played.getRange("O4").format.numberFormat = "0.0%";
styleHeader(played.getRange("N3:O3"));
styleBody(played.getRange("N4:O4"));
played.getRange(`H4:H${playedRows.length + 3}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: colors.greenSoft, font: { color: colors.green, bold: true } } });
played.getRange(`H4:H${playedRows.length + 3}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 0, format: { fill: colors.redSoft } });
played.tables.add(`A3:L${playedRows.length + 3}`, true, "PlayedTable");
played.freezePanes.freezeRows(3);
for (const [column, width] of Object.entries({ A: 12, B: 20, C: 20, D: 11, E: 18, F: 11, G: 10, H: 10, I: 11, J: 11, K: 11, L: 16, N: 18, O: 14 })) {
  played.getRange(`${column}:${column}`).format.columnWidth = width;
}

const projectedByTeam = new Map(data.projected_table.map((row) => [row.team, row]));
const standings = baseSheet("Stand", "Actuele en verwachte eindstand", "N");
standings.getRange("A3:N3").values = [["Nu", "Team", "G", "W", "Gelijk", "V", "DV", "DT", "DS", "Pt", "Verwacht #", "Verwacht pt", "Top 3", "Degradatie"]];
const standingRows = data.current_table.map((row) => {
  const projection = projectedByTeam.get(row.team) || {};
  return [Number(row.rank), row.team, Number(row.played), Number(row.wins), Number(row.draws), Number(row.losses), Number(row.gf), Number(row.ga), Number(row.gd), Number(row.points), Number(projection.projected_rank), Number(projection.expected_points), pct(projection.top3_prob), pct(projection.relegation_prob)];
});
standings.getRangeByIndexes(3, 0, standingRows.length, 14).values = standingRows;
styleHeader(standings.getRange("A3:N3"));
styleBody(standings.getRangeByIndexes(3, 0, standingRows.length, 14));
standings.getRange(`L4:L${standingRows.length + 3}`).format.numberFormat = "0.0";
standings.getRange(`M4:N${standingRows.length + 3}`).format.numberFormat = "0.0%";
standings.getRange(`M4:M${standingRows.length + 3}`).conditionalFormats.add("colorScale", { colors: [colors.white, colors.greenSoft, colors.green], thresholds: ["min", "50%", "max"] });
standings.getRange(`N4:N${standingRows.length + 3}`).conditionalFormats.add("colorScale", { colors: [colors.white, colors.amberSoft, "#D94A45"], thresholds: ["min", "50%", "max"] });
standings.tables.add(`A3:N${standingRows.length + 3}`, true, "StandingsTable");
standings.freezePanes.freezeRows(3);
standings.getRange("A:A").format.columnWidth = 8;
standings.getRange("B:B").format.columnWidth = 22;
standings.getRange("C:N").format.columnWidth = 12;

const model = baseSheet("Model", "Modelmetingen en belangrijkste features", "F");
model.getRange("A3:D7").values = [
  ["Benchmark", "Wedstrijden", "Accuracy", "Log loss"],
  ["Gekozen combinatie", Number(data.benchmarks.selected.rows), pct(data.benchmarks.selected.accuracy), Number(data.benchmarks.selected.log_loss)],
  ["XGBoost zonder odds", Number(data.benchmarks.xgboost_without_odds.rows), pct(data.benchmarks.xgboost_without_odds.accuracy), Number(data.benchmarks.xgboost_without_odds.log_loss)],
  ["Closing odds alleen", Number(data.benchmarks.market_only.rows), pct(data.benchmarks.market_only.accuracy), Number(data.benchmarks.market_only.log_loss)],
  ["2026/27 tot nu", Number(data.benchmarks.current_season.rows), pct(data.benchmarks.current_season.accuracy), Number(data.benchmarks.current_season.log_loss)],
];
styleHeader(model.getRange("A3:D3"));
styleBody(model.getRange("A4:D7"));
model.getRange("C4:C7").format.numberFormat = "0.0%";
model.getRange("D4:D7").format.numberFormat = "0.000";
model.getRange("A10:B25").values = [
  ["Feature", "Belang"],
  ...data.feature_importance.slice(0, 15).map((row) => [row.feature, pct(row.importance)]),
];
styleHeader(model.getRange("A10:B10"));
styleBody(model.getRange("A11:B25"));
model.getRange("B11:B25").format.numberFormat = "0.00%";
model.getRange("B11:B25").conditionalFormats.add("dataBar", { color: colors.green, gradient: true });
model.getRange("A:A").format.columnWidth = 34;
model.getRange("B:D").format.columnWidth = 16;

for (const sheetName of ["Overzicht", "Voorspellingen", "Gespeeld", "Stand", "Model"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(new URL(`${sheetName}.png`, previewDir), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({ kind: "sheet,table,formula", maxChars: 8000, tableMaxRows: 5, tableMaxCols: 8 });
await fs.writeFile(new URL("inspection.json", outputDir), JSON.stringify(inspection, null, 2));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(fileURLToPath(new URL("Eredivisie_voorspellingen.xlsx", outputDir)));
