import fs from "node:fs";
import path from "node:path";
import { ExcelFiller, UpdateControl } from "./controls";
import { groupByRound, selectFocusRound } from "../lib/round-selection";

type Numberish = number | string | null | undefined;

type Prediction = {
  match_key: string;
  match_number: Numberish;
  round: Numberish;
  date: string;
  kickoff_utc?: string | null;
  venue?: string;
  home_team: string;
  away_team: string;
  predicted_winner: string;
  predicted_outcome: string;
  prob_home_win: Numberish;
  prob_draw: Numberish;
  prob_away_win: Numberish;
  probability_source?: string;
  odds_source?: string;
  odds_home?: Numberish;
  odds_draw?: Numberish;
  odds_away?: Numberish;
  actual_score?: string;
  actual_outcome?: string;
  winner_correct?: boolean;
};

type Injury = {
  team: string;
  player: string;
  injury?: string;
  absence_type?: string;
  since?: string;
  expected_return?: string;
  availability?: Numberish;
  selected_proxy?: Numberish;
  matched_roster?: Numberish;
  market_value_m?: Numberish;
  estimated_value_impact_m?: Numberish;
  source?: string;
};

type Standing = {
  rank?: Numberish;
  projected_rank?: Numberish;
  champion_rank?: Numberish;
  team: string;
  played?: Numberish;
  wins?: Numberish;
  draws?: Numberish;
  losses?: Numberish;
  gf?: Numberish;
  ga?: Numberish;
  gd?: Numberish;
  points?: Numberish;
  expected_points?: Numberish;
  expected_gd?: Numberish;
  champion_prob?: Numberish;
  top3_prob?: Numberish;
  relegation_prob?: Numberish;
};

type Metrics = {
  rows?: Numberish;
  accuracy?: Numberish;
  log_loss?: Numberish;
  exact_score_accuracy?: Numberish;
};

type Dashboard = {
  metadata: {
    generated_at_utc: string;
    season: string;
    historical_matches: Numberish;
    latest_result: string;
    model_accuracy: Numberish;
    model_log_loss: Numberish;
    current_season_accuracy: Numberish;
    odds_weight: Numberish;
    draw_multiplier: Numberish;
    espn_status: string;
    injury_status?: string;
    suspension_status?: string;
    current_squad_status?: string;
    absences_total?: Numberish;
    injuries_matched?: Numberish;
    injuries_unmatched?: Numberish;
    likely_starters_unavailable?: Numberish;
  };
  upcoming: Prediction[];
  played: Prediction[];
  current_table: Standing[];
  projected_table: Standing[];
  champions: Standing[];
  injuries: Injury[];
  feature_importance: { feature: string; importance: Numberish }[];
  benchmarks: {
    selected: Metrics;
    xgboost_without_odds: Metrics;
    market_only: Metrics;
    current_season: Metrics;
  };
  downloads: Record<string, string>;
};

function loadDashboard(): Dashboard {
  const file = path.join(process.cwd(), "public", "data", "dashboard.json");
  return JSON.parse(fs.readFileSync(file, "utf-8")) as Dashboard;
}

function number(value: Numberish, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function percent(value: Numberish, digits = 1): string {
  return `${(number(value) * 100).toFixed(digits)}%`;
}

function decimal(value: Numberish, digits = 2): string {
  return number(value).toFixed(digits);
}

function kickoff(value?: string | null): string {
  if (!value) return "Tijd volgt";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("nl-NL", {
    timeZone: "Europe/Amsterdam",
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function outcomeClass(row: Prediction): string {
  if (row.winner_correct) return "correct";
  return "wrong";
}

function outcomeLabel(row: Prediction): string {
  if (row.winner_correct) return "Goed";
  return "Mis";
}

function pickLabel(row: Prediction): string {
  if (row.predicted_outcome === "home_win") return "1 · Thuis";
  if (row.predicted_outcome === "away_win") return "2 · Uit";
  return "3 · Gelijk";
}

function ModelLine({ row }: { row: Prediction }) {
  return (
    <div className="model-line">
      <span>Thuis {percent(row.prob_home_win)}</span>
      <span>Gelijk {percent(row.prob_draw)}</span>
      <span>Uit {percent(row.prob_away_win)}</span>
    </div>
  );
}

function MatchList({ rows }: { rows: Prediction[] }) {
  return (
    <div className="match-list">
      {rows.map((row) => (
        <article className="match" key={row.match_key}>
          <div className="match-time">{kickoff(row.kickoff_utc)}</div>
          <div className="teams">
            <strong>{row.home_team} <span>-</span> {row.away_team}</strong>
            <small>{row.venue || "Locatie volgt"}</small>
            <ModelLine row={row} />
          </div>
          <div className="pick">
            <span className={`outcome-pick ${row.predicted_outcome}`}>{pickLabel(row)}</span>
            <strong>{row.predicted_winner}</strong>
          </div>
          <div className="source">
            <span className={row.probability_source !== "xgboost" ? "badge odds" : "badge"}>
              {row.probability_source === "xgboost" ? "Model" : "Actuele kansen"}
            </span>
            <small>{percent(Math.max(number(row.prob_home_win), number(row.prob_draw), number(row.prob_away_win)))} zekerheid</small>
          </div>
        </article>
      ))}
    </div>
  );
}

export default function Home() {
  const data = loadDashboard();
  const rounds = groupByRound(data.upcoming);
  const [nextRound, focusRows] = selectFocusRound(data.upcoming, data.played, data.metadata.generated_at_utc);
  const otherRounds = rounds.filter(([round]) => round !== nextRound);
  const upcomingOdds = focusRows.filter((row) => row.probability_source !== "xgboost").length;
  const excelPicks = focusRows.map((row) => ({ homeTeam: row.home_team, awayTeam: row.away_team, outcome: row.predicted_outcome }));

  return (
    <main>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#voorspellingen">
            <strong>Eredivisie voorspeller</strong>
            <span>Seizoen {data.metadata.season}</span>
          </a>
          <nav aria-label="Dashboard">
            <a href="#voorspellingen">Voorspellingen</a>
            <a href="#afwezigen">Afwezigen</a>
            <a href="#gespeeld">Gespeeld</a>
            <a href="#stand">Stand</a>
            <a href="#kampioen">Kampioen</a>
            <a href="#bijwerken">Bijwerken</a>
          </nav>
          <a className="download" href="#excel">Excel</a>
        </div>
      </header>

      <div className="content">
        <section className="summary" aria-label="Samenvatting">
          <div className="summary-main">
            <p className="eyebrow">Bijgewerkt {data.metadata.generated_at_utc.slice(0, 16).replace("T", " ")} UTC</p>
            <h1>Speelronde {nextRound}</h1>
            <p>De {focusRows.length} wedstrijden voor het aankomende invulweekend. Bij {upcomingOdds} wedstrijden zijn actuele kansen gebruikt.</p>
          </div>
          <div className="metric"><span>Historische controle</span><strong>{percent(data.metadata.model_accuracy)}</strong><small>keuzes goed</small></div>
          <div className="metric"><span>Dit seizoen</span><strong>{percent(data.metadata.current_season_accuracy)}</strong><small>keuzes goed</small></div>
          <div className="metric"><span>Afwezigen</span><strong>{number(data.metadata.likely_starters_unavailable)}</strong><small>mogelijke basisspelers</small></div>
        </section>

        <ExcelFiller picks={excelPicks} round={nextRound} />

        <section id="voorspellingen" className="section">
          <div className="section-heading">
            <div>
              <h2>Voorspellingen</h2>
              <p>Per wedstrijd één keuze: 1 voor thuis, 2 voor uit of 3 voor gelijk. Afwezige spelers en actuele kansen zijn verwerkt.</p>
            </div>
          </div>
          <div className="round-stack">
            {focusRows.length > 0 && (
              <details className="round" open>
                <summary>
                  <span><strong>Speelronde {nextRound}</strong><small>{focusRows.length} wedstrijden</small></span>
                  <span className="round-toggle" aria-hidden="true" />
                </summary>
                <MatchList rows={focusRows} />
              </details>
            )}
            {otherRounds.length > 0 && (
              <details className="round-index">
                <summary>Alle andere speelrondes <small>{otherRounds.length} rondes</small><span className="round-toggle" aria-hidden="true" /></summary>
                <div className="secondary-rounds">
                  {otherRounds.map(([round, rows]) => (
                    <details className="round compact-round" key={`other-${round}`}>
                      <summary>
                        <span><strong>Speelronde {round}</strong><small>{rows.length} wedstrijd{rows.length === 1 ? "" : "en"}</small></span>
                        <span className="round-toggle" aria-hidden="true" />
                      </summary>
                      <MatchList rows={rows} />
                    </details>
                  ))}
                </div>
              </details>
            )}
          </div>
        </section>

        <section id="afwezigen" className="section">
          <details className="compact-section">
            <summary><span><strong>Afwezigen</strong><small>{data.metadata.likely_starters_unavailable || 0} mogelijke basisspelers, {data.metadata.absences_total || 0} spelers totaal</small></span><span className="round-toggle" aria-hidden="true" /></summary>
            <div className="table-shell">
              {number(data.metadata.injuries_unmatched) > 0 && <p className="warning inline-warning">Controle nodig: {data.metadata.injuries_unmatched} spelers konden niet worden gekoppeld.</p>}
              <table className="injury-table">
                <thead><tr><th>Club</th><th>Speler</th><th>Reden</th><th>Verwachte rol</th></tr></thead>
                <tbody>
                  {(data.injuries || []).map((row) => (
                    <tr key={`injury-${row.team}-${row.player}`}>
                      <td><strong>{row.team}</strong></td>
                      <td><strong>{row.player}</strong></td>
                      <td>{row.injury || "Niet beschikbaar"}<small>{row.expected_return ? `verwacht terug: ${row.expected_return}` : "terugkeer onbekend"}</small></td>
                      <td>{!number(row.matched_roster) ? "Niet herkend" : number(row.selected_proxy) ? "Waarschijnlijk basisspeler" : "Overige speler"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>

        <section id="gespeeld" className="section">
          <div className="section-heading">
            <div>
              <h2>Gespeeld</h2>
              <p>Voorspeld met een model dat alleen eerdere seizoenen kende; beoordeling is uitsluitend thuis, gelijk of uit.</p>
            </div>
          </div>
          <div className="table-shell">
            <table>
              <thead><tr><th>Wedstrijd</th><th>Voorspeld</th><th>Uitslag</th><th>Resultaat</th></tr></thead>
              <tbody>
                {data.played.map((row) => (
                  <tr key={`played-${row.match_key}`}>
                    <td><strong>{row.home_team} - {row.away_team}</strong><small>{row.date}{row.round ? ` · speelronde ${row.round}` : ""}</small></td>
                    <td><strong>{pickLabel(row)}</strong><small>{row.predicted_winner}</small></td>
                    <td><span className="score actual">{row.actual_score}</span></td>
                    <td><span className={`result ${outcomeClass(row)}`}>{outcomeLabel(row)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="stand" className="section standings-band">
          <div className="section-heading">
            <div><h2>Stand</h2><p>Links de echte stand, rechts de gemiddelde eindstand uit 10.000 simulaties.</p></div>
          </div>
          <div className="standings-grid">
            <div className="table-shell">
              <h3>Nu</h3>
              <table>
                <thead><tr><th>#</th><th>Team</th><th>G</th><th>W-G-V</th><th>DS</th><th>Pt</th></tr></thead>
                <tbody>{data.current_table.map((row) => <tr key={`now-${row.team}`}><td>{row.rank}</td><td><strong>{row.team}</strong></td><td>{row.played}</td><td>{row.wins}-{row.draws}-{row.losses}</td><td>{number(row.gd ?? null) > 0 ? "+" : ""}{row.gd}</td><td><strong>{row.points}</strong></td></tr>)}</tbody>
              </table>
            </div>
            <div className="table-shell">
              <h3>Verwachte eindstand</h3>
              <table>
                <thead><tr><th>#</th><th>Team</th><th>Pt</th><th>DS</th><th>Top 3</th><th>Degradatie</th></tr></thead>
                <tbody>{data.projected_table.map((row) => <tr key={`projected-${row.team}`}><td>{row.projected_rank}</td><td><strong>{row.team}</strong></td><td>{decimal(row.expected_points, 1)}</td><td>{decimal(row.expected_gd, 1)}</td><td>{percent(row.top3_prob)}</td><td>{percent(row.relegation_prob)}</td></tr>)}</tbody>
              </table>
            </div>
          </div>
        </section>

        <section id="kampioen" className="section">
          <div className="section-heading"><div><h2>Kampioenskansen</h2><p>Gebaseerd op de gespeelde uitslagen en 10.000 simulaties van alle resterende wedstrijden.</p></div></div>
          <div className="champion-list">
            {data.champions.slice(0, 10).map((row) => (
              <div className="champion-row" key={`champion-${row.team}`}>
                <span className="rank">{row.champion_rank}</span>
                <strong>{row.team}</strong>
                <div className="bar"><span style={{ width: percent(row.champion_prob) }} /></div>
                <b>{percent(row.champion_prob)}</b>
              </div>
            ))}
          </div>
        </section>

        <UpdateControl />
      </div>
    </main>
  );
}
