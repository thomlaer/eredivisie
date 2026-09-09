# Eredivisie Prediction

Lokale, tijdveilige voorspelpipeline voor de Eredivisie. De opzet volgt de bruikbare delen van het WK-project, maar is aangepast aan clubvoetbal.

## Databronnen

- Football-Data.co.uk: historische uitslagen, pre-match/closing odds en vanaf recente seizoenen xG, schoten, corners en kaarten.
- Football-Data.co.uk `fixtures.csv`: wekelijkse pre-match 1X2-, total-goals- en Asian-handicapodds. De feed bevat alleen competities die op dat moment zijn gepubliceerd.
- FixtureDownload: volledig programma van het lopende seizoen.
- ESPN `ned.1`: optionele live controle voor uitslagen en fixtures. De pipeline blijft werken wanneer ESPN tijdelijk blokkeert.
- Transfermarkt player-scores extract: historische basisspelers, marktwaarde, posities en recente spelerproductie. Alleen informatie van voor de te voorspellen wedstrijd wordt gebruikt.
- Transfermarkt: actuele blessures, schorsingen, selecties en seizoensminuten per club. De waarschijnlijke basiself wordt uit minuten en marktwaarde opgebouwd en per toekomstige wedstrijddatum gecorrigeerd.
- `data/manual/upcoming_odds.csv`: optionele handmatige import voor toekomstige 1X2-odds.
- `data/manual/player_availability.csv`: handmatige correcties met `availability` tussen 0 en 1 en optioneel `available_from`.

Wedstrijdstatistieken worden nooit op dezelfde wedstrijd als feature gebruikt. Ze worden pas na afloop aan de rolling teamhistorie toegevoegd.

## Lokaal draaien

```powershell
python rebuild.py
cd dashboard
npm install
npm run dev
```

Open daarna `http://localhost:3000`.

Controleer de Python-pipeline met:

```powershell
python -m unittest -v
```

De lokale Excel-export kan na een rebuild opnieuw worden gemaakt met:

```powershell
node scripts/build_workbook.mjs
```

Voor de normale wekelijkse update van bronnen, model, dashboarddata en Excel:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/weekly_update.ps1
```

De GitHub Action `.github/workflows/rebuild-predictions.yml` draait daarnaast iedere vrijdag
om 08:00 UTC en kan ook handmatig worden gestart. Op GitHub wordt `--force-live` gebruikt:
actuele selecties en spelerprestaties worden daardoor altijd vernieuwd, terwijl de grote
historische downloads alleen worden opgehaald wanneer dat nodig is.

## Belangrijkste uitvoer

- `outputs/latest/upcoming_predictions.csv`
- `outputs/latest/played_predictions.csv`
- `outputs/latest/current_table.csv`
- `outputs/latest/projected_table.csv`
- `outputs/latest/model_metrics.json`
- `data/raw/transfermarkt_eredivisie_injuries.csv`
- `data/raw/current_availability_report.json`
- `dashboard/public/data/dashboard.json`
- `outputs/019e1b53-e36c-7c60-b50f-e3af5765acce/Eredivisie_voorspellingen.xlsx`

De huidige fixturefeed bevat alle 306 competitiewedstrijden. Een rebuild probeert automatisch de wekelijkse Football-Data-odds te koppelen. Handmatige odds hebben voorrang. Als de wekelijkse feed de Eredivisie nog niet bevat, blijft het model zonder odds werken; ontbrekende odds worden nooit achteraf ingevuld.

Een normale rebuild ververst blessures en schorsingen iedere keer. De volledige selectie- en minutenlaag wordt maximaal twintig uur gecachet, zodat een wekelijkse run actueel blijft zonder 18 clubpagina's onnodig bij elke lokale test opnieuw op te halen. De huidige beschikbaarheid corrigeert alleen toekomstige voorspellingen: er is geen betrouwbare historische week-voor-week blessurearchieflaag om dit effect zonder leakage te backtesten.

Voor het lopende seizoen hebben resultaten uit Football-Data voorrang omdat die ook odds en
wedstrijdstatistieken bevatten. Als Football-Data achterloopt of tijdelijk niet bereikbaar is,
worden ontbrekende voltooide wedstrijden aangevuld uit de officiële fixturefeed. Een bestaande
rijke wedstrijdregel wordt daarbij nooit door een armere fallbackregel overschreven.

## Modelprofiel

Het defaultmodel gebruikt een compact, chronologisch getoetst profiel vanaf seizoen 2012/13:

- interne Elo en verwachte thuiswinst;
- laatste 5/10, thuis/uit en seizoensvorm voor punten, winst/gelijk/verlies, goals, schoten en corners;
- vorige-seizoenssterkte, promotiesignaal, rustdagen en head-to-head;
- marktwaarde, leeftijd, positieopbouw, stabiliteit en recente productie van de laatst bekende basiself/selectieproxy;
- pre-match 1X2-odds als primaire kansbron wanneer beschikbaar;
- actuele afwezigen als correctie op de beschikbare opstellingswaarde en spelersvorm.

Formaties van de actuele wedstrijd, actuele matchstats, eindstand na de wedstrijd en actuele basiself worden niet gebruikt voor historische pre-matchtraining. De experimentbestanden staan onder `outputs/experiments`.

De openbare voorspelling is uitsluitend `thuis`, `gelijk` of `uit`. De vaste outcome-laag gebruikt bij beschikbare odds de no-vig marktverdeling met `draw x 1,30`; zonder odds gebruikt hij XGBoost met `draw x 0,75`. Deze instellingen zijn over rollende seizoenen t/m 2024/25 gekozen, vóór de test op 2025/26. De WK-regel die soms de tweede keuze forceert is niet gebruikt, omdat die per seizoen te instabiel bleek. Twee interne Poisson-regressors worden alleen gebruikt om doelsaldo in de eindstandsimulatie te benaderen en leveren geen exact-scoreadvies.

Op de vaste 2025/26-testset haalt deze outcome-laag 53,92% accuracy wanneer de historische odds beschikbaar zijn. Over zeven walk-forwardseizoenen haalt de gekalibreerde markt ongeveer 56,4%; zonder odds haalt XGBoost ongeveer 54,6%. Dit zijn out-of-time scores; willekeurige train/test-splits geven doorgaans een misleidend hoger percentage.

De hoofdcontrole is chronologisch: trainen t/m 2022/23, modelkeuzes op 2023/24 en 2024/25, en pas daarna een eenmalige test op 2025/26. Het lopende seizoen 2026/27 is alleen een live audit. XGBoost heeft geen standaardisatie naar z-scores nodig; ontbrekende numerieke waarden worden met de trainingsmediaan ingevuld. Bookmakerodds worden wel genormaliseerd naar no-vig kansen die samen 1 zijn. Teamstatistieken worden vooral als gemiddelden, aandelen en thuis-uitverschillen aangeboden.

Niet alle WK-variabelen zijn letterlijk gekopieerd. FIFA-ranking, neutraal terrein, landreizen, toernooistand, nationale selectiecaps en knock-outronde zijn niet van toepassing op een vaste clubcompetitie. Hun clubanalogen zijn wel aanwezig: Elo, thuisvoordeel, rustdagen, stand/vorm binnen het seizoen, vorige-seizoenssterkte, marktwaarde en laatst bekende opstelling. Managerfeatures, ClubElo, extra scoremarkten en trainingsgewichten zijn lokaal getest maar niet opgenomen omdat ze niet stabiel beter presteerden.

De belangrijkste nog ontbrekende kandidaatvariabelen zijn historische blessures/schorsingen per speeldag, verwachte basisplaatsen vlak voor de aftrap, speler-xG/xA en schotkwaliteit, keeperkwaliteit en belasting uit Europese/bekerwedstrijden. Die zijn pas verantwoord toe te voegen met een gedateerd historisch archief; alleen actuele waarden in oude trainingsrijen zetten zou data leakage veroorzaken.

## Later naar GitHub/Vercel

De map `dashboard` is al een zelfstandige Next.js-app. Een latere GitHub Action hoeft alleen `python rebuild.py` te draaien, de compacte bestanden onder `dashboard/public` te committen en daarna de app naar Vercel te deployen. API-sleutels zijn voor de huidige bronnen niet nodig.
