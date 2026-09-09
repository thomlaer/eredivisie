param(
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $arguments = @("rebuild.py", "--simulations", "10000")
    if ($ForceDownload) {
        $arguments += "--force-download"
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "De Python rebuild is mislukt met exitcode $LASTEXITCODE."
    }
    & node "scripts/build_workbook.mjs"
    if ($LASTEXITCODE -ne 0) {
        throw "De Excel-export is mislukt met exitcode $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
