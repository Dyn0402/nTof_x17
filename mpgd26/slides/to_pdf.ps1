<#
.SYNOPSIS
    Regenerate mpgd26_talk.pdf from mpgd26_talk.pptx via PowerPoint COM automation.

.DESCRIPTION
    LibreOffice/soffice is not installed on this machine, so this drives the
    installed PowerPoint (Office16) directly through COM instead of shelling
    out to a converter. Point it at any .pptx; default is mpgd26_talk.pptx in
    this directory, written next to itself as mpgd26_talk.pdf.

.PARAMETER InputPptx
    Path to the source .pptx. Defaults to mpgd26_talk.pptx next to this script.

.PARAMETER OutputPdf
    Path to the destination .pdf. Defaults to the input path with a .pdf extension.

.EXAMPLE
    powershell -File to_pdf.ps1
    powershell -File to_pdf.ps1 -InputPptx mpgd26_talk_2026-08-29_pre-x17rate-line-fix.pptx
#>
param(
    [string]$InputPptx = (Join-Path $PSScriptRoot "mpgd26_talk.pptx"),
    [string]$OutputPdf
)

$ErrorActionPreference = "Stop"

$InputPptx = (Resolve-Path $InputPptx).Path
if (-not $OutputPdf) {
    $OutputPdf = [System.IO.Path]::ChangeExtension($InputPptx, "pdf")
}
if (-not [System.IO.Path]::IsPathRooted($OutputPdf)) {
    $OutputPdf = Join-Path (Get-Location) $OutputPdf
}

# ppSaveAsPDF = 32 (PpSaveAsFileType enum)
$ppSaveAsPDF = 32

Write-Host "Opening PowerPoint COM app..."
$app = New-Object -ComObject PowerPoint.Application

try {
    Write-Host "Loading $InputPptx"
    # WithWindow:=False still requires PowerPoint's UI process on most builds,
    # so don't rely on it to stay hidden — a window may briefly flash.
    $pres = $app.Presentations.Open($InputPptx, $true, $true, $false)
    try {
        Write-Host "Exporting to $OutputPdf"
        $pres.SaveAs($OutputPdf, $ppSaveAsPDF)
    }
    finally {
        $pres.Close()
    }
}
finally {
    $app.Quit()
}

Write-Host "wrote $OutputPdf"
