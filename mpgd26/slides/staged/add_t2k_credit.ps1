# Add the T2K ND280 acknowledgement to slides 9.1 and 9.2 of the MPGD26 deck.
# Edits ppt/slides/slide17.xml and slide18.xml in place inside the .pptx zip.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$pptx = 'C:\Users\Dyn04\PycharmProjects\nTof_x17\mpgd26\slides\mpgd26_talk.pptx'
$parts = @('ppt/slides/slide17.xml', 'ppt/slides/slide18.xml')

# One muted line in the header band, centred between the eyebrow and the
# "Work in progress" badge. 7.5 pt vs the eyebrow's 9.12 pt so it does not
# compete; footer grey 5D7176, with "T2K ND280" in the darker 3C5257.
$sp = @'
<p:sp><p:nvSpPr><p:cNvPr id="30" name="TextBox 29"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="4470000" y="380000"/><a:ext cx="4200000" cy="200025"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr algn="ctr"><a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="750" b="0" i="0" spc="20" dirty="0"><a:solidFill><a:srgbClr val="5D7176"/></a:solidFill><a:latin typeface="Noto Sans"/></a:rPr><a:t xml:space="preserve">Method inspired by </a:t></a:r><a:r><a:rPr lang="en-US" sz="750" b="0" i="0" spc="20" dirty="0"><a:solidFill><a:srgbClr val="3C5257"/></a:solidFill><a:latin typeface="Noto Sans SemiBold"/></a:rPr><a:t>T2K ND280</a:t></a:r><a:r><a:rPr lang="en-US" sz="750" b="0" i="0" spc="20" dirty="0"><a:solidFill><a:srgbClr val="5D7176"/></a:solidFill><a:latin typeface="Noto Sans"/></a:rPr><a:t xml:space="preserve">&#160;&#183; Atti&#233; et al., NIM A 1056 (2023) 168534</a:t></a:r></a:p></p:txBody></p:sp>
'@
$sp = $sp.Trim()

$zip = [System.IO.Compression.ZipFile]::Open($pptx, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    foreach ($name in $parts) {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq $name }
        if ($null -eq $entry) { throw "part not found: $name" }

        $reader = New-Object System.IO.StreamReader($entry.Open(), [System.Text.Encoding]::UTF8)
        $xml = $reader.ReadToEnd()
        $reader.Close()

        if ($xml -match 'T2K ND280') { Write-Output "$name : already has the credit, skipped"; continue }
        if ($xml -notmatch '</p:spTree>') { throw "no spTree in $name" }

        $new = $xml -replace '</p:spTree>', ($sp + '</p:spTree>')

        $stream = $entry.Open()
        $stream.SetLength(0)
        $writer = New-Object System.IO.StreamWriter($stream, (New-Object System.Text.UTF8Encoding($false)))
        $writer.Write($new)
        $writer.Flush()
        $writer.Close()
        Write-Output "$name : credit added ($($xml.Length) -> $($new.Length) chars)"
    }
}
finally { $zip.Dispose() }
