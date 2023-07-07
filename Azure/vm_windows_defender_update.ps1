# keep VM up-to-date with anti-virus

wget https://go.microsoft.com/fwlink/?LinkId=197094 -OutFile C:\nis.exe;

wget https://go.microsoft.com/fwlink/?LinkId=121721"&"arch=x64 -OutFile C:\mpam-fe.exe;

wget https://go.microsoft.com/fwlink/?LinkID=87341 -OutFile C:\mpfam-fe64.exe;

cd C:\;

.\nis.exe;

.\mpam-fe.exe;

.\mpfam-fe64.exe;

Update-MpSignature;

cd "C:\Program Files\Windows Defender";

.\MpCmdRun.exe -removedefinitions -dynamicsignatures;

.\MpCmdRun.exe -SignatureUpdate;

Start-MpScan -ScanType QuickScan;

Exit;

Get-MpComputerStatus | select *updated, *version