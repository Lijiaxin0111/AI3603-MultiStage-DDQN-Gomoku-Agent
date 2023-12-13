


@echo off

set pis_root=%1
set ai1=%2
set ai2=%3


set out_file=%4
set opening_idx=%5
set ai_root=%6




echo outfile: %out_file%
echo opening_idx: %opening_idx%
echo %ai1%  vs %ai2% 
echo %ai_root%


echo %pis_root%\piskvork.exe -p %ai_root%\%ai1% %ai_root%\%ai2% -outfile %out_file% -outfileformat 2 -opening %opening_idx%

%pis_root%\piskvork.exe -p %ai_root%\%ai1% %ai_root%\%ai2% -outfile %out_file% -outfileformat 2 -opening %opening_idx%

TASKKILL /T  /F /IM  %ai1%
TASKKILL /T  /F /IM  %ai2%


echo Done
