# Build the Dawn Chorus desktop app into a one-click executable.
#   ./build.ps1                    uses `python` on PATH
#   ./build.ps1 -Py C:\path\python.exe
param([string]$Py = "python")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Stage the three BirdNET models into models/ (from a birdnet-analyzer install).
$ck = & $Py -c "import os,importlib.util as u; s=u.find_spec('birdnet_analyzer'); print(os.path.join(os.path.dirname(s.origin),'checkpoints','V2.4'))"
New-Item -ItemType Directory -Force -Path models | Out-Null
Copy-Item "$ck\BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite" models\model.tflite -Force   # FP32 matches the CLI
Copy-Item "$ck\BirdNET_GLOBAL_6K_V2.4_Labels.txt" models\labels.txt -Force
Copy-Item "$ck\BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite" models\mdata.tflite -Force
Write-Host "staged models/ from $ck"

# 2. PyInstaller (onedir). --collect-all pulls each package's code + data + hidden imports.
& $Py -m PyInstaller --noconfirm --console --name dawn-chorus `
  --distpath dist --workpath build --specpath . `
  --paths ..\server --hidden-import payload `
  --add-data "models;models" `
  --collect-all librosa --collect-all resampy --collect-all soxr `
  --collect-all soundfile --collect-all audioread --collect-all ai_edge_litert `
  --collect-all dawnchorus --collect-all lazy_loader --collect-all pooch `
  --collect-submodules numba --collect-submodules llvmlite --collect-submodules scipy `
  --exclude-module tensorflow --exclude-module keras --exclude-module tensorboard `
  --exclude-module torch --exclude-module grpc --exclude-module h5py `
  --exclude-module sklearn --exclude-module ml_dtypes `
  app.py

Write-Host "`nBuilt dist/dawn-chorus/dawn-chorus.exe"
