$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project Python environment not found at $python"
}

if ([string]::IsNullOrWhiteSpace($env:NVIDIA_API_KEY)) {
    $secureKey = Read-Host "Enter your NVIDIA NIM API key (input is hidden and not saved)" -AsSecureString
    if ($secureKey.Length -eq 0) {
        throw "An NVIDIA API key is required to enable NIM."
    }

    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:NVIDIA_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
        $secureKey.Dispose()
    }
}

& $python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    throw "SensorQA exited with code $LASTEXITCODE"
}
