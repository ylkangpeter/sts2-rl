param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\configs\runtime_stack.json"),
    [switch]$IncludeClient
)

$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Output "[stack] $Message"
}

function Read-Config {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Config not found: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Resolve-ManagedPath {
    param(
        [string]$BasePath,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

function Get-PortOwner {
    param([int]$Port)
    try {
        $match = netstat -ano -p tcp |
            Select-String "LISTENING" |
            Where-Object { $_.Line -match "[:\.]$Port\s+" } |
            Select-Object -First 1
        if ($match) {
            $parts = ($match.Line -replace "\s+", " ").Trim().Split(" ")
            return [int]$parts[-1]
        }
    } catch {
    }
    return $null
}

function Test-HttpEndpoint {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Get-MatchingProcess {
    param([string]$Pattern)
    if ([string]::IsNullOrWhiteSpace($Pattern)) {
        return $null
    }
    $needle = $Pattern.Replace("\\", "\")
    return Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Replace("\\", "\").IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
        } |
        Select-Object -First 1
}

function Wait-Endpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSeconds = 20
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpEndpoint -Url $Url) {
            Write-Info "$Name is healthy at $Url"
            return $true
        }
        Start-Sleep -Milliseconds 800
    }
    Write-Warning "$Name did not become healthy within $TimeoutSeconds seconds: $Url"
    return $false
}

function Start-ManagedProcess {
    param(
        [string]$Name,
        [object]$Node,
        [string]$PythonExe,
        [string]$DefaultRoot,
        [hashtable]$SharedEnv,
        [switch]$ForceEnabled
    )

    if (-not $ForceEnabled -and -not $Node.enabled) {
        Write-Info "$Name disabled in config, skipping"
        return
    }

    $workdir = Resolve-ManagedPath -BasePath $DefaultRoot -Value $Node.workdir
    $scriptPath = Resolve-ManagedPath -BasePath $DefaultRoot -Value $Node.script
    $processMatch = [string]$Node.process_match
    $bindHost = [string]$Node.host
    $port = if ($null -ne $Node.port) { [int]$Node.port } else { $null }
    $healthPath = [string]$Node.health_path
    $healthUrl = if ($bindHost -and $port -and $healthPath) { "http://$bindHost`:$port$healthPath" } else { $null }

    if ($port) {
        $portOwner = Get-PortOwner -Port $port
        if ($portOwner) {
            if ($healthUrl -and (Test-HttpEndpoint -Url $healthUrl)) {
                Write-Info "$Name already running on port $port (pid=$portOwner)"
                return
            }
            throw "$Name expected port $port, but it is occupied by pid=$portOwner and health check failed"
        }
    } elseif ($processMatch) {
        $existing = Get-MatchingProcess -Pattern $processMatch
        if ($existing) {
            Write-Info "$Name already running (pid=$($existing.ProcessId))"
            return
        }
    }

    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "$Name script not found: $scriptPath"
    }

    $logDir = Join-Path $workdir "logs\launcher"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stdoutLog = Join-Path $logDir "$Name.out.log"
    $stderrLog = Join-Path $logDir "$Name.err.log"

    $argList = @($scriptPath)
    if ($Node.args) {
        foreach ($item in $Node.args) {
            $argList += [string]$item
        }
    }

    $envBackup = @{}
    $effectiveEnv = @{}
    foreach ($key in $SharedEnv.Keys) {
        $effectiveEnv[$key] = [string]$SharedEnv[$key]
    }
    if ($Node.env) {
        foreach ($property in $Node.env.PSObject.Properties) {
            $effectiveEnv[$property.Name] = [string]$property.Value
        }
    }
    foreach ($key in $effectiveEnv.Keys) {
        $envBackup[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, $effectiveEnv[$key], "Process")
    }

    try {
        $process = Start-Process -FilePath $PythonExe `
            -ArgumentList $argList `
            -WorkingDirectory $workdir `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -WindowStyle Hidden `
            -PassThru
    } finally {
        foreach ($key in $envBackup.Keys) {
            [Environment]::SetEnvironmentVariable($key, $envBackup[$key], "Process")
        }
    }

    Write-Info "$Name started (pid=$($process.Id))"
    Write-Info "$Name stdout: $stdoutLog"
    Write-Info "$Name stderr: $stderrLog"

    if ($healthUrl) {
        Wait-Endpoint -Name $Name -Url $healthUrl | Out-Null
    }
}

$config = Read-Config -Path $ConfigPath
$configDir = Split-Path -Parent (Resolve-Path -LiteralPath $ConfigPath)
$st2rlRoot = Resolve-ManagedPath -BasePath $configDir -Value ([string]$config.projects.st2rl_root)
$sts2CliRoot = Resolve-ManagedPath -BasePath $configDir -Value ([string]$config.projects.sts2_cli_root)
$pythonExe = [string]$config.python_executable
$sharedEnv = @{}

if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    $pythonExe = "python"
}
if (-not [string]::IsNullOrWhiteSpace([string]$config.projects.game_dir)) {
    $sharedEnv["STS2_GAME_DIR"] = [string]$config.projects.game_dir
}

Write-Info "Using config: $ConfigPath"
Write-Info "Python executable: $pythonExe"

Start-ManagedProcess -Name "service" -Node $config.service -PythonExe $pythonExe -DefaultRoot $sts2CliRoot -SharedEnv $sharedEnv
Start-ManagedProcess -Name "dashboard" -Node $config.dashboard -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv
if ($config.watchdog) {
    Start-ManagedProcess -Name "watchdog" -Node $config.watchdog -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv
}
if ($config.session_supervisor) {
    Start-ManagedProcess -Name "session_supervisor" -Node $config.session_supervisor -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv
}

if ($IncludeClient -or $config.client.enabled) {
    Start-ManagedProcess -Name "client" -Node $config.client -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv -ForceEnabled:$IncludeClient
} else {
    Write-Info "client disabled, skipping"
}
