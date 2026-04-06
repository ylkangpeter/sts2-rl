param(
    [string]$ConfigPath = "",
    [switch]$IncludeClient
)

$ErrorActionPreference = "Stop"
$mutexName = "Global\STS2RL_StartStack"
$startMutex = New-Object System.Threading.Mutex($false, $mutexName)
$mutexAcquired = $false

try {
    $mutexAcquired = $startMutex.WaitOne(0)
} catch {
    $mutexAcquired = $false
}

if (-not $mutexAcquired) {
    Write-Output "[stack] another start_stack invocation is already running, skipping"
    exit 0
}

function Write-Info {
    param([string]$Message)
    Write-Output "[stack] $Message"
}

function Read-Config {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Config not found: $Path"
    }
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Merge-ConfigObject {
    param(
        [object]$BaseObject,
        [object]$OverrideObject
    )
    if ($null -eq $BaseObject) { return $OverrideObject }
    if ($null -eq $OverrideObject) { return $BaseObject }
    if ($BaseObject -isnot [pscustomobject] -or $OverrideObject -isnot [pscustomobject]) {
        return $OverrideObject
    }

    $merged = [ordered]@{}
    foreach ($property in $BaseObject.PSObject.Properties) {
        $merged[$property.Name] = $property.Value
    }
    foreach ($property in $OverrideObject.PSObject.Properties) {
        if ($merged.Contains($property.Name) -and $merged[$property.Name] -is [pscustomobject] -and $property.Value -is [pscustomobject]) {
            $merged[$property.Name] = Merge-ConfigObject -BaseObject $merged[$property.Name] -OverrideObject $property.Value
        } else {
            $merged[$property.Name] = $property.Value
        }
    }
    [pscustomobject]$merged
}

function Resolve-ManagedPath {
    param(
        [string]$BasePath,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    [System.IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

function Get-PortOwner {
    param([int]$Port)
    try {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -First 1
    } catch {
        $null
    }
}

function Test-HttpEndpoint {
    param([string]$Url)
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    } catch {
        return $false
    }
}

function Get-MatchingProcess {
    param([string]$Pattern)
    if ([string]::IsNullOrWhiteSpace($Pattern)) { return $null }
    $needle = $Pattern.Replace("\\", "\")
    try {
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match '^python(w)?\.exe$' -and
                $_.CommandLine -and
                $_.CommandLine.Replace("\\", "\").IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
            } |
            Select-Object -First 1
    } catch {
        return $null
    }
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

    if (-not $Node) { return }
    if (-not $ForceEnabled -and -not $Node.enabled) {
        Write-Info "$Name disabled in config, skipping"
        return
    }

    $workdir = Resolve-ManagedPath -BasePath $DefaultRoot -Value ([string]$Node.workdir)
    $scriptPath = Resolve-ManagedPath -BasePath $DefaultRoot -Value ([string]$Node.script)
    $processMatch = [string]$Node.process_match
    $bindHost = [string]$Node.host
    $port = if ($null -ne $Node.port) { [int]$Node.port } else { $null }
    $healthPath = [string]$Node.health_path
    $healthUrl = if ($bindHost -and $port -and $healthPath) { "http://$bindHost`:$port$healthPath" } else { $null }

    if ($port) {
        $owner = Get-PortOwner -Port $port
        if ($owner) {
            if ($healthUrl -and (Test-HttpEndpoint -Url $healthUrl)) {
                Write-Info "$Name already running on port $port (pid=$owner)"
                return
            }
            throw "$Name expected port $port, but it is occupied by pid=$owner and health check failed"
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
    $pidLog = Join-Path $logDir "$Name.pid"

    $nodePythonExe = [string]$Node.python_executable
    if ([string]::IsNullOrWhiteSpace($nodePythonExe)) {
        $nodePythonExe = $PythonExe
    }

    $argList = @($scriptPath)
    if ($Node.args) {
        foreach ($item in $Node.args) {
            $argList += [string]$item
        }
    }

    $envBackup = @{}
    $effectiveEnv = [System.Collections.Specialized.OrderedDictionary]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($key in $SharedEnv.Keys) {
        $effectiveEnv[[string]$key] = [string]$SharedEnv[$key]
    }
    if ($Node.env) {
        foreach ($property in $Node.env.PSObject.Properties) {
            $effectiveEnv[[string]$property.Name] = [string]$property.Value
        }
    }

    $pythonDir = Split-Path -Parent $nodePythonExe
    $venvRoot = Split-Path -Parent $pythonDir
    if (Test-Path -LiteralPath $venvRoot) {
        $effectiveEnv["VIRTUAL_ENV"] = $venvRoot
        $existingPath = [Environment]::GetEnvironmentVariable("Path", "Process")
        if ([string]::IsNullOrWhiteSpace($existingPath)) {
            $effectiveEnv["Path"] = $pythonDir
        } else {
            $effectiveEnv["Path"] = "$pythonDir;$existingPath"
        }
    }

    $envBackup["PATH"] = [Environment]::GetEnvironmentVariable("PATH", "Process")
    $envBackup["Path"] = [Environment]::GetEnvironmentVariable("Path", "Process")
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $null, "Process")
    foreach ($key in $effectiveEnv.Keys) {
        $envBackup[[string]$key] = [Environment]::GetEnvironmentVariable([string]$key, "Process")
        [Environment]::SetEnvironmentVariable([string]$key, [string]$effectiveEnv[$key], "Process")
    }

    try {
        $process = Start-Process -FilePath $nodePythonExe `
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
    Set-Content -LiteralPath $pidLog -Value ([string]$process.Id) -Encoding UTF8

    if ($healthUrl) {
        Wait-Endpoint -Name $Name -Url $healthUrl | Out-Null
    }
}

$defaultConfigPath = Join-Path $PSScriptRoot "..\configs\runtime_stack.json"
$localConfigPath = Join-Path $PSScriptRoot "..\configs\runtime_stack.local.json"
if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $config = Read-Config -Path $ConfigPath
    $resolvedConfigPath = $ConfigPath
} else {
    $baseConfig = Read-Config -Path $defaultConfigPath
    if (Test-Path -LiteralPath $localConfigPath) {
        $localConfig = Read-Config -Path $localConfigPath
        $config = Merge-ConfigObject -BaseObject $baseConfig -OverrideObject $localConfig
        $resolvedConfigPath = "$defaultConfigPath + $localConfigPath"
    } else {
        $config = $baseConfig
        $resolvedConfigPath = $defaultConfigPath
    }
}

$configDir = Split-Path -Parent (Resolve-Path -LiteralPath $defaultConfigPath)
$st2rlRoot = Resolve-ManagedPath -BasePath $configDir -Value ([string]$config.projects.st2rl_root)
$sts2CliRoot = Resolve-ManagedPath -BasePath $configDir -Value ([string]$config.projects.sts2_cli_root)
$pythonExe = [string]$config.python_executable
if ([string]::IsNullOrWhiteSpace($pythonExe)) {
    $pythonExe = "python"
}

$sharedEnv = @{
    "PYTHONUNBUFFERED" = "1"
}
if (-not [string]::IsNullOrWhiteSpace([string]$config.projects.game_dir)) {
    $sharedEnv["STS2_GAME_DIR"] = [string]$config.projects.game_dir
}

Write-Info "Using config: $resolvedConfigPath"
Write-Info "Python executable: $pythonExe"

Start-ManagedProcess -Name "service" -Node $config.service -PythonExe $pythonExe -DefaultRoot $sts2CliRoot -SharedEnv $sharedEnv
Start-ManagedProcess -Name "dashboard" -Node $config.dashboard -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv
Start-ManagedProcess -Name "watchdog" -Node $config.watchdog -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv
Start-ManagedProcess -Name "session_supervisor" -Node $config.session_supervisor -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv

if ($IncludeClient -or $config.client.enabled) {
    Start-ManagedProcess -Name "client" -Node $config.client -PythonExe $pythonExe -DefaultRoot $st2rlRoot -SharedEnv $sharedEnv -ForceEnabled:$IncludeClient
} else {
    Write-Info "client disabled, skipping"
}

if ($mutexAcquired) {
    $startMutex.ReleaseMutex() | Out-Null
}
$startMutex.Dispose()
