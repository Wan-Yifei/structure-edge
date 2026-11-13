<#
.SYNOPSIS
    Register (or remove) the logon task that starts OpenD and the scheduler.

.DESCRIPTION
    After a restart nothing starts collecting on its own. On 2026-11-13 the
    machine restarted at 05:44 and again at 05:49; OpenD and the scheduler
    have no autostart entry, so nothing ran until the 08:04 logon -- 284 of
    that session's 649 minutes with no tape.

    This registers a scheduled task that runs scripts/start_stack.py when
    this user logs on. start_stack.py is what knows the ORDER: OpenD first,
    then wait until its API port accepts a connection, then the scheduler.

    WHY "at logon" AND NOT "at startup":
    Both OpenD and the scheduler draw windows. A task set to "run whether
    the user is logged on or not" runs in session 0, which has no desktop,
    and neither program can come up there. Fully unattended start therefore
    needs Windows auto-logon as well -- a separate decision, not made here.

    The console window is deliberately left visible. If OpenD is waiting for
    its password, the window says so; hiding it would turn a five-minute
    wait into a silent failure.

    Nothing here touches OpenD credentials.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Status
    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$Status,
    # Seconds to wait after logon before starting. The desktop, the network
    # stack and the tray are not all ready the instant winlogon hands over.
    [int]$DelaySeconds = 45
)

$ErrorActionPreference = 'Stop'

$TaskName = 'moomoo start stack'
$Repo     = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Script   = Join-Path $Repo 'scripts\start_stack.py'

function Find-Uv {
    $c = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $p = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path $p) { return $p }
    throw "uv.exe not found; this project runs everything through uv"
}

function Show-Status {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) { "not registered"; return }
    $i = $t | Get-ScheduledTaskInfo
    "name       : $($t.TaskName)"
    "state      : $($t.State)"
    "runs as    : $($t.Principal.UserId)  (logon type $($t.Principal.LogonType))"
    "action     : $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
    "working in : $($t.Actions[0].WorkingDirectory)"
    "last run   : $($i.LastRunTime)  result=$($i.LastTaskResult)"
}

if ($Status) { Show-Status; return }

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        "removed '$TaskName'"
    } else {
        "'$TaskName' was not registered"
    }
    return
}

if (-not (Test-Path $Script)) { throw "missing $Script" }
$uv = Find-Uv

$action = New-ScheduledTaskAction -Execute $uv `
    -Argument "run python `"$Script`"" -WorkingDirectory $Repo

# Interactive: it must land in the logged-on desktop session, because both
# programs it starts have windows.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = [System.Xml.XmlConvert]::ToString(
    [TimeSpan]::FromSeconds($DelaySeconds))

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# No idle condition, no "stop if it runs too long": start_stack waits up to
# five minutes for an OpenD login, and a task killed mid-wait would leave
# OpenD up with no scheduler, which is the worst of the three states.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description ("Starts OpenD, waits for its API port, then starts the " +
                  "scheduler. Order matters: the collectors connect to " +
                  "OpenD on startup.") | Out-Null

"registered '$TaskName'"
""
Show-Status
