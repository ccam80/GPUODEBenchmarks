# Shared dataset-key helper for the Windows runners.
#
# Emits a key "<os>_<gpu>" identifying the current machine so benchmark output
# files can be additively populated across machines. The GPU name comes from
# nvidia-smi (the single source of truth shared by every framework) and is
# sanitised identically everywhere: tokenise on non-alphanumeric characters, drop
# the "NVIDIA"/"GeForce" vendor words, and join the rest with '-'.
# e.g. "NVIDIA GeForce RTX 2060 SUPER" -> "RTX-2060-SUPER".

function Get-DatasetKey {
    $platform = [System.Environment]::OSVersion.Platform
    if ($IsWindows -or
        $env:OS -eq 'Windows_NT' -or
        $platform -eq [System.PlatformID]::Win32NT) {
        $os = 'windows'
    }
    elseif ($IsMacOS) { $os = 'macos' }
    elseif ($IsLinux) { $os = 'linux' }
    else { $os = 'unknown' }

    $raw = ''
    try {
        $raw = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null |
                Select-Object -First 1)
    } catch { }
    if ($null -eq $raw) { $raw = '' }

    $tokens = ($raw -split '[^A-Za-z0-9]+') |
              Where-Object { $_ -ne '' -and $_ -ne 'NVIDIA' -and $_ -ne 'GeForce' }
    if ($tokens) { $gpu = ($tokens -join '-') } else { $gpu = 'unknown-gpu' }

    return "${os}_${gpu}"
}

# When run directly (not dot-sourced) print the key so batch scripts can capture it.
if ($MyInvocation.InvocationName -ne '.') {
    Get-DatasetKey
}
