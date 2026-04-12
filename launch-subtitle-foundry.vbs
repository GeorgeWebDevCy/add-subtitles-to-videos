Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
launcherScript = repoRoot & "\scripts\launch_windows_outside_venv.ps1"
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & launcherScript & """"

shell.CurrentDirectory = repoRoot
shell.Run command, 0, False
