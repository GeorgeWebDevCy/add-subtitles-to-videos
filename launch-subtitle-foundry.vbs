Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = repoRoot & "\.venv\Scripts\pythonw.exe"
command = """" & pythonwPath & """ -m add_subtitles_to_videos"

shell.CurrentDirectory = repoRoot
shell.Run command, 0, False
