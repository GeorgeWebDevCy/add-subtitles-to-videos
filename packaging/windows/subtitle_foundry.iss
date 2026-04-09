#define AppName "Subtitle Foundry"
#define AppVersion "0.1.0"
#define AppPublisher "Subtitle Foundry"
#define AppExeName "SubtitleFoundry.exe"
#define RepoRoot "..\.."

[Setup]
AppId={{9C7B6A42-16A8-40CE-9B26-AE4AF8152713}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Subtitle Foundry
DefaultGroupName=Subtitle Foundry
OutputDir={#RepoRoot}\build\installer
OutputBaseFilename=subtitle-foundry-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#RepoRoot}\assets\branding\subtitle-foundry-icon.ico
WizardImageFile={#RepoRoot}\assets\branding\installer-sidebar.bmp
WizardSmallImageFile={#RepoRoot}\assets\branding\installer-small.bmp
UninstallDisplayIcon={app}\SubtitleFoundry.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#RepoRoot}\dist\SubtitleFoundry\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Subtitle Foundry"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{autodesktop}\Subtitle Foundry"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#AppExeName}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch Subtitle Foundry"; Flags: nowait postinstall skipifsilent
