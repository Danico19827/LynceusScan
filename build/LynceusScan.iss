; SPDX-License-Identifier: GPL-3.0-or-later
; LynceusScan per-user installer (Inno Setup 6).
;
; Build from the repo root AFTER `pyinstaller build\LynceusScan.spec`:
;   & "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" build\LynceusScan.iss
;
; Per-user install (%LocalAppData%, no admin prompt). User data
; (%APPDATA%\LynceusScan: projects, extensions, consent) is never touched
; by install/uninstall, so updates preserve it.
;
; NOTE: keep MyAppVersion in sync with lynceus/__init__.py::__version__.

#define MyAppName "LynceusScan"
#define MyAppVersion "0.1.0b4"
#define MyAppPublisher "Taritolay, Nicolas Daniel"
#define MyAppURL "https://github.com/Danico19827/LynceusScan"
#define MyAppExeName "LynceusScan.exe"

[Setup]
AppId={{3B4E7A1C-9F2D-4B6A-8E5C-1A2B3C4D5E6F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=..\dist-installer
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
SetupIconFile=..\assets\icon.ico
WizardImageFile=..\assets\wizard_image.png
WizardSmallImageFile=..\assets\wizard_small.png
UninstallDisplayIcon={app}\{#MyAppExeName}
DisableProgramGroupPage=yes

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\LynceusScan\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; .lynx project association (per-user, no admin required).
; Fully removed on uninstall (the app is gone; nothing may linger).
Root: HKCU; Subkey: "Software\Classes\.lynx"; ValueType: string; ValueName: ""; ValueData: "{#MyAppName}.Project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\{#MyAppName}.Project"; ValueType: string; ValueName: ""; ValueData: "LynceusScan Project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\{#MyAppName}.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKCU; Subkey: "Software\Classes\{#MyAppName}.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
; Zero-trace uninstall: user data, sessions, preferences and any runtime
; leftovers inside the install dir are all removed, so a reinstall starts
; factory-fresh (default theme, English, no recents, T&C gate again).
Type: filesandordirs; Name: "{app}"

[CustomMessages]
spanish.KeepUserDataPrompt=¿Conservar los datos de usuario (extensiones, sesiones y preferencias)?%n%nElegí No para una desinstalación sin rastros: la próxima instalación arranca de fábrica.
english.KeepUserDataPrompt=Keep user data (extensions, sessions and preferences)?%n%nChoose No for a zero-trace uninstall: the next install starts factory-fresh.

[Code]
var
  KeepUserData: Boolean;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  // Default = keep (safe). Silent uninstalls never prompt: keep as well.
  KeepUserData := True;
  if not UninstallSilent() then
  begin
    if MsgBox(CustomMessage('KeepUserDataPrompt'), mbConfirmation, MB_YESNO) = IDNO then
      KeepUserData := False;
  end;
  Log(Format('LynceusScan: InitializeUninstall silent=%d keep=%d cmd=%s', [Ord(UninstallSilent()), Ord(KeepUserData), GetCmdTail()]));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  // User-data wipe runs here (in-process globals are valid), never via
  // Check: on [UninstallDelete] entries: those are not evaluated there.
  if (CurUninstallStep = usUninstall) and (not KeepUserData) then
  begin
    Log('LynceusScan: wiping user data');
    DelTree(ExpandConstant('{userappdata}\LynceusScan'), True, True, True);
    DelTree(ExpandConstant('{localappdata}\LynceusScan'), True, True, True);
    RegDeleteKeyIncludingSubkeys(HKCU, 'Software\LynceusScan');
  end;
end;
