; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "CogStat"
#define MyAppVersion "2.2"
#define MyAppPublisher "Attila Krajcsi"
#define MyAppURL "https://www.cogstat.org"
#define MySource "C:\Users\Attila\CogStat\cogstat_source\"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{9EE8E7AC-4AB1-4335-8F5F-0B3E31A03C99}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=CogStat_Windows_installer_{#MyAppVersion}
OutputDir=windows_installers
Compression=lzma
SetupIconFile="{#MySource}\cogstat\resources\CogStat.ico"
SolidCompression=yes
WizardImageFile="{#MySource}\cogstat\resources\CogStat logo.bmp"
WizardImageStretch=no
WizardImageBackColor=clWhite

[Languages]
; There are some unofficial language files, download them from http://www.jrsoftware.org/files/istrans/
Name: "bulgarian"; MessagesFile: "compiler:Languages\Bulgarian.isl"
Name: "croatian"; MessagesFile: "compiler:Languages\Croatian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "estonian"; MessagesFile: "compiler:Languages\Estonian.isl"
Name: "farsi"; MessagesFile: "compiler:Languages\Farsi.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "greek"; MessagesFile: "compiler:Languages\Greek.isl"
Name: "hebrew"; MessagesFile: "compiler:Languages\Hebrew.isl"
Name: "hungarian"; MessagesFile: "compiler:Languages\Hungarian.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "kazakh"; MessagesFile: "compiler:Languages\Kazakh.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "norwegian"; MessagesFile: "compiler:Languages\Norwegian.isl"
Name: "romanian"; MessagesFile: "compiler:Languages\Romanian.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "slovak"; MessagesFile: "compiler:Languages\Slovak.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "thai"; MessagesFile: "compiler:Languages\Thai.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 0,6.1

[Files]
Source: "{#MySource}run_cogstat_gui.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MySource}*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\Attila\AppData\Local\Programs\Python\Python38\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppName} {#MyAppVersion}"; Filename: "{app}\pythonw.exe"; WorkingDir: "{app}"; Parameters: """{app}\run_cogstat_gui.py"""; IconFilename: "{app}\cogstat\resources\CogStat.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName} {#MyAppVersion}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName} {#MyAppVersion}"; Filename: "{app}\pythonw.exe"; WorkingDir: "{app}"; Parameters: """{app}\run_cogstat_gui.py"""; IconFilename: "{app}\cogstat\resources\CogStat.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName} {#MyAppVersion}"; Filename: "{app}\pythonw.exe"; WorkingDir: "{app}"; Parameters: """{app}\run_cogstat_gui.py"""; IconFilename: "{app}\cogstat\resources\CogStat.ico"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\pythonw.exe"; WorkingDir: "{app}"; Parameters: """{app}\run_cogstat_gui.py"""; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall skipifsilent
