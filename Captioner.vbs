' Captioner - silent launcher.
'
' Double-clicking (or the desktop shortcut) runs this instead of the .bat, so
' no console window flashes. If the virtualenv does not exist yet, it hands
' off to run_captioner.bat *visibly* once, because first-run setup needs to
' show download progress; every later launch is fully silent.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = root & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pyw) Then
    sh.Run """" & root & "\run_captioner.bat""", 1, False
Else
    sh.CurrentDirectory = root
    sh.Run """" & pyw & """ -m ui.dashboard", 0, False
End If
