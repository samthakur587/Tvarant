# third_party

`winsdk/` is a local extract of the Microsoft.Windows.SDK.CPP NuGet packages.
It is only needed on Windows machines that have MSVC but no system Windows SDK
(so `stddef.h` / UCRT headers are missing). It is gitignored.

To recreate:

```powershell
# from repo root
python -c "import urllib.request; urllib.request.urlretrieve('https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.CPP/10.0.26100.4654', 'sdk.cpp.nupkg')"
python -c "import urllib.request; urllib.request.urlretrieve('https://www.nuget.org/api/v2/package/Microsoft.Windows.SDK.CPP.x64/10.0.26100.4654', 'sdk.x64.nupkg')"
```

Then unzip the `.nupkg` files into `third_party/winsdk/sdk.cpp` and
`third_party/winsdk/sdk.x64`. `setup.py` will pick them up automatically.
