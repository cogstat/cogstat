#!usr/bin/env bash
export IDENTITY="THE NAME OF A Developer ID Application IDENTITY (MUST HAVE APPLE DEVELOPER PAID ACCOUNT AND THE PRIVATE KEY INSTALLED IN KEYCHAIN"
export OPTIONS="--force --timestamp --entitlements entitlements.plist --options=runtime"
export APP="dist/Cogstat.app"

echo "Signing Python Framework"
export FRAMEWORK="${APP}/Contents/Frameworks/Python.framework/Versions/3.11/Python"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} ${FRAMEWORK}

echo "Signing Qt Frameworks"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickControls2.framework/Versions/5/QtQuickControls2"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickParticles.framework/Versions/5/QtQuickParticles"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtRemoteObjects.framework/Versions/5/QtRemoteObjects"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick3DRender.framework/Versions/5/QtQuick3DRender"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtDesigner.framework/Versions/5/QtDesigner"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtNfc.framework/Versions/5/QtNfc"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick3DAssetImport.framework/Versions/5/QtQuick3DAssetImport"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickWidgets.framework/Versions/5/QtQuickWidgets"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick3DRuntimeRender.framework/Versions/5/QtQuick3DRuntimeRender"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtHelp.framework/Versions/5/QtHelp"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtPrintSupport.framework/Versions/5/QtPrintSupport"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtGui.framework/Versions/5/QtGui"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtDBus.framework/Versions/5/QtDBus"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtWebSockets.framework/Versions/5/QtWebSockets"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick3DUtils.framework/Versions/5/QtQuick3DUtils"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickTemplates2.framework/Versions/5/QtQuickTemplates2"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtPositioningQuick.framework/Versions/5/QtPositioningQuick"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtLocation.framework/Versions/5/QtLocation"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtXml.framework/Versions/5/QtXml"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtSerialPort.framework/Versions/5/QtSerialPort"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtWebView.framework/Versions/5/QtWebView"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick.framework/Versions/5/QtQuick"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtCore.framework/Versions/5/QtCore"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQml.framework/Versions/5/QtQml"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtWebChannel.framework/Versions/5/QtWebChannel"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtMultimedia.framework/Versions/5/QtMultimedia"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQmlWorkerScript.framework/Versions/5/QtQmlWorkerScript"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtOpenGL.framework/Versions/5/QtOpenGL"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtMacExtras.framework/Versions/5/QtMacExtras"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtTest.framework/Versions/5/QtTest"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtWidgets.framework/Versions/5/QtWidgets"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtPositioning.framework/Versions/5/QtPositioning"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtBluetooth.framework/Versions/5/QtBluetooth"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuick3D.framework/Versions/5/QtQuick3D"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickShapes.framework/Versions/5/QtQuickShapes"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQuickTest.framework/Versions/5/QtQuickTest"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtNetwork.framework/Versions/5/QtNetwork"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtXmlPatterns.framework/Versions/5/QtXmlPatterns"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtSvg.framework/Versions/5/QtSvg"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtMultimediaWidgets.framework/Versions/5/QtMultimediaWidgets"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtQmlModels.framework/Versions/5/QtQmlModels"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtSensors.framework/Versions/5/QtSensors"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtTextToSpeech.framework/Versions/5/QtTextToSpeech"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtSql.framework/Versions/5/QtSql"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/Resources/lib/python3.11/PyQt5/Qt5/lib/QtConcurrent.framework/Versions/5/QtConcurrent"


export ZIP_NAME="python311.zip"
export ORIGINAL_ZIP_DIR="${APP}/Contents/Resources/lib"

export PYTHON_ZIP="${ORIGINAL_ZIP_DIR}/${ZIP_NAME}"

export TEMP_DIR="/tmp"
export UNZIP_DIR="python311"

echo "Get copy of unsigned zip file"

cp -p ${PYTHON_ZIP} ${TEMP_DIR}

echo "Unzip it"
/usr/bin/ditto -x -k "${TEMP_DIR}/${ZIP_NAME}" "${TEMP_DIR}/${UNZIP_DIR}"

find "${TEMP_DIR}/${UNZIP_DIR}/PIL/.dylibs" -iname '*.dylib' |
while read libfile; do
    # echo "Signing $libfile"
    arch -x86_64 codesign --sign "${IDENTITY}" "${libfile}" ${OPTIONS}
done;

echo "Remove old temp copy zip file"
rm -vrf "${TEMP_DIR}/${ZIP_NAME}"

echo "recreate zip file"
/usr/bin/ditto -c -k "${TEMP_DIR}/${UNZIP_DIR}" "${TEMP_DIR}/${ZIP_NAME}"

echo "Move signed zip back"
cp -p "${TEMP_DIR}/${ZIP_NAME}" ${ORIGINAL_ZIP_DIR}

echo "Sign libraries"

find "${APP}" -iname '*.so' -or -iname '*.dylib' |
while read libfile; do
    # echo "Signing $libfile";
    arch -x86_64 codesign --sign "${IDENTITY}" "${libfile}" ${OPTIONS} ;
done;

arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/MacOS/python"
arch -x86_64 codesign --sign "${IDENTITY}" ${OPTIONS} "${APP}/Contents/MacOS/cogstat"

echo "Signing done"
echo "Verifying..."

arch -x86_64 codesign --verify -dvvv "${APP}"
arch -x86_64 codesign --verify --deep --verbose=4 "${APP}"
