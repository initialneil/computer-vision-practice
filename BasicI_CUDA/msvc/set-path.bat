REM This is a bat for setting required system variables. Please change specific path to your own.
set TOOLKITS=C:\toolkits
setx TOOLKITS %TOOLKITS%

setx OPENCV_3_4_3_CUDA_10_0 "%TOOLKITS%\opencv-3.4.3-contrib-cuda10.0"
setx OPENCV_3_4_3_CPU "%TOOLKITS%\opencv-3.4.3-prebuilt\build"
setx BOOST_1_68_0 "%TOOLKITS%\boost_1_68_0"
setx LIB_PROTOBUF_3_0_0 "%TOOLKITS%\protobuf-3.0.0"

setx GFLAGS "%TOOLKITS%\gflags-2.1.2"
setx GLOG "%TOOLKITS%\glog"
setx EIGEN_3_3_5 "%TOOLKITS%\eigen-3.3.5"

setx LIB_SOLVER "%TOOLKITS%\SolverSuite"
setx CERES "%TOOLKITS%\SolverSuite\ceres-solver-1.13.0"

setx PCL_1_8_1 "%TOOLKITS%\pcl-1.8.1-cuda10.0"
setx OPENNI2 "%TOOLKITS%\OpenNI2"
setx ZLIB "%TOOLKITS%\zlib"

setx LIB_OPENGL "%TOOLKITS%\opengl-utils"
setx LIB_JPEG "%TOOLKITS%\libjpeg-turbo64"
setx LIB_JSONCPP "%TOOLKITS%\jsoncpp-2018.10.3"
setx JSONCPP_1_9_2 "%TOOLKITS%\jsoncpp-1.9.2\vs2015-x64"

setx LIB_STREAM "%TOOLKITS%\stream-utils"

setx libAzureKinectSDK "%TOOLKITS%\Azure\1.4.3"

pause
