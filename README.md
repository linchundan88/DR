
1.License
This software is under the GNU GENERAL PUBLIC LICENSE V2.

2.Project background
This project is part of the automated Diabetic Retinopathy screening platform.
The platform contains three parts: training and validation, RPC service and web application.
This project contains the first two parts of the platform.
And the third part(web application) can be found at the project "DR_Web".

3.Code structure
Please see requirements.txt for dependency libraries.
The LIBS sub-directory includes shared libraries.
The RPC sub-directory provide RPC services, which can be called by the web application
The run_rpc.sh is the script to start RPC services. 
Except that, every sub-directory, including DR grading, image quanlity and left right eye correspond to one classifier.  




