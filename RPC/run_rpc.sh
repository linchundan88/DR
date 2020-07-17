#source activate tensorflow2.2

#cd ~/PycharmProjects/DR/RPC

python RPC_server_single_class.py 1 6001 &  # DR
python RPC_server_single_class.py 11 6011 &  # image quality
python RPC_server_single_class.py 12 6012 &  # left right eye

python RPC_server_CAM.py 1 6101 CAM & # DR
python RPC_server_CAM.py 1 6102 grad_cam & # DR
python RPC_server_CAM.py 1 6103 gradcam_plus & # DR

python RPC_server_deep_shap.py 1 0 6200 &  # class type, gpu_no  port no

#sudo lsof -i:5000
