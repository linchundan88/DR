#source activate tensorflow2.2

#cd ~/dlp/PycharmProjects/ROP/RPC

python RPC_server_single_class.py 0 5000 &  # image quality
python RPC_server_single_class.py 1 5001 &  # left right eye
python RPC_server_single_class.py 2 5002 &  # stage
python RPC_server_single_class.py 3 5003 &  # hemorhrage

python RPC_server_deep_shap.py 0 5100 &  # class type,gpu_no  port no


#sudo lsof -i:5000
