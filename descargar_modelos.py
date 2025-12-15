import layoutparser as lp
import shutil
import os

list_models={"HJDataset_faster":"lp://HJDataset/faster_rcnn_R_50_FPN_3x/config","HJDataset_mask":"lp://HJDataset/mask_rcnn_R_50_FPN_3x/config","HJDataset_retina":"lp://HJDataset/retinanet_R_50_FPN_3x/config","PubLayNet_faster":"lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config","PubLayNet_mask":"lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config","PubLayNet_mask_X":"lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config","PrimaLayout_mask":"lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config","NewspaperNavigator_faster":"lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config","TableBank_faster":"lp://TableBank/faster_rcnn_R_50_FPN_3x/config","TableBank_faster_X":"lp://TableBank/faster_rcnn_R_101_FPN_3x/config","MFD_faster":"lp://MFD/faster_rcnn_R_50_FPN_3x/config"}

carpeta_descarga="/home/diego/.torch/iopath_cache/s"#carpeta usual donde se descargan los archivos
reiniciar_carpeta_torch="/home/diego/.torch/"
shutil.rmtree(reiniciar_carpeta_torch)
nombre_descarga_modelo="model_final.pth?dl=1"#nombre usual que se le asigna al descargar el archivo model
nombre_descarga_config="config.yml?dl=1"#nombre usual que se asigna al descargar el archivo config
carpeta_destino="/home/diego/Documentos/proyecto_pam/models"
for model in list_models:
    modelo=lp.Detectron2LayoutModel(config_path=list_models[model])
    if os.path.exists(carpeta_descarga):
        lista=os.listdir(carpeta_descarga)
        for carpeta in lista:
            subcarpeta=os.join(carpeta_descarga,carpeta)
            sublista=os.listdir(subcarpeta)
            for archivo in sublista:
                if archivo==nombre_descarga_modelo:
                    path_archivo=os.join(subcarpeta,archivo)
                    carpeta_destino_1=os.join(carpeta_destino,f"{model}")
                    if not os.path.exists(carpeta_destino_1):
                        os.makedirs(carpeta_destino_1)
                    shutil.copy(path_archivo,os.join(carpeta_destino_1,f"{model}.pth"))
                    print(f"Modelo ya descargado: {model} en carpeta {carpeta+"/"+archivo}")
                elif archivo==nombre_descarga_config:
                    path_archivo=os.join(subcarpeta,archivo)
                    carpeta_destino_2=os.join(carpeta_destino,f"{model}")
                    if not os.path.exists(carpeta_destino_2):
                        os.makedirs(carpeta_destino_2)
                    shutil.copy(path_archivo,os.join(carpeta_destino_2,f"{model}.yml"))
                    print(f"Config ya descargado: {model} en carpeta {carpeta+"/"+archivo}")
    else: 
        print("No existe la carpeta de descarga")
        print("La última descarga fue:",model)
        break
    shutil.rmtree(reiniciar_carpeta_torch)