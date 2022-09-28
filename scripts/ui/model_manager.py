from webui_streamlit import st

#other imports
import pandas as pd
from io import StringIO

# Temp imports 


# end of imports
#---------------------------------------------------------------------------------------------------------------

class PluginInfo():
    plugname = "model_manager"
    description = "Model Manager"
    isTab = False
    displayPriority = 0


def layout():
    #search = st.text_input(label="Search", placeholder="Type the name of the model you want to search for.", help="")

    csvString = f"""
                    ,Stable Diffusion v1.4            , ./models/ldm/stable-diffusion-v1               , https://huggingface.co/CompVis/stable-diffusion-v-1-4-original                  
                    ,GFPGAN v1.3                      , ./src/gfpgan/experiments/pretrained_models     , https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth                     
                    ,RealESRGAN_x4plus                , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth            
                    ,RealESRGAN_x4plus_anime_6B       , ./src/realesrgan/experiments/pretrained_models , https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth 
                    ,Waifu Diffusion v1.2             , ./content/models                                , https://huggingface.co/hakurei/waifu-diffusion
                    ,Waifu Diffusion v1.2 Pruned      , ./content/models                                , https://huggingface.co/crumb/pruned-waifu-diffusion
                    ,TrinArt Stable Diffusion v2      , ./content/models                                , https://huggingface.co/naclbit/trinart_stable_diffusion_v2
                    ,Stable Diffusion Concept Library , ./content/custom/sd-concepts-library             , https://huggingface.co/sd-concepts-library
                    ,AdaBins_nyu.pt and AdaBins_kitti.pt,./content/models                                , https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA
                    ,MiDas dpt_large                   , ./content/models                               , https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
                    ,MiDas dpt_hybrid                  , ./content/models                               , https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt
                    ,MiDas midas_v21_smal              , ./content/models                               , https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt
                    ,MiDas midas_21                    , ./content/models                               , https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt
                    ,SD CLIP 1.4                        , ./content/models                              , https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned/blob/main/sd-clip-vit-l14-img-embed_ema_only.ckpt
                    """
    colms = st.columns((1, 3, 5, 5))
    columns = ["№",'Model Name','Save Location','Download Link']

    # Convert String into StringIO
    csvStringIO = StringIO(csvString)
    df = pd.read_csv(csvStringIO, sep=",", header=None, names=columns)		

    for col, field_name in zip(colms, columns):
        # table header
        col.write(field_name)

    for x, model_name in enumerate(df["Model Name"]):
        col1, col2, col3, col4 = st.columns((1, 3, 4, 6))
        col1.write(x)  # index
        col2.write(df['Model Name'][x])
        col3.write(df['Save Location'][x])
        col4.write(df['Download Link'][x])    