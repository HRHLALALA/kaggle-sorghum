import os, timm
import argparse

from timm import model_entrypoint, get_model_default_value
from timm.models import split_model_name
from timm.models.hub import load_model_config_from_hf

os.environ['TORCH_HOME'] = "./pretrain"
from multiprocessing import Pool

def download(model_name):
    source_name, model_name = split_model_name(model_name)
    if source_name == 'hf_hub':
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)

    url = get_model_default_value(model_name, 'url')
    os.system(f"wget {url} -P {os.path.join(os.environ['TORCH_HOME'], 'hub/checkpoints')}")

def download_pretrained_weights(model_list):
    with Pool(args.num_threads) as p:
        p.map(download, model_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern",default="*nfnet*")
    parser.add_argument("-t", "--num_threads", default=os.cpu_count())
    args = parser.parse_args()
    model_list = timm.list_models(args.pattern, pretrained=True)
    if len(model_list) ==0:
        print("No model with pattern", args.pattern)
        exit()

    for i in model_list:
        print(i)
    cmd = input("Do you want to download these checkpoints?")
    if cmd.lower() == "y" or cmd.lower() == "yes":
        download_pretrained_weights(model_list)

