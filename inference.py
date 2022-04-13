from tqdm import tqdm
import torch
import os
os.environ['TORCH_HOME'] = "./pretrain"
import pandas as pd

@torch.no_grad()
def inference(model, dataloader):
    model.cuda()
    model.eval()

    predictions = []
    for batch in tqdm(test_loader):
        image = batch['image'].cuda()
        outputs = model(image)
        preds = outputs.detach().cpu()
        predictions.append(preds.argmax(1))
            
    tmp = predictions[0]
    for i in range(len(predictions) - 1):
        tmp = torch.cat((tmp, predictions[i+1]))
        
    predictions = [unique_cultivars[pred] for pred in tmp]
    sub = pd.read_csv(PATH + "sample_submission.csv")
    sub["cultivar"] = predictions
    sub.to_csv('submission.csv', index=False)
    print(sub.head())
    
if __name__ == "__main__":
    TEST_DIR = PATH + 'test/'
    model = CustomEffNet(model_name=CFG.model_name, pretrained=False)
    checkpoint = "./logs/tf_efficientnet_b3_ns/version_0/checkpoints/last.ckpt"
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    sub = pd.read_csv(PATH + "sample_submission.csv")
    sub.head()
    
    sub["file_path"] = sub["filename"].apply(lambda image: TEST_DIR + image)
    sub["cultivar_index"] = 0
    sub.head()
    
    test_dataset = SorghumDataset(sub, get_transform('valid'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)
    
    inference(model,test_loader)
