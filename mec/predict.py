from mec.builder import MEC
import torchvision.models as models

import monai.transforms as mt
from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference
    
def main(ckpt_path,data_path, save_path, arch='resnet50', n_channels=6, dim=2048, pred_dim=512):
    model = MEC(models.__dict__[arch], n_channels, dim, pred_dim)
    # ckpt = torch.load(ckpt_path)
    # model.load_state_dict(ckpt['state_dict'])
    model.eval()

    normalization_data = {'1_AF488': (682.0776, 1358.4601),
    '1_AF555': (169.55586, 209.92102),
    '1_AF647': (642.41504, 1808.3195),
    '1_AF750': (292.6707, 313.36597),
    '1_DAPI': (1510.9457, 2075.8213),
    '2_AF647': (1369.3427, 2853.1277)}

    scales_dict = {k:1 for k in normalization_data}

    normalize = [
        mt.LoadImaged(keys=normalization_data.keys()), 
        RandomMultiScaleCropd(keys = normalization_data.keys(), patch_shape=[224,224], selection_fn = lambda x: (x['1_DAPI']>0).mean()>0.1, scales_dict=scales_dict)
    ]

    for k, v in normalization_data.items():
        normalize.append(mt.NormalizeIntensityd(keys=[k], subtrahend = v[0], divisor = v[1]))
    normalize += [
        mt.ConcatItemsd(keys = normalization_data.keys(), name = 'img'),
    ]

    df = pd.read_csv(data_path)
    df = [row.to_dict() for i, row in df.iterrows()]

    predict_dataset = Dataset(df, transform=transforms)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, 
        shuffle=True,
        num_workers=workers,
        pin_memory=False, 
        sampler=None, 
        drop_last=True,
        persistent_workers=False
    )

    for im in tqdm.tqdm(predict_loader):
        with torch.no_grad():
            im = im['img'].cuda()
            out = sliding_window_inference(im, [224,224], model.inference_forward, overlap=0, progress=True)
        breakpoint()
