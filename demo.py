import paddle
from ssm.models.backbones import HRNet_W18
from ssm.models.seg import OCRNet
from ssm.models.cd import SiamOCRNet
from ssm.datasets import RSDataset
import ssm.datasets.transforms as T
from ssm.models.losses import MixedLoss, BCELoss, DiceLoss
from ssm.core import train, predict

# DEBUG
import cv2


# model
x1 = paddle.randn([2, 3, 256, 256])
x2 = paddle.randn([2, 3, 256, 256])
model = SiamOCRNet(num_classes=2, backbone=HRNet_W18(in_channels=3), 
                   backbone_indices=[0], siam=True, cat=True)
pred = model(x1, x2)
# model = OCRNet(num_classes=2, backbone=HRNet_W18(in_channels=3), backbone_indices=[0])
# pred = model(x1)
print(pred[0].shape)

# data
def img2show(img):
    img = img.transpose([1, 2, 0]).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

train_transforms = [
    T.RandomHorizontalFlip(),
    T.RandomRotation(),
    T.Resize(target_size=(256, 256))
]
train_dataset = RSDataset(
    transforms=train_transforms,
    dataset_root='DataSet',
    num_classes=2,
    mode='train',
    # work='seg',
    work='cd',
    # file_path='DataSet/train_list.txt',  # seg / block
    # file_path='DataSet/train_list_2.txt',  # seg / big_map
    # file_path='DataSet/train_list_3.txt',  # cd / block
    file_path='DataSet/train_list_4.txt',  # cd / big_map
    separator=' ',
    # big_map=False
    big_map=True
)

val_transforms = [
    T.Resize(target_size=(256, 256))
]
val_dataset = RSDataset(
    transforms=val_transforms,
    dataset_root='DataSet',
    num_classes=2,
    mode='val',
    # work='seg',
    work='cd',
    # file_path='DataSet/train_list.txt',  # seg / block
    # file_path='DataSet/train_list_2.txt',  # seg / big_map
    file_path='DataSet/train_list_3.txt',  # cd / block
    # file_path='DataSet/train_list_4.txt',  # cd / big_map
    separator=' ',
    big_map=False
    # big_map=True
)

infer_dataset = RSDataset(
    transforms=val_transforms,
    dataset_root='DataSet',
    num_classes=2,
    mode='infer',
    # work='seg',
    work='cd',
    # file_path='DataSet/train_list_i.txt',  # seg / block
    file_path='DataSet/train_list_3_i.txt',  # cd / block
    separator=' ',
    big_map=False
    # big_map=True
)

# display train datas
lens = len(train_dataset)
print(f"lens={lens}")
# for idx, data in enumerate(train_dataset):
#     if len(data) == 3:
#         img1, img2, lab = data
#         print(idx, img1.shape, img2.shape, lab.shape)
#     elif len(data) == 2:
#         img1, lab = data
#         img2 = None
#         print(idx, img1.shape, lab.shape)
#     # cv2.imshow("img1", img2show(img1))
#     # if img2 is not None:
#     #     cv2.imshow("img2", img2show(img2))
#     # cv2.imshow("lab", lab)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# train
lr = 3e-5
epochs = 300
batch_size = 4
iters = epochs * len(train_dataset) // batch_size

optimizer = paddle.optimizer.AdamW(lr, parameters=model.parameters())
losses = {}
losses["types"] = [MixedLoss([BCELoss(), DiceLoss()], [1, 1])] * 2
losses["coef"] = [1, 0.4]

train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir="output",
    iters=iters,
    batch_size=batch_size,
    save_interval=iters // 5,
    log_iters=1,
    num_workers=0,
    losses=losses,
    use_vdl=True
)

# predict
predict(
    model,
    "output/best_model/model.pdparams",
    infer_dataset
)