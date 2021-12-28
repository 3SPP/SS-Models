import paddle
from ssm.models.seg import FarSeg
from ssm.models.cd import SiamUNet
from ssm.datasets import RSDataset
import ssm.datasets.transforms as T
from ssm.models.losses import MixedLoss, BCELoss, DiceLoss
from ssm.core import train

# DEBUG
import cv2


# model
x1 = paddle.randn([2, 3, 256, 256])
x2 = paddle.randn([2, 3, 256, 256])
model = SiamUNet(num_classes=2, siam=True, cat=True)
pred = model(x1, x2)
# model = FarSeg(num_classes=2)
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

# display
lens = len(train_dataset)
print(f"lens={lens}")
for idx, data in enumerate(train_dataset):
    if len(data) == 3:
        img1, img2, lab = data
        print(idx, img1.shape, img2.shape, lab.shape)
    elif len(data) == 2:
        img1, lab = data
        img2 = None
        print(idx, img1.shape, lab.shape)
    # cv2.imshow("img1", img2show(img1))
    # if img2 is not None:
    #     cv2.imshow("img2", img2show(img2))
    # cv2.imshow("lab", lab)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# train
lr = 3e-5
epochs = 2
batch_size = 2
iters = epochs * len(train_dataset) // batch_size

optimizer = paddle.optimizer.AdamW(lr, parameters=model.parameters())
losses = {}
losses["types"] = [MixedLoss([BCELoss(), DiceLoss()], [1, 1])]
losses["coef"] = [1]

train(
    model=model,
    train_dataset=train_dataset,
    optimizer=optimizer,
    save_dir="output",
    iters=iters,
    batch_size=batch_size,
    save_interval=1,
    log_iters=1,
    num_workers=0,
    losses=losses,
    use_vdl=True)