import paddle
from ssm.models.seg import FarSeg
from ssm.models.cd import SiamUNet
from ssm.datas import RSDataset
import ssm.datas.transforms as T


# DEBUG
import cv2

# # seg
# data = paddle.randn((1, 3, 256, 256), dtype="float32")
# model = FarSeg()
# pred = model(data)
# print(pred[0].shape)


# # cd
# x1 = paddle.randn([2, 3, 256, 256])
# x2 = paddle.randn([2, 3, 256, 256])
# model = SiamUNet(num_classes=2, siam=True, cat=True)
# pred = model(x1, x2)
# print(pred[0].shape)


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
    work='cd',
    file_path='DataSet/train_list_3.txt',
    # file_path='DataSet/train_list_2.txt',
    separator=' ',
    big_map=False  # True
)

lens = len(train_dataset)
print(f"lens={lens}")
for idx, (img1, img2, lab) in enumerate(train_dataset):
    if img2 is not None:
        print(idx, img1.shape, img2.shape, lab.shape)
    else:
        print(idx, img1.shape, lab.shape)
    cv2.imshow("img1", img2show(img1))
    if img2 is not None:
        cv2.imshow("img2", img2show(img2))
    cv2.imshow("lab", lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()