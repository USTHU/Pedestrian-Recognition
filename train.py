import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from PIL import Image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
import copy
from crfseg import CRF
import pydensecrf.densecrf as dcrf

from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, vmin=0, vmax=7)
    plt.show()


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    CLASSES = ["building", "farm", "forest", "river", "road", "grass", "other"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace("jpg", "png")) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(self.masks_fps[i])
        mask = np.array(mask)
        mask[mask == 255] = len(self.CLASSES)
        #mask = np.expand_dims(mask, axis=-1)

        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = mask.astype(int)

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        #albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0.1, p=1, border_mode=0, mask_value=7),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0, mask_value=7),
        albu.RandomCrop(height=256, width=256, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=None),
    ]
    return albu.Compose(_transform)


class MIoU(base.Metric):
    __name__ = "miou_score"

    def __init__(self, activation=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = base.Activation(activation)
        self.ignore_index = ignore_index

    def forward(self, y_pr, y_gt):
        # y_pr N*C*W*H
        # y_gt N*W*H

        class_num = y_pr.shape[1]
        y_pr = self.activation(y_pr)
        y_pr = torch.argmax(y_pr, dim=1)
        mask = (y_gt!=self.ignore_index)
        score = 0
        for i in range(class_num):
            pii = torch.sum((y_pr==i)*(y_gt==i)*mask).item()
            sum_pij = torch.sum((y_gt==i)*mask).item()
            sum_pji = torch.sum((y_pr==i)*mask).item()
            score += pii/(sum_pij+sum_pji-pii+1e-6)

        score /= class_num

        return torch.tensor(score)


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class MIoULoss(base.Loss):

    def __init__(self, activation=None, ignore_index=None, eps=1e-6):
        super(MIoULoss, self).__init__()
        self.activation = base.Activation(activation)
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pr, y_gt):
        """pr: N*C*W*H
            gt: N*W*H"""

        N, C = y_pr.shape[0], y_pr.shape[1]
        y_pr = self.activation(y_pr)

        #convert y_gt from label to one-hot
        y_gt = y_gt.cpu().numpy()
        y_gt_one_hot = [(y_gt == c) for c in range(C)]
        y_gt = np.stack(y_gt_one_hot, axis=1).astype('float32')
        y_gt = torch.tensor(y_gt, device=y_pr.device)

        total_loss = 0

        for i in range(C):
            intersection = torch.sum(y_gt[:, i, :, :] * y_pr[:, i, :, :])
            gt_sum = torch.sum(y_gt[:, i, :, :])
            pr_sum = torch.sum(y_pr[:, i, :, :])
            dice_loss = 1 - (2 * intersection + self.eps) / (gt_sum + pr_sum + self.eps) / N
            total_loss = total_loss + dice_loss

        return total_loss / C


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)



class Lovasz_Softmax(base.Loss):
    """Multi-class lovasz-softmax loss"""
    def __init__(self, activation=None, classes='present', per_image=False, ignore_index=None):
        """
        :param activation: activation function
        :param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average
        :param per_image: compute the loss per image instead of per batch
        :param ignore_index: void class labels
        :return: None
        """
        super(Lovasz_Softmax, self).__init__()
        self.activation = base.Activation(activation)
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_image = per_image

    def forward(self, probas, labels):
        """
        :param probas:[B, C, H, W] Variable
        :param labels: [B, H, W] ground turth
        :return:
        """
        probas = self.activation(probas)
        if self.per_image:
            loss = mean(
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore_index), classes=self.classes)
                for prob, lab in zip(probas, labels))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self.ignore_index), classes=self.classes)
        return loss


class CustomLoss(base.Loss):

    def __init__(self, activation=None, ignore_index=None, eps=1e-6):
        super(CustomLoss, self).__init__()
        self.activation = activation
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pr, y_gt):
        """pr: N*C*W*H
            gt: N*W*H"""

        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        lovasz_loss = Lovasz_Softmax(activation=self.activation, ignore_index=self.ignore_index)
        #miou_loss = MIoULoss(activation=self.activation, ignore_index=self.ignore_index, eps=self.eps)
        coeff = 0.5

        return coeff * ce_loss(y_pr, y_gt) + (1 - coeff) * lovasz_loss(y_pr, y_gt)


def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(3, 3), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    _, n_classes, h, w = probs.shape

    probs = probs[0]  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
    return np.expand_dims(preds, 0)


"""dataset directory"""
data_root = '../Remote_Sensing'
x_train_dir = os.path.join(data_root, 'train_data/img_train')
y_train_dir = os.path.join(data_root, 'train_data/lab_train')


"""model settings"""
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["building", "farm", "forest", "river", "road", "grass", "other"]
ACTIVATION = 'softmax2d'    # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
#model = torch.load('./fpn_combo_model.pth')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


"""train and validation dataset split"""
val_num = int(0.2*len(dataset))
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_num, val_num])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=True)
#
# # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
#
loss = Lovasz_Softmax(activation=ACTIVATION, per_image=False, ignore_index=len(CLASSES))
#loss = CustomLoss(activation=ACTIVATION, ignore_index=len(CLASSES), eps=1e-6)
#loss = smp.utils.losses.CrossEntropyLoss(ignore_index=len(CLASSES))
metrics = [
    MIoU(activation=ACTIVATION, ignore_index=len(CLASSES)),
]
#metrics = [
#    smp.utils.metrics.IoU(threshold=0.5),
#]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),#, momentum=0.9),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 40 epochs

max_score = 0

for i in range(0, 40):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    #do something (save model, change lr, etc.)
    if max_score < valid_logs['miou_score']:
        max_score = valid_logs['miou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# load best saved checkpoint
best_model = torch.load('./fpn_combo_model.pth')
# evaluate model on test set
# test_epoch = smp.utils.train.ValidEpoch(
#     model=best_model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
# )
#
# logs = test_epoch.run(valid_loader)

x_test_dir = os.path.join(data_root, 'img_testA')
test_img_fps = os.listdir(x_test_dir)
mask_save_dir = os.path.join(data_root, 'mask_testA')
if not os.path.isdir(mask_save_dir):
    os.mkdir(mask_save_dir)

test_processing_fn = get_preprocessing(preprocessing_fn)

mask_fps = os.listdir(mask_save_dir)
for mask_fp in mask_fps:
    mask = Image.open(os.path.join(mask_save_dir, mask_fp))
    pixels = np.array(mask)

color_palette = Image.open('../Remote_Sensing/train_data/lab_train/T000000.png').getpalette()

for i in range(len(test_img_fps)):
    if test_img_fps[i] == '.DS_Store':
        continue
    img_fp = os.path.join(x_test_dir, test_img_fps[i])
    image = cv2.imread(img_fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raw_image = copy.deepcopy(image)
    image = test_processing_fn(image=image)['image']

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    raw_img_ep = np.expand_dims(raw_image, axis=0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = base.Activation(ACTIVATION)(pr_mask)
    pr_mask = pr_mask.cpu().numpy()
    pr_mask = dense_crf(pr_mask,raw_img_ep)
    pr_mask = np.argmax(pr_mask, axis=1)
    pr_mask = pr_mask.squeeze().astype('uint8')

    #visualize(image=raw_image, mask=pr_mask)

    mask_name = test_img_fps[i].replace('jpg', 'png')
    save_path = os.path.join(mask_save_dir, mask_name)
    mask = Image.fromarray(pr_mask, mode='P')
    mask.putpalette(color_palette)
    mask.save(save_path)



for i in range(200, 300):
    n = np.random.choice(len(valid_dataset))

    image_vis = valid_dataset[n][0].astype('uint8')
    image, gt_mask = valid_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = torch.argmax(pr_mask, dim=1)
    pr_mask = pr_mask.squeeze().cpu().numpy()

    image_vis = np.transpose(image_vis, (1, 2, 0))
    image = np.transpose(image, (1, 2, 0))
    #gt_mask = np.transpose(gt_mask, (1, 2, 0))
    #pr_mask = np.transpose(pr_mask, (1, 2, 0))

    visualize(
        image=image,
        ground_truth_building=gt_mask,
        pred_mask=pr_mask,
    )

print("work done!")