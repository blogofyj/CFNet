import torch
# from test.attacks import *
from advertorch.attacks import *
import torchattacks
from torchattacks import BIM
from My_DeepFool import DeepFool
# attack parameters
fgsm = [0.25, 0.3, 0.35]
rfgsm = [0.25, 0.3, 0.35]
cw = [5, 10, 15]

# function for construct adversarial images
def add_adv(model, image, label, adv, i=0, default=False):
    # fast gradient sign method
    if adv == 'fgsm':
        if default:
            fgsm_attack = GradientSignAttack(model)
        else:
            # print('it do')
            fgsm_attack = GradientSignAttack(model, eps=0.3)
        adv_image = fgsm_attack(image, label)

    elif adv == 'l-bfgs':
        l_bfgs_attack = LBFGSAttack(model, 101)
        adv_image = l_bfgs_attack(image, label)

    elif adv == 'bim':
        bim_attack = BIM(model, eps=0.04)
        adv_image = bim_attack(image, label)

    elif adv == 'deepfool':
        deepfool = DeepFool(nb_candidate=101, max_iter=101)
        adv_image = deepfool.attack(model, image)
    # iterative fast gradient sign method
    elif adv == 'i-fgsm':
        ifgsm = LinfBasicIterativeAttack(model, eps=0.3)
        adv_image = ifgsm(image, label)
    # iterative least likely sign method
    # random fast gradient sign method
    elif adv == 'r-fgsm':
        alpha = 0.05
        data = torch.clamp(image + alpha * torch.empty(image.shape).normal_(mean=0,std=1).cuda(), min=0, max=1)
        if default:
            rfgsm_attack = GradientSignAttack(model, eps=0.3-alpha)
        else:
            rfgsm_attack = GradientSignAttack(model, eps=rfgsm[i]-alpha)
        adv_image = rfgsm_attack(data, label)
    # momentum iterative fast gradient sign method
    elif adv == 'mi-fgsm':
        mifgsm = MomentumIterativeAttack(model)
        adv_image = mifgsm(image, label)
    # projected gradient sign method
    elif adv == 'pgd':
        pgd = PGDAttack(model)
        adv_image = pgd(image, label)

    elif adv == 'pgd_n':
        print("hi")
        pgd = LinfPGDAttack(model, eps=0.08)
        adv_image = pgd(image)

    elif adv == 'pgd_t':
        pgd = LinfPGDAttack(model, eps=0.03, targeted=True)
        adv_image = pgd(image, label)

    # Carlini-Wagner attack
    elif adv == 'cw_n':
        cw_attack = CarliniWagnerL2Attack(model, 10, confidence=1, max_iterations=500, initial_const=1)
        # cw_attack = CarliniWagnerL2Attack(model, 101, confidence=1, max_iterations=500, initial_const=1)
        adv_image = cw_attack(image)

    elif adv == 'ddn_n':
        ddn_attack = DDNL2Attack(model)
        adv_image = ddn_attack(image)

    elif adv == 'aa_n':
        aan_attack = torchattacks.AutoAttack(model, eps=0.05, version='rand')
        # aan_attack = torchattacks.AutoAttack(model, eps=0.03, version='standard')
        adv_image = aan_attack(image, label)

    elif adv == 'jsma_t':
        # jsmat_attack = JacobianSaliencyMapAttack(model, num_classes=10)
        jsmat_attack = JacobianSaliencyMapAttack(model, num_classes=101)
        adv_image = jsmat_attack(image, label)

    # simba attack
    elif adv == 'single':
        single = SinglePixelAttack(model)
        adv_image = single(image, label)
    else:
        _, adv_image = image
        print('Did not perform attack on the images!')
    # if attack fails, return original
    if adv_image is None:
        adv_image = image
    if torch.cuda.is_available():
        image = image.cuda()
        adv_image = adv_image.cuda()

    return image, adv_image