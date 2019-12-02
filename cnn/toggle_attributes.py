import torch
import matplotlib.pyplot as plt
import os
import utils
import vae
from collections import OrderedDict
import argparse

DEFAULT_ATTR_DICT = OrderedDict({
    '5_o_Clock_Shadow': 0,
    'Arched_Eyebrows': 0,
    'Attractive': 0,
    'Bags_Under_Eyes': 0,
    'Bald': 0,
    'Bangs': 0,
    'Big_Lips': 0,
    'Big_Nose': 0,
    'Black_Hair': 0,
    'Blond_Hair': 1,
    'Blurry': 0,
    'Brown_Hair': 0,
    'Bushy_Eyebrows': 0,
    'Chubby': 0,
    'Double_Chin': 0,
    'Eyeglasses': 0,
    'Goatee': 0,
    'Gray_Hair': 0,
    'Heavy_Makeup': 0,
    'High_Cheekbones': 0,
    'Male': 1,
    'Mouth_Slightly_Open': 1,
    'Mustache': 0,
    'Narrow_Eyes': 0,
    'No_Beard': 0,
    'Oval_Face': 0,
    'Pale_Skin': 0,
    'Pointy_Nose': 0,
    'Receding_Hairline': 0,
    'Rosy_Cheeks': 0,
    'Sideburns': 0,
    'Smiling': 1,
    'Straight_Hair': 0,
    'Wavy_Hair': 0,
    'Wearing_Earrings': 0,
    'Wearing_Hat': 0,
    'Wearing_Lipstick': 0,
    'Wearing_Necklace': 0,
    'Wearing_Necktie': 0,
    'Young': 0,
})

def toggle_attributes(img_path, model_path, save_dir=None, attr_dict=None):
    """ helper function to allow altering conditional encoding
        and to visualize decoded output """
    # create save location
    save_dir = save_dir if save_dir else 'toggled_attributes'
    if os.path.exists(save_dir):
        raise RuntimeError(f'Save location {save_dir} already exists')
    else:
        os.mkdir(save_dir)

    # generate attr tensor
    attr_dict = attr_dict if attr_dict else DEFAULT_ATTR_DICT
    print('Base attributes:')
    print(attr_dict)
    attr = torch.tensor(list(attr_dict.values()), dtype=torch.float)
    attr = attr.view(1, -1)

    # load model and image
    ae = vae.load_model(model_path = model_path)
    ae.eval()
    img = utils.load_img(img_path).unsqueeze(0)

    with torch.no_grad():
        img = img.to(vae.DEVICE)
        for i, (key, value) in enumerate(attr_dict.items()):
            prev = int(attr[0, i]) # save current value
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))

            # turn attribute i off and plot
            attr[0, i] = 0
            out, mu, sig = ae(img, attr)
            ax[0].imshow(out[0].permute(1,2,0).numpy())
            ax[0].set_title(f'{key} = False')

            # turn attribute i on and plot
            attr[0, i] = 1
            out, mu, sig = ae(img, attr)
            ax[1].imshow(out[0].permute(1,2,0).numpy())
            ax[1].set_title(f'{key} = True')

            fig.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{key}.png'))
            plt.close(fig)
            attr[0, i] = prev # return to previous value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Editing conditional CelebA attributes for VAE model')
    parser.add_argument('--img_path', type=str, default=None, help='path to image to encode and alter')
    parser.add_argument('--model_path', type=str, default=None, help='path to trained model')
    parser.add_argument('--save_path', type=str, default=None, help='location to save examples')
    args = parser.parse_args()
    if not args.img_path:
        raise RuntimeError('Must provide --img_path')
    if not args.model_path:
        raise RuntimeError('Must provide --model_path')
    toggle_attributes(args.img_path, args.model_path, save_dir=args.save_path)
