import os
from torch.autograd import Variable
import torch
from model import DCDiscriminator, DCGenerator

                
def to_var(tensor, cuda=True):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

    
def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def gan_checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, f'G_{iteration}.pkl')
    D_path = os.path.join(opts.checkpoint_dir, f'D_{iteration}.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def load_checkpoint(opts, iteration):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_path = os.path.join(opts.load, f'G_{iteration}.pkl')
    D_path = os.path.join(opts.load, f'D_{iteration}.pkl')

    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim, spectral_norm=opts.spectral_norm)
    D = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def gan_save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = to_data(generated_images)
    # save images in sample dir

def create_model(opts):
    """Builds the generators and discriminators.
    """
    ### GAN
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim, spectral_norm=opts.spectral_norm)
    D = DCDiscriminator(conv_dim=opts.d_conv_dim, spectral_norm=opts.spectral_norm)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')
    return G, D

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)
