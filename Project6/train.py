import torch.optim as optim
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from utils import create_dir, create_model, sample_noise, to_var, gan_save_samples, gan_checkpoint
from dataset import get_emoji_loader


def gan_training_loop(dataloader, test_dataloader, opts):

    G, D = create_model(opts)

    g_params = G.parameters()  # Get generator parameters
    d_params = D.parameters()  # Get discriminator parameters

    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr * 2., [opts.beta1, opts.beta2])

    train_iter = iter(dataloader)

    test_iter = iter(test_dataloader)

    # Get some fixed data from domains X for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

    iter_per_epoch = len(train_iter)
    total_train_iters = opts.train_iters

    losses = {"iteration": [], "D_fake_loss": [], "D_real_loss": [], "G_loss": []}


    try:
        for iteration in range(1, opts.train_iters + 1):

            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)

            real_images, real_labels = train_iter.next()
            real_images, real_labels = to_var(real_images), to_var(real_labels).long().squeeze()

            # ones = Variable(torch.Tensor(real_images.shape[0]).float().cuda().fill_(1.0), requires_grad=False)
            loss = torch.nn.MSELoss()

            for d_i in range(opts.d_train_iters):
                d_optimizer.zero_grad()

                # FILL THIS IN
                # 1. Compute the discriminator loss on real images
                D_out_real = D(real_images).type(torch.FloatTensor)
                ones = torch.ones(size=real_labels.size()).type(torch.FloatTensor)
                ones.requires_grad = False
                zeros = torch.zeros(size=real_labels.size()).type(torch.FloatTensor)
                zeros.requires_grad = False

                D_real_loss = loss(D_out_real, zeros)
                D_real_loss = D_real_loss.type(torch.FloatTensor)* (1 / D_out_real.shape[0])
                real_labels = real_labels.type(torch.FloatTensor)

                # 2. Sample noise
                noise = sample_noise(batch_size=opts.batch_size, dim=opts.noise_size)

                # 3. Generate fake images from the noise
                fake_images = G(noise)

                # 4. Compute the discriminator loss on the fake images
                D_out_fake = D(fake_images).type(torch.FloatTensor)
                ones = torch.ones(size=D_out_fake.shape).type(torch.FloatTensor)
                ones.requires_grad = False
                D_fake_loss = loss(D_out_fake, ones)
                D_fake_loss = D_fake_loss.type(torch.FloatTensor) * (1 / D_out_fake.shape[0])

                # --------------------------
                # 5. Compute the total discriminator loss
                D_total_loss = D_fake_loss + D_real_loss
                D_total_loss = D_total_loss.type(torch.FloatTensor)

                D_total_loss.backward()
                d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            noise = sample_noise(batch_size=opts.batch_size, dim=opts.noise_size)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            D_out_fake1 = D(fake_images).type(torch.FloatTensor)
            zeros = torch.zeros(size=D_out_fake1.shape).type(torch.FloatTensor)
            zeros.requires_grad = False
            G_loss = loss(D_out_fake1, zeros)
            G_loss = G_loss.type(torch.FloatTensor) * (1 / D_out_fake1.shape[0])

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                losses['iteration'].append(iteration)
                losses['D_real_loss'].append(D_real_loss.item())
                losses['D_fake_loss'].append(D_fake_loss.item())
                losses['G_loss'].append(G_loss.item())
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                    iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                gan_save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                gan_checkpoint(iteration, G, D, opts)

    except KeyboardInterrupt:
        print('Exiting early from training.')

    plt.figure()
    plt.plot(losses['iteration'], losses['D_real_loss'], label='D_real')
    plt.plot(losses['iteration'], losses['D_fake_loss'], label='D_fake')
    plt.plot(losses['iteration'], losses['G_loss'], label='G')
    plt.legend()
    plt.savefig(os.path.join(opts.sample_dir, 'losses.png'))
    plt.close()




def train(opts):

    dataloader_X, test_dataloader_X = get_emoji_loader(opts.train_path, opts.test_path, opts.batch_size, opts.image_size)
    
    # Set the random seed manually for reproducibility.
    # ......

    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    gan_training_loop(dataloader_X, test_dataloader_X, opts)



if __name__ == '__main__':
    # Load config file
    # args = 
    train(args)
