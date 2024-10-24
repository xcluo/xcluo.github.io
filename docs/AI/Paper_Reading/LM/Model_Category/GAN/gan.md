Generative Adversarial Network对抗生成模型，包含生成器和判断器两部分

=== "Generator"
    目的是使生成的结果更趋近真实结果，只更新生成器模型的参数，$\mathcal{L}_G = \text{D}\big(\text{G}(noise), y\_real\big)$

    ```python
    x_gen = G(z)
    D_G_z = D(x_gen)
    lossG = criterion(D_G_z, lab_real)
    ```
    > 由于生成器的输入为噪声，所以会随机生成参与训练的所有$x\_real$（定向生成需要固定seed）

=== "Discriminator"
    目的是能够准确识别生成的结果和真实结果，只更新判别器模型的参数，$\mathcal{L}_D = \text{D}\big(\text{G}(x\_real), y\_real\big) + \text{D}\big(\text{G}(noise), y\_fake\big)$

    ```python
    D_x = D(x_real)
    lossD_real = criterion(D_x, lab_real)

    x_gen = G(z).detach()
    D_G_z = D(x_gen)
    lossD_fake = criterion(D_G_z, lab_fake)

    lossD = lossD_real + lossD_fake
    ```
