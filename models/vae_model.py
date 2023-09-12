import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_cond = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, c):
        h = F.leaky_relu(self.fc1(x))
        h_cond = F.relu(self.fc_cond(c))
        h = torch.cat([h, h_cond], dim=1)
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, cond_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = F.leaky_relu(self.fc1(zc))
        x = torch.sigmoid(self.fc2(h))
        return x


class ConditionVAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, latent_dim=64, output_dim=384, cond_dim=384):
        super(ConditionVAE, self).__init__()
        self.encoder_x = Encoder(input_dim, hidden_dim, latent_dim, cond_dim)
        self.encoder_c = Encoder(cond_dim, hidden_dim, latent_dim, 0)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, cond_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, c):

        mu_x, logvar_x = self.encoder_x(x, c)
        z = self.reparameterize(mu_x, logvar_x)
        x_recon = self.decoder(z, c)
        return x_recon, mu_x, logvar_x

    def generate(self, z, c):
        x_gen = self.decoder(z, c)
        return x_gen


def vae_loss(x, x_recon, mu, log_var):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


class ConditionVAE_1(nn.Module):
    def __init__(self, num_layers=5, input_dim=384, condition_dim=384, hidden_dim=128, latent_dim=64):
        super(ConditionVAE_1, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 定义编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim + condition_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        # 定义解码器
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, condition_dim)

        # 定义均值和方差线性层
        self.fc_mu = nn.Linear(input_dim + condition_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim + condition_dim, latent_dim)

        # 定义解码器输出层
        self.fc_out = nn.Linear(latent_dim + condition_dim, input_dim)

    def encode(self, x, c):
        # 将输入和条件向量连接起来
        input = torch.cat((x, c), dim=-1)

        # 使用编码器将输入编码为潜在变量
        encoded = self.encoder(input)

        # 计算潜在变量的均值和方差
        # encoded = encoded.mean(dim=0)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)

        # 返回潜在变量的均值和方差
        return mu, logvar

    def decode(self, z, c):
        # 使用解码器将潜在变量解码为输出
        decoded = self.decoder(z, c)

        # # 计算输出并返回
        # output = self.fc_out(decoded.mean(dim=0))
        return decoded

    def reparameterize(self, mu, logvar):
        # 从正态分布中采样潜在变量
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)

        # 从潜在变量中采样
        z = self.reparameterize(mu, logvar)

        # 解码潜在变量为输出
        output = self.decode(z, c)

        # 返回输出、均值和方差
        return output, mu, logvar


class ConditionVAE_2(nn.Module):
    def __init__(self, num_layers=5, input_dim=128, condition_dim=384, hidden_dim=128, latent_dim=64):
        super(ConditionVAE_2, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 定义编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim + condition_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        # 定义解码器
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, condition_dim)

        # 定义均值和方差线性层
        self.fc_mu = nn.Linear(input_dim + condition_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim + condition_dim, latent_dim)

        # 定义解码器输出层
        self.fc_out = nn.Linear(latent_dim + condition_dim, input_dim)

    def encode(self, x, c):
        # 将输入和条件向量连接起来
        input = torch.cat((x, c), dim=-1)

        # 使用编码器将输入编码为潜在变量
        encoded = self.encoder(input)

        # 计算潜在变量的均值和方差
        # encoded = encoded.mean(dim=0)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)

        # 返回潜在变量的均值和方差
        return mu, logvar

    def decode(self, z, c):
        # 使用解码器将潜在变量解码为输出
        decoded = self.decoder(z, c)

        # # 计算输出并返回
        # output = self.fc_out(decoded.mean(dim=0))
        return decoded

    def reparameterize(self, mu, logvar):
        # 从正态分布中采样潜在变量
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)

        # 从潜在变量中采样
        z = self.reparameterize(mu, logvar)

        # 解码潜在变量为输出
        output = self.decode(z, c)

        # 返回输出、均值和方差
        return output, mu, logvar


class ConditionVAE_wo(nn.Module):

    def __init__(self, num_layers=5, input_dim=384, hidden_dim=128, latent_dim=64):
        super(ConditionVAE_wo, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 定义编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )

        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(128, 384)
        )

        # 定义均值和方差线性层
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)

        # 定义解码器输出层
        self.fc_out = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        input = x
        encoded = self.encoder(input)

        # 计算潜在变量的均值和方差
        # encoded = encoded.mean(dim=0)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)

        # 返回潜在变量的均值和方差
        return mu, logvar

    def decode(self, z):
        # 使用解码器将潜在变量解码为输出
        decoded = self.decoder(z)
        return decoded

    def reparameterize(self, mu, logvar):
        # 从正态分布中采样潜在变量
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)

        # 从潜在变量中采样
        z = self.reparameterize(mu, logvar)

        # 解码潜在变量为输出
        output = self.decode(z)

        # 返回输出、均值和方差
        return output, mu, logvar
