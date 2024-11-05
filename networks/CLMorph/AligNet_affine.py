import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import LeakyReLU

def conv(dim=2):
    if dim == 2:
        return nn.Conv2d
    return nn.Conv3d

def avpool(dim=2):
    if dim == 2:
        return nn.AdaptiveAvgPool2d
    return nn.AdaptiveAvgPool3d

def convolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return conv(dim=dim)(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1):
    return nn.Sequential(LeakyReLU(leakyr_slope), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))

def averagePooling(out_channels=1, dim=2):
    return avpool(dim=dim)(output_size=out_channels)


class AligNetAffineStem(nn.Module):
    """
    AligNet affine stem. This is the first part of the AligNet network. A multi-layer convolutional network that calculates the affine transformation parameters.

    Args:
        dim (int): Dimension of the input image.
        channels (int): Number of channels in the first convolution.
        flow_multiplier (float): Multiplier for the flow output.
        im_size (int): Size of the input image.
        in_channels (int): Number of channels in the input image.
    """
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=1):
        super(AligNetAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim             = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1   = convolveLeakyReLU(in_channels, channels, 3, 2, dim=self.dim)
        self.conv2   = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3   = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4   = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5   = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6   = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # Applying global average pooling to have the same feature map dimensions
        self.avpool =  averagePooling(out_channels=1, dim=dim)
        
        # I'm assuming that the image's shape is like (im_size, im_size, im_size) 
        self.last_conv_size = im_size // (self.channels * 4)
        self.fc_loc         = nn.Sequential(
            nn.Linear(1024, 1024), #(512 * self.last_conv_size**dim, 2048), while applying avp
            #nn.Linear(36864, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512), 
            #nn.Linear(2048, 1024), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256), 
            #nn.Linear(1024, 256), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1))
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        """
        Identity Matrix
            | 1 0 0 0 |
        I = | 0 1 0 0 |
            | 0 0 1 0 |
        """
        if dim == 3:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.create_flow = self.cr_flow


    def cr_flow(self, theta, size):
        shape = size[2:]
        flow  = F.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow
    
    
    def wr_flow(self, theta, size):
        flow = F.affine_grid(theta, size, align_corners=False)  # batch x 512 x 512 x 2
        if self.dim == 2:
            flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
            flow = flow.permute(0, 4, 1, 2, 3)
        return flow
    
    
    def rev_affine(self, theta, dim=2):
        b = theta[:, :, dim:]
        inv_w = torch.inverse(theta[:, :dim, :dim])
        neg_affine = torch.cat([inv_w, -inv_w@b], dim=-1)
        return neg_affine
    
    
    def neg_flow(self, theta, size):
        neg_affine = self.rev_affine(theta, dim=self.dim)
        return self.create_flow(neg_affine, size)


    def forward(self, fixed, moving):
        """
        Calculate the affine transformation parameters

        Returns:
            flow: the flow field
            theta: dict, with the affine transformation parameters
        """
        # Fixed Image
        x1   = self.conv1(fixed)    #  16 x 256 x 256       ----- # 16 x 96 x 96 x 104      -----A # 16 x 96 x 80 x 128
        x2   = self.conv2(x1)       #  32 x 128 x 128       ----- # 32 x 48 x 48 x  52      -----A # 32 x 48 x 40 x  64
        x3   = self.conv3(x2)       #  64 x  64 x  64       ----- # 64 x 24 x 24 x  26      -----A # 64 x 24 x 20 x  32
        x3_1 = self.conv3_1(x3)     #  64 x  64 x  64       ----- # 64 x 24 x 24 x  26      -----A # 64 x 24 x 20 x  32
        x4   = self.conv4(x3_1)     # 128 x  32 x  32       ----- #128 x 12 x 12 x  13      -----A #128 x 12 x 10 x  16
        x4_1 = self.conv4_1(x4)     # 128 x  32 x  32       ----- #128 x 12 x 12 x  13      -----A #128 x 12 x 10 x  16
        x5   = self.conv5(x4_1)     # 256 x  16 x  16       ----- #256 x  6 x  6 x   7      -----A #256 x  6 x  5 x   8
        x5_1 = self.conv5_1(x5)     # 256 x  16 x  16       ----- #256 x  6 x  6 x   7      -----A #256 x  6 x  5 x   8
        x6   = self.conv6(x5_1)     # 512 x   8 x   8       ----- #512 x  3 x  3 x   4      -----A #512 x  3 x  3 x   4
        x6_1 = self.conv6_1(x6)     # 512 x   8 x   8       ----- #512 x  3 x  3 x   4      -----A #512 x  3 x  3 x   4
        x7   = self.avpool(x6_1)    # 512 x   1 x   1       ----- #512 x  1 x  1 x   1      -----A #512 x  1 x  1 x   1
        
        # Moving Image
        y1   = self.conv1(moving)   #  16 x 256 x 256       ----- # 16 x 96 x 96 x 104      -----A # 16 x 96 x 80 x 128
        y2   = self.conv2(y1)       #  32 x 128 x 128       ----- # 32 x 48 x 48 x  52      -----A # 32 x 48 x 40 x  64
        y3   = self.conv3(y2)       #  64 x  64 x  64       ----- # 64 x 24 x 24 x  26      -----A # 64 x 24 x 20 x  32
        y3_1 = self.conv3_1(y3)     #  64 x  64 x  64       ----- # 64 x 24 x 24 x  26      -----A # 64 x 24 x 20 x  32
        y4   = self.conv4(y3_1)     # 128 x  32 x  32       ----- #128 x 12 x 12 x  13      -----A #128 x 12 x 10 x  16
        y4_1 = self.conv4_1(y4)     # 128 x  32 x  32       ----- #128 x 12 x 12 x  13      -----A #128 x 12 x 10 x  16
        y5   = self.conv5(y4_1)     # 256 x  16 x  16       ----- #256 x  6 x  6 x   7      -----A #256 x  6 x  5 x   8
        y5_1 = self.conv5_1(y5)     # 256 x  16 x  16       ----- #256 x  6 x  6 x   7      -----A #256 x  6 x  5 x   8
        y6   = self.conv6(y5_1)     # 512 x   8 x   8       ----- #512 x  3 x  3 x   4      -----A #512 x  3 x  3 x   4
        y6_1 = self.conv6_1(y6)     # 512 x   8 x   8       ----- #512 x  3 x  3 x   4      -----A #512 x  3 x  3 x   4
        y7   = self.avpool(y6_1)    # 512 x   1 x   1       ----- #512 x  1 x  1 x   1      -----A #512 x  1 x  1 x   1

        # Concatenation of fixed and moving
        #tensors    = get_same_dim_tensors([x6_1, y6_1],y6_1.size(-1), -1) # Uncomment if needed!
        #xy = torch.cat((x6_1, y6_1), dim=1)  
        xy = torch.cat((x7, y7), dim=1)  # 1024 x   1 x   1       ----- #1024 x  1 x  1 x   1      -----A #1024 x  1 x  1 x   1
        
        # Affine transformation
        xs = xy.view(-1, 512*2)  #512 * self.last_conv_size ** self.dim) # When applying avp
        #xs = xy.view(1, 36864) #512 * self.last_conv_size ** self.dim)
        if self.dim == 3:
            theta = self.fc_loc(xs).view(-1, 3, 4)
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        flow = self.create_flow(theta, moving.size())
        # theta: the affine param
        return flow, {'theta': theta}


if __name__ == "__main__":
    model = AligNetAffineStem(dim=3, im_size=16)
    x     = torch.randn(1, 1, 192, 192, 208)
    y     = torch.randn(1, 1, 190, 160, 256)
    af_out= model(x, y)
    print('len output: ', len(af_out))
    print('Aligned Output shape: ', af_out[0].shape)
    print('Flow shape: ', af_out[1]['theta'].shape)


'''import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import LeakyReLU

def conv(dim=2):
    if dim == 2:
        return nn.Conv2d
    return nn.Conv3d

def avpool(dim=2):
    if dim == 2:
        return nn.AdaptiveAvgPool2d
    return nn.AdaptiveAvgPool3d

def convolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return conv(dim=dim)(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1):
    return nn.Sequential(LeakyReLU(leakyr_slope), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))

def averagePooling(out_channels=1, dim=2):
    return avpool(dim=dim)(output_size=out_channels)


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, dim=2):
        super(AttentionLayer, self).__init__()
        if dim == 2:
            self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
            self.conv2 = nn.Conv3d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * attention


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, feature_channels, out_channels, dim=2):
        super(FeaturePyramidNetwork, self).__init__()
        if dim == 2:
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(channels, out_channels, kernel_size=1)
                for channels in feature_channels
            ])
            self.smooth_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in feature_channels
            ])
        else:
            self.lateral_convs = nn.ModuleList([
                nn.Conv3d(channels, out_channels, kernel_size=1)
                for channels in feature_channels
            ])
            self.smooth_convs = nn.ModuleList([
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in feature_channels
            ])

    def forward(self, features):
        pyramid_features = []
        for i in range(len(features) - 1, -1, -1):
            if i == len(features) - 1:
                pyramid_features.append(self.lateral_convs[i](features[i]))
            else:
                upsampled_feature = F.interpolate(pyramid_features[-1], size=features[i].shape[2:], mode='trilinear', align_corners=False)
                lateral_feature = self.lateral_convs[i](features[i])
                
                # Ensure the sizes match exactly
                if upsampled_feature.shape != lateral_feature.shape:
                    # Crop or pad the upsampled_feature to match lateral_feature
                    diff = [lateral_feature.size(dim) - upsampled_feature.size(dim) for dim in range(2, len(lateral_feature.size()))]
                    
                    # If any dimension is smaller, pad; if larger, crop
                    for dim in range(2, len(diff) + 2):
                        if diff[dim - 2] > 0:  # Padding needed
                            upsampled_feature = F.pad(upsampled_feature, (0, diff[dim - 2]), mode='replicate')
                        elif diff[dim - 2] < 0:  # Cropping needed
                            upsampled_feature = upsampled_feature[..., :lateral_feature.size(dim)]
                
                # Add upsampled feature and lateral feature
                pyramid_features.append(self.smooth_convs[i](lateral_feature + upsampled_feature))

        pyramid_features.reverse()
        return pyramid_features



# Modified AligNetAffineStem with attention and FPN
class AligNetAffineStem(nn.Module):
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=1):
        super(AligNetAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim             = dim

        # Feature extractor
        self.conv1   = convolveLeakyReLU(in_channels, channels, 3, 2, dim=self.dim)
        self.conv2   = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3   = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4   = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5   = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6   = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # Attention Layers for both Fixed and Moving
        self.attention_fixed  = AttentionLayer(32 * channels, dim=dim)
        self.attention_moving = AttentionLayer(32 * channels, dim=dim)

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            feature_channels=[4 * channels, 8 * channels, 16 * channels, 32 * channels], 
            out_channels=256,
            dim=self.dim
        )

        # Update the fully connected layer input size for FPN features
        self.fc_loc = nn.Sequential(
            nn.Linear(36864 + 4*256, 2048),  # Adjust input size based on FPN
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1))
        )
        self.fc_loc[-1].weight.data.zero_()
        if dim == 3:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.create_flow = self.cr_flow

    def cr_flow(self, theta, size):
        shape = size[2:]
        flow  = F.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow
    
    
    def wr_flow(self, theta, size):
        flow = F.affine_grid(theta, size, align_corners=False)  # batch x 512 x 512 x 2
        if self.dim == 2:
            flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
            flow = flow.permute(0, 4, 1, 2, 3)
        return flow
    
    
    def rev_affine(self, theta, dim=2):
        b = theta[:, :, dim:]
        inv_w = torch.inverse(theta[:, :dim, :dim])
        neg_affine = torch.cat([inv_w, -inv_w@b], dim=-1)
        return neg_affine
    
    
    def neg_flow(self, theta, size):
        neg_affine = self.rev_affine(theta, dim=self.dim)
        return self.create_flow(neg_affine, size)


    def adaptive_avg_pool(self, feature):
        """Performs 2D or 3D adaptive average pooling based on input dimension."""
        if self.dim == 2:
            return F.adaptive_avg_pool2d(feature, 1).view(feature.size(0), -1)  # Global pooling for 2D
        else:
            return F.adaptive_avg_pool3d(feature, 1).view(feature.size(0), -1)  # Global pooling for 3D


    def forward(self, fixed, moving):
        # Fixed Image
        x1   = self.conv1(fixed)
        x2   = self.conv2(x1)
        x3   = self.conv3(x2)
        x3_1 = self.conv3_1(x3)
        x4   = self.conv4(x3_1)
        x4_1 = self.conv4_1(x4)
        x5   = self.conv5(x4_1)
        x5_1 = self.conv5_1(x5)
        x6   = self.conv6(x5_1)
        x6_1 = self.conv6_1(x6)

        # Moving Image
        y1   = self.conv1(moving)
        y2   = self.conv2(y1)
        y3   = self.conv3(y2)
        y3_1 = self.conv3_1(y3)
        y4   = self.conv4(y3_1)
        y4_1 = self.conv4_1(y4)
        y5   = self.conv5(y4_1)
        y5_1 = self.conv5_1(y5)
        y6   = self.conv6(y5_1)
        y6_1 = self.conv6_1(y6)

        # Apply Attention
        x6_1 = self.attention_fixed(x6_1)
        y6_1 = self.attention_moving(y6_1)

        # Concatenate fixed and moving
        xy = torch.cat((x6_1, y6_1), dim=1)

        # FPN on fixed image
        fpn_features = self.fpn([x3_1, x4_1, x5_1, x6_1])

        # Apply adaptive pooling on FPN features and concatenate
        fpn_pooled = torch.cat([self.adaptive_avg_pool(fpn_feature) for fpn_feature in fpn_features], dim=1)

        print("Shape of xy after flattening:", xy.view(xy.size(0), -1).shape)
        print("Shape of FPN pooled features:", fpn_pooled.shape)

        # Concatenate the FPN features with the xy features
        combined_features = torch.cat((xy.view(xy.size(0), -1), fpn_pooled), dim=1)
        print("Shape of combined features:", combined_features.shape)

        # Pass through fully connected layers to get affine transformation parameters
        theta = self.fc_loc(combined_features).view(-1, 3, 4) if self.dim == 3 else self.fc_loc(combined_features).view(-1, 2, 3)
        flow  = self.create_flow(theta, moving.size())

        return flow, {'theta': theta}


if __name__ == "__main__":
    model = AligNetAffineStem(dim=3, im_size=16)
    x     = torch.randn(1, 1, 192, 160, 256)
    y     = torch.randn(1, 1, 192, 160, 256)
    af_out= model(x, y)
    print('len output: ', len(af_out))
    print('Aligned Output shape: ', af_out[0].shape)
    print('Flow shape: ', af_out[1]['theta'].shape)'''