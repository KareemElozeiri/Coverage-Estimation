import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareRegressionLoss(nn.Module):
    """
    Edge-aware loss designed specifically for regression tasks where preserving
    sharp transitions (edges) is important, such as coverage maps with buildings.
    """
    def __init__(self, edge_weight=1.0, smoothness_weight=0.1):
        super(EdgeAwareRegressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.edge_weight = edge_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, outputs, targets):
        # Base regression loss (MSE)
        mse_loss = self.mse(outputs, targets)
        
        # Edge detection using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=outputs.dtype, device=outputs.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=outputs.dtype, device=outputs.device).view(1, 1, 3, 3)
        
        # Calculate gradients for both outputs and targets
        batch_size, channels, height, width = outputs.shape
        
        # Process each channel separately
        total_edge_loss = 0
        total_smoothness_loss = 0
        
        for c in range(channels):
            # Extract single channel
            pred_channel = outputs[:, c:c+1]
            target_channel = targets[:, c:c+1]
            
            # Calculate gradients
            pred_grad_x = F.conv2d(pred_channel, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred_channel, sobel_y, padding=1)
            target_grad_x = F.conv2d(target_channel, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_channel, sobel_y, padding=1)
            
            # Calculate gradient magnitudes
            pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
            
            # Create edge weight map - more weight at strong edges in target
            # This emphasizes preserving sharp changes where they exist in ground truth
            edge_weight_map = torch.exp(target_grad_mag * 5) - 1  # Exponential weighting for edges
            
            # Edge preservation loss - weighted by importance of edges
            edge_diff = torch.abs(pred_grad_mag - target_grad_mag)
            weighted_edge_loss = (edge_weight_map * edge_diff).mean()
            total_edge_loss += weighted_edge_loss
            
            # Piecewise smoothness loss - penalize gradients in non-edge areas
            # This encourages flat regions to stay flat
            non_edge_mask = torch.exp(-target_grad_mag * 5)  # Low weight at edges
            smoothness_term = (non_edge_mask * (torch.abs(pred_grad_x) + torch.abs(pred_grad_y))).mean()
            total_smoothness_loss += smoothness_term
        
        # Average over channels if multi-channel
        if channels > 0:
            total_edge_loss /= channels
            total_smoothness_loss /= channels
            
        # Final loss combines MSE with edge preservation and smoothness constraint
        total_loss = mse_loss + self.edge_weight * total_edge_loss + self.smoothness_weight * total_smoothness_loss
        
        return total_loss


class TotalVariationRegressionLoss(nn.Module):
    """
    Combination of MSE and Total Variation loss.
    TV loss encourages piecewise smoothness with sharp transitions,
    which is ideal for preserving building edges in regression tasks.
    """
    def __init__(self, tv_weight=0.1):
        super(TotalVariationRegressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.tv_weight = tv_weight
        
    def forward(self, outputs, targets):
        # MSE loss
        mse_loss = self.mse(outputs, targets)
        
        # Calculate edge importance map (high at edges in target)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=targets.dtype, device=targets.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=targets.dtype, device=targets.device).view(1, 1, 3, 3)
        
        # Get target gradients to identify where edges should be
        batch_size, channels, height, width = targets.shape
        edge_importance = torch.zeros_like(targets)
        
        for c in range(channels):
            target_channel = targets[:, c:c+1]
            grad_x = F.conv2d(target_channel, sobel_x, padding=1)
            grad_y = F.conv2d(target_channel, sobel_y, padding=1)
            edge_importance[:, c:c+1] = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # High weight at edges, low weight elsewhere
        edge_weights = torch.exp(edge_importance * 5) 
        
        # Calculate TV loss - differences between adjacent pixels
        diff_x = torch.abs(outputs[:, :, :, :-1] - outputs[:, :, :, 1:])
        diff_y = torch.abs(outputs[:, :, :-1, :] - outputs[:, :, 1:, :])
        
        # Weight the TV loss to encourage sharp transitions at actual edges
        # and smoothness elsewhere
        weight_x = edge_weights[:, :, :, :-1]
        weight_y = edge_weights[:, :, :-1, :]
        
        tv_loss = torch.mean(
            (1 - weight_x) * diff_x + weight_x * (1 - diff_x) +
            (1 - weight_y) * diff_y + weight_y * (1 - diff_y)
        )
        
        # Combined loss
        total_loss = mse_loss + self.tv_weight * tv_loss
        
        return total_loss


class StructuralEdgeRegressionLoss(nn.Module):
    """
    Regression loss that preserves structural edges using a combination of
    MSE, gradient similarity, and structural similarity (SSIM).
    Particularly effective for preserving sharp building boundaries.
    """
    def __init__(self, edge_weight=1.0, ssim_weight=0.5, window_size=11):
        super(StructuralEdgeRegressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        
    def gaussian_window(self, size, sigma=1.5):
        """Create a Gaussian window for SSIM calculation"""
        coords = torch.arange(size, dtype=torch.float)
        coords -= size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window
    
    def ssim(self, img1, img2, window_size=11):
        """Calculate SSIM between two tensors"""
        # Create window
        window = self.gaussian_window(window_size).to(img1.device, img1.dtype)
        window = window.expand(img1.shape[1], 1, window_size, window_size)
        
        # Calculations for SSIM
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    
    def forward(self, outputs, targets):
        # Basic MSE loss
        mse_loss = self.mse(outputs, targets)
        
        # Edge detection using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=outputs.dtype, device=outputs.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=outputs.dtype, device=outputs.device).view(1, 1, 3, 3)
        
        batch_size, channels, height, width = outputs.shape
        total_edge_loss = 0
        total_ssim = 0
        
        for c in range(channels):
            # Extract single channel
            pred_channel = outputs[:, c:c+1]
            target_channel = targets[:, c:c+1]
            
            # Calculate gradients
            pred_grad_x = F.conv2d(pred_channel, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred_channel, sobel_y, padding=1)
            target_grad_x = F.conv2d(target_channel, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_channel, sobel_y, padding=1)
            
            # Gradient magnitude 
            pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
            target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
            
            # Edge similarity loss
            grad_loss = F.l1_loss(pred_grad, target_grad)
            total_edge_loss += grad_loss
            
            # Structural similarity (SSIM)
            ssim_value = self.ssim(pred_channel, target_channel, self.window_size)
            total_ssim += ssim_value
        
        # Average over channels
        if channels > 0:
            total_edge_loss /= channels
            total_ssim /= channels
        
        # Total loss
        ssim_loss = 1 - total_ssim  # Convert to loss (1 - similarity)
        total_loss = mse_loss + self.edge_weight * total_edge_loss + self.ssim_weight * ssim_loss
        
        return total_loss