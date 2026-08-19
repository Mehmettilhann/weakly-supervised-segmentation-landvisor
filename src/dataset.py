import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_points=20, ignore_index=255, transform=None):
        """
        Point-labeled dataset class for deep learning models.

        Args:
            image_dir: Index of the original satellite images.
            mask_dir: Index of fully labeled masks.
            num_points: Number of random points to select from each class in the image.
            ignore_index: Value of pixels to leave unlabeled.
            transform: Data augmentation operations.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_points = num_points
        self.ignore_index = ignore_index
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def _simulate_point_labels(self, mask):
        """"
        It takes the full mask and transforms it into a low-surveillance spot mask.
        """
        # Initially, we set all pixels to ignore_index (unlabeled).
        point_mask = np.full_like(mask, self.ignore_index, dtype=np.int64)

        # Identifying the existing classes within the masks.
        unique_classes = np.unique(mask)

        for cls in unique_classes:
            if cls == self.ignore_index:
                continue  # Skip the ignore_index class
        
            # Retrieving the coordinates of the pixels belonging to the class.
            y_coords, x_coords = np.where(mask == cls)
            num_available = len(y_coords)

            # Edge Case: To avoid errors if there are very few pixels of the target class in the image. 
            # Select the minimum value.
            points_to_sample = min(self.num_points, num_available)

            if points_to_sample > 0:
                random_indices = np.random.choice(num_available, points_to_sample, replace=False)
                
                sel_y = y_coords[random_indices]
                sel_x = x_coords[random_indices]

                # Assigning the class label to the selected points in the point mask.
                point_mask[sel_y, sel_x] = cls

        return point_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        if not img_name.endswith('.jpg'):
            return self.__getitem__((idx + 1) % len(self.images))
            
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('_sat.jpg', '_mask.png') 
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            return self.__getitem__((idx + 1) % len(self.images))
        
         
        image = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_img = Image.open(mask_path).convert("RGB").resize((512, 512), Image.NEAREST)
        
        image = np.array(image)
        mask_img = np.array(mask_img)
        
        
        mask = np.full((512, 512), self.ignore_index, dtype=np.int64)
        colors = {
            0: [0, 255, 255],   
            1: [255, 255, 0],   
            2: [255, 0, 255],   
            3: [0, 255, 0],     
            4: [0, 0, 255],     
            5: [255, 255, 255]  
        }
        
        for cls_idx, color in colors.items():
            match = np.all(mask_img == color, axis=-1)
            mask[match] = cls_idx
            
        point_mask = self._simulate_point_labels(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=point_mask)
            image = augmented['image']
            point_mask = augmented['mask']
            
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        point_mask = torch.tensor(point_mask, dtype=torch.long)
        
        return image, point_mask