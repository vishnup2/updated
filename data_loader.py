import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DentalSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        """
        Args:
            images_dir (str): Directory with all the dental images.
            masks_dir (str): Directory with all the segmentation masks.
            transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        
        # Load images and masks
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming mask is grayscale (L mode)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Example transformations (you can modify these depending on your needs)
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Converts to 1 channel float tensor
])

def get_dataloaders(batch_size, images_dir, masks_dir, shuffle=True, num_workers=4):
    dataset = DentalSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # Example usage
    images_dir = "images/"
    masks_dir = "masks/"
    batch_size = 8

    dataloader = get_dataloaders(batch_size, images_dir, masks_dir)

    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        # Do something with the images and masks