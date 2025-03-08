from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image

class FloodDataset(Dataset):
    def __init__(self, root_folder):
        self.image_dir = os.path.join(root_folder, "Image")
        self.mask_dir = os.path.join(root_folder, "Mask")

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.image_files) == len(self.mask_files), "Số lượng ảnh và mask không khớp"
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            assert img_file.split(".")[0] == mask_file.split(".")[0], "Tên file ảnh và mask không khớp"

        # Define transformation for image and mask 
        self.image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return {
            "image": image,
            "mask": mask
        }