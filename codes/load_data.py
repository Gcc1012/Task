
# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# from PIL import Image
# from torchvision import transforms

# class Classification(Dataset):
    
#     def __init__(self, root_dir,mode, transform=None):
        
#         self.transform = transform
#         self.root_dir = root_dir 
#         self.mode = mode
#         self.class_to_idx ={'cat':0, 'dog': 1}        #maps easily


#         self.image_paths =[]
#         self.labels =[]

#         print(f"Total images found: {len(self.image_paths)}")
#         print(f"Example path: {self.image_paths[:3]}")

#         for class_name in os.listdir(self.root_dir):
#             class_dir = os.path.join(self.root_dir, class_name)       #class_dir -->class_name  class_dir -->file_name
#             if not os.path.isdir(class_dir):
#                 continue
#             label = self.class_to_idx.get(class_name.lower())
#             if label is None:
#                 continue
#             for file_name in os.listdir(class_dir):
#                 if file_name.lower().endswith(('.jpg','png','jpeg')):
#                     self.image_paths.append(os.path.join(class_dir, file_name))
#                     self.labels.append(label)

        

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         label = self.labels[idx]
        
#         image = Image.open(image_path)

#         if self.transform:
#             image = self.transform(image)

#         return image,label

#     def get_loaders(self, batch_size=32):
#         train_loader = DataLoader(self, batch_size=batch_size, shuffle=self.mode == "train")
#         val_loader = DataLoader(self, batch_size=batch_size, shuffle=False)
#         return train_loader, val_loader

# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])



# dataset = Classification(root_dir="C:/Users/Gayatri/Documents/task/dataset", transform=transform, mode="train")

# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# for images, labels in train_loader:
#     print("Images shape:", images.shape)   # Should be [4, 3, 224, 224]
#     print("Labels:", labels)               # Should be [0, 1, 0, 1] for cat/dog
#     break


#
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image


class Classification(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir 
        self.class_to_idx = {'cat': 0, 'dog': 1}

        self.image_paths = []
        self.labels = []

        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx.get(class_name.lower())
            if label is None:
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224,224)),
     transforms.ToTensor()
])


dataset = Classification(root_dir="C:/Users/Gayatri/Documents/task/data", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


for images, labels in train_loader:
    print("Train batch - images shape:", images.shape)
    print("Train batch - labels:", labels)
    break