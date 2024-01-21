import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO 2017 train dataset
dataset = foz.load_zoo_dataset("coco-2017", split="train")

# Specify the directory where you want to export the dataset
export_dir = "C:/path/to/your/exported/dataset/train"  # Adjust this path as needed

# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth"  # This is the default field where FiftyOne stores labels
)

print(f"Dataset exported to {export_dir}")
    # This might include resizing images,
